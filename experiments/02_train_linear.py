import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os.path
from imodels.util.extract import extract_marginal_curves
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from imodels import MarginalShrinkageLinearModelRegressor
import pmlb

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import joblib
import imodels
from imodels import TreeGAMClassifier
from sklearn.base import RegressorMixin, ClassifierMixin
import inspect
import cache_save_utils
import warnings
from imodels.util.data_util import DSET_KWARGS


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args ######################
    parser.add_argument(
        "--dataset_name", type=str, default="heart", help="name of dataset"
    )
    parser.add_argument(
        "--y_train_noise_std",
        type=float,
        default=0.0,
        help="standard deviation of noise",
    )
    parser.add_argument(
        "--collinearity_factor",
        type=float,
        default=0,
        help="factor to make covariates more collinear",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="fraction of data to use for training",
    )

    # training misc args ######################
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results", "test"),
        help="directory for saving",
    )

    # modeling args ######################
    parser.add_argument(
        "--est_marginal_name",
        type=str,
        default="ridge",
        help="model class to use for marginal. 'None' for None.",
    )
    parser.add_argument(
        "--est_main_name",
        type=str,
        default="ridge",
        help="model class to use for main effects. 'None' for None.",
    )
    parser.add_argument(
        "--use_marginal_sign_constraint",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to constrain main effects to be same sign as marginal effects",
    )
    parser.add_argument(
        "--use_marginal_divide_by_d",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to divide marginal effects by d",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="alpha to use for regularization",
    )
    parser.add_argument(
        "--elasticnet_ratio",
        type=float,
        default=0.5,
        help="ratio of l1 to l2 regularization for elastic net",
    )

    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    return parser


def make_covariates_more_collinear(X, collinearity_factor=0.5, seed=1):
    """Make covariates more collinear by adding a linear combination of them to each other"""
    X = X.copy()
    n, d = X.shape
    rng = np.random.default_rng(seed)
    for i in range(d):
        random_feature_weights = rng.uniform(low=-1, high=1, size=d)
        random_feature_weights[i] = 0
        X[:, i] = (1 - collinearity_factor) * X[:, i] + collinearity_factor * np.dot(
            X, random_feature_weights
        )
    return X


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )

    # split data
    if args.dataset_name in DSET_KWARGS:
        X, y, feat_names = imodels.get_clean_dataset(**DSET_KWARGS[args.dataset_name])
    elif args.dataset_name in pmlb.dataset_names:
        X, y, feat_names = imodels.get_clean_dataset(
            args.dataset_name, data_source="pmlb"
        )
    # remove any rows with nan
    idxs_nan = np.isnan(X).any(axis=1) | np.isnan(y)
    X = X[~idxs_nan]
    y = y[~idxs_nan]

    if args.collinearity_factor > 0:
        X = make_covariates_more_collinear(
            X, collinearity_factor=args.collinearity_factor, seed=args.seed
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1 - args.train_frac,
        random_state=args.seed,
    )

    # optionally add noise to the data
    if args.y_train_noise_std > 0:
        y_train = y_train.astype(float) + np.random.normal(
            0, args.y_train_noise_std, size=y_train.shape
        )

    # z-score X and y
    mx = StandardScaler()
    my = StandardScaler()
    X_train = mx.fit_transform(X_train)
    X_test = mx.transform(X_test)
    y_train = my.fit_transform(y_train.reshape(-1, 1)).squeeze()
    y_test = my.transform(y_test.reshape(-1, 1)).squeeze()

    # alphas = (0.1, 1, 10, 100, 1000, 10000)  # (0.1, 1, 10, 100, 1000, 10000)
    # alphas = (0.1, 1, 10, 100, 1000, 10000)  # (0.1, 1, 10, 100, 1000, 10000)
    m = MarginalShrinkageLinearModelRegressor(
        random_state=args.seed,
        est_marginal_name=args.est_marginal_name,
        est_main_name=args.est_main_name,
        alphas=args.alpha,
        marginal_divide_by_d=args.use_marginal_divide_by_d,
        marginal_sign_constraint=args.use_marginal_sign_constraint,
        elasticnet_ratio=args.elasticnet_ratio,
    )

    m.fit(X_train, y_train)

    # check roc auc score
    if isinstance(m, ClassifierMixin):
        r["roc_auc_train"] = metrics.roc_auc_score(
            y_train, m.predict_proba(X_train)[:, 1]
        )
        r["acc_train"] = metrics.accuracy_score(y_train, m.predict(X_train))
        r["roc_auc_test"] = metrics.roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        r["acc_test"] = metrics.accuracy_score(y_test, m.predict(X_test))
    else:
        r["r2_train"] = metrics.r2_score(y_train, m.predict(X_train))
        r["mse_train"] = metrics.mean_squared_error(y_train, m.predict(X_train))
        r["r2_test"] = metrics.r2_score(y_test, m.predict(X_test))
        r["mse_test"] = metrics.mean_squared_error(y_test, m.predict(X_test))

    lin = m.est_main_
    r["coef"] = lin.coef_
    r["alpha"] = lin.alpha_


    # save data stuff
    r['n_samples'] = X_train.shape[0]
    r['n_features'] = X_train.shape[1]

    # save results
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    joblib.dump(m, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
