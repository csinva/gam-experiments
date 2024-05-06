from imodels.util.data_util import DSET_KWARGS, DSET_CLASSIFICATION_KWARGS, DSET_REGRESSION_KWARGS, DSET_CLASSIFICATION_MULTITASK_KWARGS
import warnings
import cache_save_utils
import inspect
from sklearn.base import RegressorMixin, ClassifierMixin
from imodels import TreeGAMClassifier
import imodels
import joblib
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
import os.path
from imodels.util.extract import extract_marginal_curves
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.preprocessing import StandardScaler
import pmlb
from imodelsx import KANClassifier, KANRegressor, KANGAMClassifier, KANGAMRegressor
import imodels.algebraic.gam_multitask
from imodels.util.ensemble import ResidualBoostingRegressor, SimpleBaggingRegressor
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args ######################
    parser.add_argument(
        "--dataset_name", type=str, default="california_housing", help="name of dataset"
    )
    parser.add_argument(
        '--use_input_normalization', type=int, default=1, choices=[0, 1],
    )
    parser.add_argument(
        "--y_train_noise_std",
        type=float,
        default=0.0,
        help="standard deviation of noise",
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
    parser.add_argument('--model_type', type=str,
                        default='kan', choices=['kan', 'kangam', 'ebm', 'randomforest', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden dim for kan/kangam/mlp')
    parser.add_argument('--regularize_ridge', type=float, default=1.0,
                        help='regularization parameter for lin weight')
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


def _get_data(args, problem_type: str):
    # fetch data
    if args.dataset_name in DSET_KWARGS:
        X, y, feat_names = imodels.get_clean_dataset(
            **DSET_KWARGS[args.dataset_name])
    elif args.dataset_name in pmlb.dataset_names:
        X, y, feat_names = imodels.get_clean_dataset(
            args.dataset_name, data_source="pmlb"
        )
    # remove nan rows and normalize
    if len(y.shape) == 1 or y.shape[1] == 1:
        idxs_nan = np.isnan(X).any(axis=1) | pd.isna(y)
        y = y.reshape(-1, 1)
    else:
        idxs_nan = np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1)

    X = X[~idxs_nan]
    y = y[~idxs_nan]

    if problem_type == 'regression':
        y = StandardScaler().fit_transform(y)
    if y.shape[1] == 1:
        y = y.reshape(-1)
    if args.use_input_normalization:
        X = StandardScaler().fit_transform(X)

    # process and split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - args.train_frac,
        random_state=args.seed,
    )
    # optionally add noise to the data
    if args.y_train_noise_std > 0:
        y_train = y_train.astype(float) + np.random.normal(
            0, args.y_train_noise_std, size=y_train.shape
        )
    return X_train, X_test, y_train, y_test


def _get_model(args, problem_type):
    if problem_type == 'classification':
        if args.model_type == 'kan':
            m = KANClassifier(hidden_layer_size=args.hidden_dim)
        elif args.model_type == 'kangam':
            m = KANGAMClassifier(
                hidden_layer_size=args.hidden_dim, regularize_ridge=args.regularize_ridge)
        elif args.model_type == 'ebm':
            m = ExplainableBoostingClassifier(interactions=0)

    else:
        if args.model_type == 'kan':
            m = KANRegressor(hidden_layer_size=args.hidden_dim)
        elif args.model_type == 'kangam':
            m = KANGAMRegressor(hidden_layer_size=args.hidden_dim,
                                regularize_ridge=args.regularize_ridge)
        elif args.model_type == 'ebm':
            m = ExplainableBoostingRegressor(interactions=0)
    return m


def evaluate(args, r, m, X, y, problem_type, suffix='_train'):
    def _roc_no_error(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0

    if problem_type == 'classification':
        if len(y.shape) == 1 or y.shape[1] == 1:
            r["roc_auc" + suffix] = _roc_no_error(
                y, m.predict_proba(X)[:, 1]
            )
            r["acc" + suffix] = metrics.accuracy_score(
                y, m.predict(X))
        else:
            preds = m.predict(X)
            preds_proba = m.predict_proba(X)
            preds_proba = np.vstack([p[:, 1] for p in preds_proba]).T
            r['acc' + suffix] = np.mean(preds == y)
            rocs = [_roc_no_error(y, preds_proba[:, i])
                    for i in range(y.shape[1])]
            r['roc_auc' + suffix] = np.mean(rocs)

    elif problem_type == 'regression':
        r["r2" + suffix] = metrics.r2_score(y, m.predict(X))
        r["mse" + suffix] = metrics.mean_squared_error(
            y, m.predict(X))
        r['corr' + suffix] = np.corrcoef(y, m.predict(X).flatten())[0, 1]
    return r


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
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

    # get model

    if args.dataset_name in list(DSET_CLASSIFICATION_KWARGS.keys()) + list(DSET_CLASSIFICATION_MULTITASK_KWARGS.keys()) + pmlb.classification_dataset_names:
        problem_type = "classification"
    elif args.dataset_name in list(DSET_REGRESSION_KWARGS.keys()) + pmlb.regression_dataset_names:
        problem_type = "regression"
    else:
        raise ValueError(f"dataset {args.dataset_name} not found")
    m = _get_model(args, problem_type)

    X_train, X_test, y_train, y_test = _get_data(args, problem_type)
    # split cross-validation
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed
    )

    # fit model
    m.fit(X_train, y_train)
    r = evaluate(args, r, m, X_train, y_train, problem_type, suffix='_train')
    r = evaluate(args, r, m, X_tune, y_tune, problem_type, suffix='_tune')
    r = evaluate(args, r, m, X_test, y_test, problem_type, suffix='_test')

    # save data stuff
    r['n_samples'] = X_train.shape[0]
    r['n_features'] = X_train.shape[1]
    r['problem_type'] = problem_type

    # save results
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    # joblib.dump(m, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
