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
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
import pmlb
import imodels.algebraic.gam_multitask
import pandas as pd

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args ######################
    parser.add_argument(
        "--dataset_name", type=str, default="emotions_multitask", help="name of dataset"
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
        "--use_multitask",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to use multitask",
    )
    parser.add_argument(
        '--interactions',
        type=float,
        default=0.95,
        help='''Interaction terms to be included in the model. Options are:
Integer (1 <= interactions): Count of interactions to be automatically selected
Percentage (interactions < 1.0): Determine the integer count of interactions by multiplying the number of features by this percentage
List of tuples: The tuples contain the indices of the features within the additive term
''')
    parser.add_argument(
        '--use_onehot_prior',
        type=int,
        default=0,
        choices=[0, 1],
        help='whether to use onehot prior'
    )
    parser.add_argument(
        '--use_internal_classifiers',
        type=int,
        default=0,
        choices=[0, 1],
        help='whether to use internal classifiers (as opposed to regressors)'
    )
    parser.add_argument(
        '--linear_penalty',
        type=str,
        default='ridge',
        choices=['ridge', 'lasso', 'elasticnet'],
        help='penalty for linear model'
    )
    parser.add_argument(
        '--use_fit_target_curves',
        type=int,
        default=1,
        choices=[0, 1],
        help='whether to fit target curve directly (must be 1 when given multiple outputs)',
    )
    parser.add_argument(
        '--use_normalize_feature_targets',
        type=int,
        default=0,
        choices=[0, 1],
        help='whether to normalize outputs for internal classifiers (as opposed to regressors)'
    )
    parser.add_argument(
        '--use_renormalize_features',
        type=int,
        default=0,
        choices=[0, 1],
        help='whether to renormalize features from ebm curves'
    )
    parser.add_argument(
        '--n_boosting_rounds',
        type=int,
        default=0,
        help='number of boosting rounds'
    )
    parser.add_argument(
        '--max_rounds',
        type=int,
        default=5000,
        help='number of cyclic rounds for each EBM (lower in conjunction with increasing n_boosting_rounds)',
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


def _check_args(args):
    """Check that the arguments are valid"""
    if not args.use_multitask:
        assert args.linear_penalty == 'ridge'
        assert args.n_boosting_rounds >= 0


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
    if args.collinearity_factor > 0:
        X = make_covariates_more_collinear(
            X, collinearity_factor=args.collinearity_factor, seed=args.seed
        )
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


def _get_model(args):
    kwargs = dict(
        multitask=args.use_multitask,
        linear_penalty=args.linear_penalty,
        interactions=args.interactions,
        random_state=args.seed,
        onehot_prior=bool(args.use_onehot_prior),
        renormalize_features=bool(args.use_renormalize_features),
        use_internal_classifiers=bool(args.use_internal_classifiers),
        use_normalize_feature_targets=bool(args.use_normalize_feature_targets),
        fit_target_curves=bool(args.use_fit_target_curves),
        ebm_kwargs={'max_rounds': args.max_rounds},
    )
    if args.dataset_name in list(DSET_CLASSIFICATION_KWARGS.keys()) + list(DSET_CLASSIFICATION_MULTITASK_KWARGS.keys()) + pmlb.classification_dataset_names:
        est = imodels.algebraic.gam_multitask.MultiTaskGAMClassifier(**kwargs)
    elif args.dataset_name in list(DSET_REGRESSION_KWARGS.keys()) + pmlb.regression_dataset_names:
        est = imodels.algebraic.gam_multitask.MultiTaskGAMRegressor(**kwargs)
    else:
        raise ValueError(f"dataset {args.dataset_name} not found")
    if args.n_boosting_rounds > 0:
        if isinstance(est, ClassifierMixin):
            m = AdaBoostClassifier(
                estimator=est, n_estimators=args.n_boosting_rounds)
        else:
            m = AdaBoostRegressor(
                estimator=est, n_estimators=args.n_boosting_rounds)
    else:
        m = est
    return m


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()
    _check_args(args)

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.ERROR)

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
    m = _get_model(args)
    problem_type = "classification" if isinstance(
        m, ClassifierMixin) else "regression"

    # decide between classification and regression & initialize model
    X_train, X_test, y_train, y_test = _get_data(args, problem_type)

    # fit model
    # print('shapes', X_train.shape, y_train.shape, problem_type)
    m.fit(X_train, y_train)

    # check roc auc score
    if problem_type == 'classification':
        def _roc_no_error(y_true, y_pred):
            try:
                return roc_auc_score(y_true, y_pred)
            except ValueError:
                return 0
        if len(y_test.shape) == 1 or y_test.shape[1] == 1:
            r["roc_auc_train"] = _roc_no_error(
                y_train, m.predict_proba(X_train)[:, 1]
            )
            r["acc_train"] = metrics.accuracy_score(
                y_train, m.predict(X_train))
            r["roc_auc_test"] = _roc_no_error(
                y_test, m.predict_proba(X_test)[:, 1])
            r["acc_test"] = metrics.accuracy_score(y_test, m.predict(X_test))
        else:
            preds = m.predict(X_train)
            preds_proba = m.predict_proba(X_train)
            preds_proba = np.vstack([p[:, 1] for p in preds_proba]).T
            r['acc_train'] = np.mean(preds == y_train)
            rocs = [_roc_no_error(y_train, preds_proba[:, i])
                    for i in range(y_train.shape[1])]
            r['roc_auc_train'] = np.mean(rocs)

            preds = m.predict(X_test)
            preds_proba = m.predict_proba(X_test)
            preds_proba = np.vstack([p[:, 1] for p in preds_proba]).T
            r['acc_test'] = np.mean(preds == y_test)
            rocs = [_roc_no_error(y_test[:, i], preds_proba[:, i])
                    for i in range(y_test.shape[1])]
            r['roc_auc_test'] = np.mean(rocs)

    elif problem_type == 'regression':
        r["r2_train"] = metrics.r2_score(y_train, m.predict(X_train))
        r["mse_train"] = metrics.mean_squared_error(
            y_train, m.predict(X_train))
        r['corr_train'] = np.corrcoef(y_train, m.predict(X_train))[0, 1]
        r["r2_test"] = metrics.r2_score(y_test, m.predict(X_test))
        r["mse_test"] = metrics.mean_squared_error(y_test, m.predict(X_test))
        r['corr_test'] = np.corrcoef(y_test, m.predict(X_test))[0, 1]

    if args.use_multitask and hasattr(m, 'lin_model') and hasattr(m.lin_model, 'alpha_'):
        # r['coef_'] = m.lin_model.coef_
        r['best_alpha'] = m.lin_model.alpha_
        # if hasattr(m, 'term_names_'):
        # r['term_names_'] = m.term_names_

    # save data stuff
    r['n_samples'] = X_train.shape[0]
    r['n_features'] = X_train.shape[1]
    r['problem_type'] = problem_type

    # save results
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    # joblib.dump(m, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
