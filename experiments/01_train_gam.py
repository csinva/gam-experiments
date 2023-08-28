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

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import joblib
import imodels
from imodels import TreeGAMClassifier
import inspect
import cache_save_utils
import warnings
from imodels.util.data_util import DSET_KWARGS


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_name", type=str, default="heart", help="name of dataset"
    )

    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results", 'test'),
        help="directory for saving",
    )
    parser.add_argument(
        "--n_boosting_rounds",
        type=int,
        default=0,
        help="number of cyclical boosting rounds",
    )
    parser.add_argument(
        "--reg_param",
        type=float,
        default=0.0,
        help="regularization parameter for cyclical boosting",
    )
    parser.add_argument(
        "--n_boosting_rounds_marginal",
        type=int,
        default=0,
        help="number of marginal boosting rounds",
    )
    parser.add_argument(
        "--fit_linear_marginal",
        type=str,
        default="None",
        choices=[None, "None", "ridge", "NNLS"],
        help="whether to fit linear marginal",
    )
    parser.add_argument(
        "--reg_param_marginal",
        type=float,
        default=0.0,
        help="regularization parameter for marginal boosting",
    )
    parser.add_argument(
        "--boosting_strategy",
        type=str,
        default="cyclic",
        choices=["cyclic", "greedy"],
        help="strategy for boosting",
    )
    parser.add_argument(
        "--bagging_ensemble",
        type=str,
        default="None",
        choices=["None", "samples", "features", "both"],
        help="whether to use bagging ensemble of GAMs",
    )
    parser.add_argument(
        "--bagging_n_estimators",
        type=int,
        default=30,
        help="Number of GAMs to use in bagging ensemble",
    )
    parser.add_argument(
        "--use_select_linear_marginal",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to use select cycle features using marginal nnls",
    )
    parser.add_argument(
        "--decay_rate_towards_marginal",
        type=float,
        default=1.0,
        help="decay rate towards marginal",
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


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # Check arguments
    if args.decay_rate_towards_marginal < 1.0 and (
        (args.n_boosting_rounds_marginal == 0) or (args.n_boosting_rounds == 0)
    ):
        warnings.warn(
            "Must have n_boosting_rounds_marginal > 0 and n_boosting_rounds > 0 if decay_rate_towards_marginal < 1.0"
        )
        exit(0)
    elif args.use_select_linear_marginal and (
        args.fit_linear_marginal != "NNLS" or args.n_boosting_rounds_marginal == 0
    ):
        warnings.warn(
            "Must have fit_linear_marginal == 'NNLS' if use_select_linear_marginal == True"
        )
        exit(0)

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

    X, y, feat_names = imodels.get_clean_dataset(**DSET_KWARGS[args.dataset_name])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,  # don't shuffle this for now, to make sure gam curves have same points (just being lazy)
        stratify=y,
    )

    clf = TreeGAMClassifier(
        n_boosting_rounds=args.n_boosting_rounds,
        reg_param=args.reg_param,
        n_boosting_rounds_marginal=args.n_boosting_rounds_marginal,
        reg_param_marginal=args.reg_param_marginal,
        fit_linear_marginal=args.fit_linear_marginal,
        boosting_strategy=args.boosting_strategy,
        select_linear_marginal=args.use_select_linear_marginal,
        decay_rate_towards_marginal=args.decay_rate_towards_marginal,
        random_state=args.seed,
    )
    if args.bagging_ensemble != "None":
        m = BaggingClassifier(
            estimator=clf,
            n_estimators=args.bagging_n_estimators,
            random_state=args.seed,
            bootstrap=args.bagging_ensemble in ["samples", "both"],
            bootstrap_features=args.bagging_ensemble in ["features", "both"],
        )
        m.fit(X_train, y_train)
        # r["val_score"] = m.oob_score_
    else:
        m = clf
        m.fit(X_train, y_train)
        r["val_score"] = -m.mse_val_

    r["roc_auc_train"] = metrics.roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])
    r["acc_train"] = metrics.accuracy_score(y_train, m.predict(X_train))
    r["roc_auc_test"] = metrics.roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    r["acc_test"] = metrics.accuracy_score(y_test, m.predict(X_test))

    feature_vals_list, shape_function_vals_list = extract_marginal_curves(
        m, X_train, max_evals=100
    )
    r["shape_function_vals_list"] = shape_function_vals_list

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    joblib.dump(m, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
