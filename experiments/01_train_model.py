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

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import joblib
import imodels
from imodels import TreeGAMClassifier
import inspect
import cache_save_utils


DSET_KWARGS = {
    "heart": {"dataset_name": "heart", "data_source": "imodels"},
    "breast_cancer": {"dataset_name": "breast_cancer", "data_source": "sklearn"},
    "credit-g": {"dataset_name": "credit_g", "data_source": "imodels"},
    "juvenile": {"dataset_name": "juvenile_clean", "data_source": "imodels"},
    "compas": {"dataset_name": "compas_two_year_clean", "data_source": "imodels"},
    # 'adult': {'dataset_name': '1182', 'data_source': 'openml'},
}


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
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )
    parser.add_argument(
        "--n_boosting_rounds",
        type=int,
        default=0,
        help="number of cyclical boosting rounds",
    )
    parser.add_argument(
        "--n_boosting_rounds_marginal",
        type=int,
        default=0,
        help="number of marginal boosting rounds",
    )
    parser.add_argument(
        "--fit_linear_marginal",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to fit linear marginal",
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
        random_state=42,
        stratify=y,  # don't shuffle this for now, to make sure gam curves have same points
    )

    m = TreeGAMClassifier(
        n_boosting_rounds=args.n_boosting_rounds,
        n_boosting_rounds_marginal=args.n_boosting_rounds_marginal,
        random_state=args.seed,
        fit_linear_marginal=args.fit_linear_marginal,
    )
    m.fit(X_train, y_train)
    r["roc_auc_train"] = metrics.roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])
    r["acc_train"] = metrics.accuracy_score(y_train, m.predict(X_train))
    r["roc_auc_test"] = metrics.roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    r["acc_test"] = metrics.accuracy_score(y_test, m.predict(X_test))

    feature_vals_list, shape_function_vals_list = m.get_shape_function_vals(
        X_train, max_evals=100
    )
    r["shape_function_vals_list"] = shape_function_vals_list

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    joblib.dump(m, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
