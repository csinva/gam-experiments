from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import pmlb
import numpy as np
from imodels.util.data_util import DSET_CLASSIFICATION_MULTITASK_KWARGS

repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping
figs_dsets_regr = [
    # regression datasets
    "bike_sharing",
    "friedman1",
    "friedman2",
    "friedman3",
    "diabetes_regr",
    # "abalone", # this has some string issue...
    "echo_months",
    "satellite_image",
    "california_housing",
]
figs_dsets_classification = [
    "heart",
    "breast_cancer",
    "diabetes",
    "breast_cancer",
    "credit_g",
    "juvenile",
    "compas",
]  # add support2? # add mimic? # add CDI?


# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "dataset_name":
        [d for d in pmlb.regression_dataset_names if not '_fri_' in d] +
        figs_dsets_regr,
        # [n for n in DSET_CLASSIFICATION_MULTITASK_KWARGS] +
        # figs_dsets_classification +
        # pmlb.classification_dataset_names,

    "seed": [1],
    "save_dir": [join(repo_dir, "results", "multitask_gam_mar12")],
    # 'train_frac': [0.1, 0.25, 0.5, 0.8],
    'train_frac': [0.8],
    "use_cache": [1],

}
params_coupled_dict = {
    (
        "use_multitask",
        "interactions",
        "n_boosting_rounds",
        'max_rounds',
        'boosting_strategy',
        # "linear_penalty",
        # 'use_internal_classifiers',
        # 'use_onehot_prior',
        # 'use_fit_target_curves',
    ): [
        # baseline (single-task)
        (0, 0.95, 0, 5000, 'adaboost'),
        (0, 0.95, 8, 5000, 'bagging'),

        # multitask
        (1, 0.95, 0, 5000, 'adaboost'),  # current best
        (1, 0.95, 8, 5000, 'bagging'),  # current best

        # variations
        (1, 0.95, 8, 50, 'residual'),  # vary boosting
        (1, 0.95, 4, 100, 'residual'),  # vary boosting
        (1, 0.95, 2, 100, 'residual'),  # vary boosting
        (1, 0.95, 8, 50, 'bagging'),  # vary boosting
        (1, 0.95, 8, 100, 'bagging'),  # vary boosting


        #### repeat with no interactions #####
        # baseline (single-task)
        (0, 0, 0, 5000, 'adaboost'),
        (0, 0, 8, 5000, 'bagging'),

        # multitask
        (1, 0, 0, 5000, 'adaboost'),  # current best
        (1, 0, 8, 5000, 'bagging'),  # current best

        # variations
        (1, 0, 8, 50, 'residual'),  # vary boosting
        (1, 0, 4, 100, 'residual'),  # vary boosting
        (1, 0, 2, 100, 'residual'),  # vary boosting
        (1, 0, 8, 50, 'bagging'),  # vary boosting
        (1, 0, 8, 100, 'bagging'),  # vary boosting

        # (1, 0.95, 8, 5000, 'bagging'),  # vary boosting
        # (1, 0.95, 'ridge', 0, 0, 0),  # don't fit target curves
        # (1, 0.95, 'ridge', 1, 0, 0),  # use internal classifiers
        # (1, 0.95, 'ridge', 0, 1, 0),  # use onehot_prior
        # (1, 0, 'ridge', 0),  # remove interactions
        # (1, 0.95, 8, 50, 'adaboost'),  # vary boosting
        # (1, 0.95, 4, 100, 'adaboost'),  # vary boosting
        # (1, 0.95, 2, 100, 'adaboost'),  # vary boosting

        # renormalize_features
        # remove onehot prior
        # vary linear penalty (lasso, elasticnet)
        # vary boosting rounds
    ]
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
# args_list = args_list[:1]
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "03_multitask_gam.py"),
    # actually_run=False,
    # repeat_failed_jobs=True,
    n_cpus=32,
    # n_cpus=1,
    shuffle=True,
    # reverse=True,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
