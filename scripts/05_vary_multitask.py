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
],  # add support2? # add mimic? # add CDI?


# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "dataset_name": pmlb.regression_dataset_names,
    # ['flags_multitask'],
    # figs_dsets_regr \

    # [n for n in DSET_CLASSIFICATION_MULTITASK_KWARGS],
    # figs_dsets_cls + pmlb.classification_dataset_names + \

    "seed": [1],
    # "save_dir": [join(repo_dir, "results", "multitask_gam")],
    "save_dir": [join(repo_dir, "results", "multitask_gam_mar12")],
    "use_cache": [1],

}
params_coupled_dict = {
    (
        "use_multitask",
        "interactions",
        "linear_penalty",
        'use_internal_classifiers',
        'use_onehot_prior',
        "n_boosting_rounds",
    ): [
        # baseline (single-task)
        # (0, 0.95, 'ridge', 0, 0),
        # (0, 0, 'ridge', 0),  # remove interactions

        # multitask
        (1, 0.95, 'ridge', 0, 0, 0),  # current best
        (1, 0.95, 'ridge', 1, 0, 0),  # use internal classifiers
        (1, 0.95, 'ridge', 0, 1, 0),  # use onehot_prior
        # (1, 0, 'ridge', 0),  # remove interactions

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
    n_cpus=80,
    # n_cpus=1,
    # shuffle=True,
    # reverse=True,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
