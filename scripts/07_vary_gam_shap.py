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
        # [d for d in pmlb.regression_dataset_names if not '_fri_' in d] +
        # figs_dsets_regr,
        # [n for n in DSET_CLASSIFICATION_MULTITASK_KWARGS] +
        figs_dsets_classification +
        pmlb.classification_dataset_names,

    "seed": [1],
    "save_dir": [join(repo_dir, "results", "gam_shap_no_interactions_nov6")],
    # 'train_frac': [0.1, 0.25, 0.5, 0.8],
    'train_frac': [0.8],
    "use_cache": [1],
    'use_shap_gam': [0, 1],
}
params_coupled_dict = {
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
# args_list = args_list[:1]
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "05_shap_gam.py"),
    # actually_run=False,
    # repeat_failed_jobs=True,
    n_cpus=32,
    # n_cpus=1,
    # shuffle=True,
    # reverse=True,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
