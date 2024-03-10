from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import pmlb
import numpy as np

repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "dataset_name": [
        # regression datasets
        "bike_sharing",
        "friedman1",
        "friedman2",
        "friedman3",
        "diabetes_regr",
        "abalone",
        "echo_months",
        "satellite_image",
        "california_housing",
    ] + pmlb.regression_dataset_names,
    # + \
    # pmlb.classification_dataset_names + \
    # [
    #     # classification datasets (treating these as regression)
    #     "heart",
    #     "breast_cancer",
    #     "diabetes",
    #     "breast_cancer",
    #     "credit_g",
    #     "juvenile",
    #     "compas",
    # ],  # add support2? # add mimic? # add CDI?
    "seed": [1],
    "save_dir": [join(repo_dir, "results", "multitask_gam")],
    "use_cache": [1],
    'use_multitask': [0, 1],
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
    script_name=join(repo_dir, "experiments", "03_multitask_gam.py"),
    # actually_run=False,
    repeat_failed_jobs=True,
    n_cpus=48,
    # reverse=True,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
