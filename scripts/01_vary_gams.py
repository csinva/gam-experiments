from imodelsx import submit_utils
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "dataset_name": [
        "sonar",
        "heart",
        "diabetes",
        "breast_cancer",
        "credit_g",
        "juvenile",
        "compas",
        # "bike_sharing", (regression, not supported)
        # "readmission",
        # "adult"
    ],  # add support2? # add mimic? # add CDI?
    "seed": [1, 2, 3],
    "save_dir": [join(repo_dir, "results", 'main')],
    "use_cache": [1],
    "n_boosting_rounds": [0, 5, 25, 125],
    "n_boosting_rounds_marginal": [0, 5, 25, 125],
    "fit_linear_marginal": ["None"],  # , "nnls", "ridge"],
    "reg_param": [0.0, 100.0, 1e4],
    "reg_param_marginal": [0.0, 100, 1e4],
    "boosting_strategy": ["cyclic", "greedy"],
    'use_bagging_ensemble': [0, 1],
}
params_coupled_dict = {}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "01_train_gam.py"),
    # actually_run=False,
    n_cpus=32,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
