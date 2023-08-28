from imodelsx import submit_utils
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "dataset_name": [
        "bike_sharing",
        "friedman1",
        "friedman2",
        "friedman3",
        "diabetes_regr",
        "abalone",
        "echo_months",
        "satellite_image",
        "california_housing",
    ],  # add support2? # add mimic? # add CDI?
    "seed": [1, 2, 3],
    "save_dir": [join(repo_dir, "results", "linear")],
    "use_cache": [1],
    # "train_frac": [0.2, 0.5, 0.8],
    # "y_train_noise_std": [0.0, 1, 10],
    "collinearity_factor": [0.0, 0.25, 0.5, 0.75, 1.0],
}
params_coupled_dict = {
    ("est_marginal_name", "est_main_name", "use_marginal_divide_by_d"): [
        ("ridge", "ridge", 0),
        ("ridge", "ridge", 1),
        ("None", "ridge", 1),
        ("ridge", "None", 1),
    ]
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "02_train_linear.py"),
    # actually_run=False,
    n_cpus=64,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
