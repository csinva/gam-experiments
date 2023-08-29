from imodelsx import submit_utils
from os.path import dirname, join
import os.path

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
        # classification datasets (treating these as regression)
        "heart",
        "breast_cancer",
        "diabetes",
        "breast_cancer",
        "credit_g",
        "juvenile",
        "compas",
    ],  # add support2? # add mimic? # add CDI?
    # "seed": [1, 2, 3],
    "seed": [1],
    "save_dir": [join(repo_dir, "results", "linear")],
    "use_cache": [1],
    ############# vary data ############
    # "train_frac": [0.2, 0.5, 0.8],
    # "y_train_noise_std": [0.0, 1, 10],
    # "collinearity_factor": [0.0, 0.5, 1.0],
    # "alpha": [0.1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
    # "alpha": np.logspace(-3, 3, num=5).tolist(),
    # "alpha": np.logspace(-3, 3, num=5).tolist(),
}
params_coupled_dict = {
    # ("est_marginal_name", "est_main_name", "use_marginal_divide_by_d"): [
    #     ("ridge", "ridge", 0),
    #     ("ridge", "ridge", 1),
    #     ("None", "ridge", 1),
    #     ("ridge", "None", 1),
    # ]
    (
        "est_marginal_name",
        "est_main_name",
        "use_marginal_sign_constraint",
        "use_marginal_divide_by_d",
    ): [
        # baselines
        ("None", "ridge", 0, 1),
        ("None", "elasticnet", 0, 1),
        ("None", "lasso", 0, 1),

        # marginal only (for coef comparisons)
        ("ridge", "None", 0, 1),

        # sign regularization
        ("ridge", "ridge", 1, 1),
        ("ridge", "elasticnet", 1, 1),
        ("ridge", "lasso", 1, 1),

        # value regularization


        # ("None", "ridge", 0, 1),
        # ("None", "lasso", 0, 1),
        # marginal shrinkage
        # ("ridge", "ridge", 0, 1),
        # ("ridge", "NNLS-ridge", 1, 1),
        # ("ridge", "NNLS-lasso", 1, 1),
        # ("ridge", "None", 0, 1),  # marginal only
        # ("ridge", "ridge", 0, 0),  # don't divide by d
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
    repeat_failed_jobs=True,
    n_cpus=64,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
