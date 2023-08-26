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
    "n_boosting_rounds": [0, 1000],
    "n_boosting_rounds_marginal": [0, 1000],
    "decay_rate_towards_marginal": [0.5, 0.75, 1.0],
}
params_coupled_dict = {
    ("fit_linear_marginal", "use_select_linear_marginal"): [
        ('None', 0),
        ('NNLS', 1),
    ]
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, "experiments", "01_train_model.py"),
    # actually_run=False,
    n_cpus=1,
    # error_logs_directory=join(repo_dir, 'scripts', 'error_logs')
)
