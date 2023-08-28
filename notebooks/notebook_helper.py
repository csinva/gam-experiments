import argparse
import sys
import os.path
import numpy as np
from os.path import dirname, join
import imodelsx.process_results
from tqdm import tqdm
import pandas as pd
repo_dir = dirname(dirname(os.path.abspath(__file__)))


def calc_mean_std_across_curves(shape_function_vals_list_list):
    """
    shape_function_vals_list_list: list of lists of arrays
        num_seeds x num_features x num_points
    """
    stds = []
    n_features = len(shape_function_vals_list_list[0])
    for feature_num in range(n_features):
        shape_function_vals = np.array(
            [
                shape_function_vals_list_list[i][feature_num]
                for i in range(len(shape_function_vals_list_list))
            ]
        )
        shape_function_mean = np.mean(shape_function_vals, axis=0)
        stds.append(np.mean(np.std(shape_function_vals, axis=0)))
    return 100 * np.mean(stds)


def get_ravg_with_stability(r, experiment_filename="01_train_model.py"):
    ravg = []
    group_keys = [
        k
        for k in imodelsx.process_results.get_main_args_list(
            experiment_filename=experiment_filename
        )
        if not k == "seed"
    ]
    rg = r.groupby(group_keys)
    for group_key, group_idx in tqdm(rg.groups.items()):
        g = r.iloc[group_idx]
        numeric_keys = [
            k for k in list(g.select_dtypes("number")) if not k in group_keys
        ]
        # numeric_keys = ['roc_auc_test', 'stability']
        row = g[numeric_keys].mean()
        for k in group_keys:
            row[k] = g[k].iloc[0]
        try:
            row["instability"] = calc_mean_std_across_curves(
                g["shape_function_vals_list"].values.tolist()
            )
            ravg.append(row)
        except:
            pass
    return pd.DataFrame(ravg)
