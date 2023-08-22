import argparse
import sys
import os.path
import numpy as np
from os.path import dirname, join
repo_dir = dirname(dirname(os.path.abspath(__file__)))


def calc_mean_std_across_curves(shape_function_vals_list_list):
    '''
    shape_function_vals_list_list: list of lists of arrays
        num_seeds x num_features x num_points
    '''
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