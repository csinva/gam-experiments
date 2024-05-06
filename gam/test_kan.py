from collections import defaultdict
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
from tqdm import tqdm
from gam.kan_modules import KAN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
import numpy as np
from gam.kan_classifiers import KANClassifier
import imodels
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

if __name__ == '__main__':
    n = 2000
    d = 20
    hidden_layer_sizes = 10
    # model_type = 'KAN'
    model_type = 'KANGAM'
    X, y = make_classification(n_samples=n, n_features=d, n_informative=2)

    # X, y, feature_names = imodels.get_clean_dataset("heart")
    X, y, feature_names = imodels.get_clean_dataset("breast_cancer")
    # X, y, feature_names = imodels.get_clean_dataset("juvenile_clean")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit
    r = defaultdict(list)
    torch.manual_seed(42)
    np.random.seed(42)
    models = {
        'KAN-5': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KAN'),
        'KANGAM-5-r0.1': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KANGAM', regularize_ridge=0.1),
        'KANGAM-5-r1': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KANGAM', regularize_ridge=1.0),
        'KANGAM-5-r10': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KANGAM', regularize_ridge=10),
        'KAN-20': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KAN'),
        'KANGAM-20-r0.1': KANClassifier(device='cpu', hidden_layer_size=20, model_type='KANGAM', regularize_ridge=0.1),
        'KANGAM-20-r1': KANClassifier(device='cpu', hidden_layer_size=20, model_type='KANGAM', regularize_ridge=1.0),
        'KANGAM-20-r10': KANClassifier(device='cpu', hidden_layer_size=20, model_type='KANGAM', regularize_ridge=10),
        'KAN-50': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KAN'),
        'KANGAM-50-r0.1': KANClassifier(device='cpu', hidden_layer_size=50, model_type='KANGAM', regularize_ridge=0.1),
        'KANGAM-50-r1': KANClassifier(device='cpu', hidden_layer_size=50, model_type='KANGAM', regularize_ridge=1.0),
        'KANGAM-50-r10': KANClassifier(device='cpu', hidden_layer_size=50, model_type='KANGAM', regularize_ridge=10),
        'KAN-100': KANClassifier(device='cpu', hidden_layer_size=5, model_type='KAN'),
        'KANGAM-100-r0.1': KANClassifier(device='cpu', hidden_layer_size=100, model_type='KANGAM', regularize_ridge=0.1),
        'KANGAM-100-r1': KANClassifier(device='cpu', hidden_layer_size=100, model_type='KANGAM', regularize_ridge=1.0),
        'KANGAM-100-r10': KANClassifier(device='cpu', hidden_layer_size=100, model_type='KANGAM', regularize_ridge=10),
        'ebm': ExplainableBoostingClassifier(interactions=0),
    }

    # fit model
    for name, model in models.items():
        model.fit(X_train, y_train)
        if 'KAN' in name:
            model.model.eval()
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"{name} Accuracy: {accuracy}")
        r['Accuracy'].append(accuracy)
        r['Model'].append(name)

    print('baseline', (y_test == 0).mean())
    print(pd.DataFrame(r))
