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

if __name__ == '__main__':
    n = 2000
    d = 20
    # model_type = 'KAN'
    model_type = 'KANGAM'
    X, y = make_classification(n_samples=n, n_features=d, n_informative=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit
    clf = KANClassifier(device='cpu')
    clf.fit(X_train, y_train)
    clf.model.eval()
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy}")
