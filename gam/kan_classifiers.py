from sklearn.datasets import make_classification
import torch
import torch.nn as nn
from tqdm import tqdm

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
import logging
from gam.kan_modules import KAN, KANGAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KANClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_size=64, model_type='KAN', device=device, regularize_ridge=0.0):
        '''
        Params
        ------
        hidden_layer_size : int
            Number of neurons in the hidden layer.
        model_type: str
            Type of model to use. Either 'KAN' or 'KANGAM'.
        '''
        self.hidden_layer_size = hidden_layer_size
        self.model_type = model_type
        self.device = device
        self.regularize_ridge = regularize_ridge

    def fit(self, X, y, batch_size=512):
        if isinstance(self, ClassifierMixin):
            check_classification_targets(y)
            self.classes_, y = np.unique(y, return_inverse=True)
            num_classes = len(self.classes_)
        num_features = X.shape[1]
        if self.model_type == 'KAN':
            self.model = KAN(
                layers_hidden=[num_features,
                               self.hidden_layer_size, num_classes]
            ).to(self.device)
        elif self.model_type == 'KANGAM':
            self.model = KANGAM(
                num_features=num_features,
                hidden_layer_size=self.hidden_layer_size,
                n_classes=num_classes,
            ).to(self.device)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        X_train, X_tune, y_train, y_tune = train_test_split(
            X, y, test_size=0.2, random_state=42)

        dset_train = torch.utils.data.TensorDataset(X_train, y_train)
        dset_tune = torch.utils.data.TensorDataset(X_tune, y_tune)
        loader_train = DataLoader(
            dset_train, batch_size=batch_size, shuffle=True)
        loader_tune = DataLoader(
            dset_tune, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        # Define loss
        criterion = nn.CrossEntropyLoss()
        val_losses = []
        for epoch in tqdm(range(100)):
            # Train
            self.model.train()
            # with tqdm(loader_train) as pbar:
            # for i, (x, labs) in enumerate(pbar):
            for x, labs in loader_train:
                x = x.view(-1, num_features).to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, labs.to(self.device))

                if self.model_type == 'KANGAM':
                    loss += self.model.regularization_loss(
                        regularize_ridge=self.regularize_ridge)

                loss.backward()
                optimizer.step()
                # accuracy = (output.argmax(dim=1) ==
                # labs.to(self.device)).float().mean()
                # pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(
                # ), lr=optimizer.param_groups[0]['lr'])

            # Validation
            self.model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for x, labs in loader_tune:
                    x = x.view(-1, num_features).to(self.device)
                    output = self.model(x)
                    val_loss += criterion(output, labs.to(self.device)).item()
                    val_accuracy += (
                        (output.argmax(dim=1) == labs.to(
                            self.device)).float().mean().item()
                    )
            val_loss /= len(loader_tune)
            val_accuracy /= len(loader_tune)
            val_losses.append(val_loss)

            # Update learning rate
            scheduler.step()

            # print(
            #     f"Epoch {epoch + 1}, Tune Loss: {val_loss:.4f}, Tune Acc: {val_accuracy:.4f}"
            # )

            # apply early stopping
            if len(val_losses) > 3 and val_losses[-1] > val_losses[-2]:
                logging.debug("Early stopping")
                return self

        return self

    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        output = self.model(X)
        return output.argmax(dim=1).cpu().numpy()
