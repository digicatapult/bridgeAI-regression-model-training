"""Model definition for profiler."""

import torch
import torch.nn.functional as F
from torch import nn


class NNModel(nn.Module):
    """Simple feed forward network to predict CPU utilisation."""

    def __init__(self, in_feats: int = 8):
        """Init."""
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(in_feats, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def save_model(model, path):
    """Save the pytorch model's state dictionary."""
    torch.save(model, path)


def load_model_weights(model, weights_path: str):
    """Load model weights."""
    model.load_state_dict(torch.load(weights_path))


class EarlyStopping:
    """Early stopping class."""

    def __init__(self, patience=5, delta=0):
        """Init the early stopping class"""
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model = None

    def __call__(self, loss, model):
        if (self.best_loss + self.delta) >= loss:
            self.best_loss = loss
            self.save_checkpoint(loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        self.best_loss = val_loss
        self.best_model = model.state_dict().copy()
