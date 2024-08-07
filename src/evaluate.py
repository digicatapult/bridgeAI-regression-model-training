"""Model evaluation on a given dataset."""

import os
import pickle
from pathlib import Path

import mlflow
import torch
from torch import nn

from src.model import NNModel
from src.preprocess import create_dataloader
from utils import get_device


def evaluate(model, criterion, dataloader, device):
    """Evaluate the model on the given dataset."""
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(data)
            loss += criterion(outputs.view(-1), labels).item()
    loss /= len(dataloader.dataset)
    return loss


def evaluate_on_test_data(config, run_id):
    """Evaluate the model on test dataset and log the metrics to mlflow."""
    device = get_device(config)

    # Load preprocessed test data and prepare dataloader
    with open(
        Path(config["dvc"]["test_data_path"]).with_suffix(".pkl"), "rb"
    ) as f:
        test_x, test_y = pickle.load(f)
    test_dataloader = create_dataloader(
        test_x, test_y, batch_size=config["model"]["test_batch_size"]
    )
    in_feats = test_dataloader.dataset[0][0].shape[0]

    model = NNModel(in_feats=in_feats)

    model.load_state_dict(torch.load(f'{config["model"]["model_name"]}.pth'))
    model = model.to(device)
    criterion = nn.MSELoss()

    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
    )
    # Set the tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Initialize the MLflow client
    client = mlflow.tracking.MlflowClient()

    test_loss = evaluate(model, criterion, test_dataloader, device=device)
    # Log the test loss to mlflow;
    client.log_metric(run_id, "test_loss", test_loss)

    print(test_loss)
