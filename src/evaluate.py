"""Model evaluation on a given dataset."""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import torch
from torch import nn

from src import utils
from src.preprocess import create_dataloader
from src.utils import get_device, logger


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


def evaluate_on_test_data(config, run_id: str, metric_name: str):
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
    # Load the trained model weights
    model_save_path = (
        Path("./artefacts") / f'{config["model"]["model_name"]}.pth'
    )

    model = torch.load(model_save_path, weights_only=False)
    model = model.to(device)
    criterion = nn.MSELoss()

    logger.info(f"Loaded the model from {model_save_path} for evaluation")

    # Evaluate the model
    test_loss = evaluate(model, criterion, test_dataloader, device=device)

    logger.info(f"Model evaluated: {metric_name} = {test_loss: .4f}")

    if run_id is not None:
        try:
            mlflow_tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
            )
            # Set the tracking URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Initialize the MLflow client
            client = mlflow.tracking.MlflowClient()

            # Log the test loss to mlflow;
            client.log_metric(run_id, metric_name, test_loss)
        except Exception as e:
            logger.error(f"MLFlow logging of metric failed: {e}")
    else:
        logger.warning(
            f"MLFlow run_id is None! Not logging the metric {metric_name}"
        )

    print(test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="The MLFlow run ID where the "
        "evaluation result should be updated",
    )

    parser.add_argument(
        "--metric_name",
        type=str,
        default="test_loss",
        required=False,
        help="The MLFlow metric name to log the evaluation result",
    )

    args = parser.parse_args()
    config = utils.load_yaml_config()
    evaluate_on_test_data(config, args.run_id, args.metric_name)
