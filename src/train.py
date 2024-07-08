"""NN model training loop with logging."""

import sys
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from torch import nn

from src import mlflow_utils
from src.evaluate import evaluate
from src.model import EarlyStopping, save_model
from src.utils import logger


def train(
    model,
    trainloader,
    valloader,
    testloader,
    config,
    verbose=True,
):
    """Train the model on the training data set.

    Steps involved here:
        1. Define the model training criteria, optimiser, early stopper, etc
        2. Train the model for given number of epochs using train data
        3. At the end of each epoch, validate the model using validation data
        4. Repeat the training until epochs are completed or early stopping
            criteria is met
        5. Save the best model
        7. At the end of training do a test on test data and log the results
    """
    # Selecting device- 'cpu' or 'gpu'
    if config["model"]["use_gpu"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logger.warn("Falling back to 'CPU'; no usable 'GPU' found!")
    else:
        device = torch.device("cpu")

    logger.info(f"Using {device}")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"]
    )

    # Initialise early stopping class
    early_stopper = EarlyStopping(
        patience=config["model"]["es_patience"],
        delta=config["model"]["es_delta"],
    )

    # Model save path
    timestamp_ = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_unique_name = f'{config["model"]["model_name"]}-{timestamp_}.pth'
    model_save_path = Path(config["model"]["save_path"]) / model_unique_name

    # Training for given number of epochs
    n_epochs = config["model"]["n_epochs"]
    # Training loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        # Iterating over batches of data within an epoch
        for batch in trainloader:
            data, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss

        # Update the metrics
        val_loss = evaluate(model, criterion, valloader, device=device)
        train_loss /= len(trainloader.dataset)

        # Update early stopping details
        early_stopper(val_loss, model)

        # Update the metrics to mlflow
        mlflow.log_metric("train_loss", train_loss.item(), step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # log the metrics to console
        if verbose:
            sys.stdout.write(
                f"\rEpoch {epoch}/{n_epochs}, Train Loss: {train_loss: .4f}, "
                f"Val Loss: {val_loss: .4f}"
            )
            sys.stdout.flush()

        # Early stop the training if no model improvements observed
        if early_stopper.early_stop:
            logger.info("Early stopping...")
            # Save the best model
            save_model(early_stopper.best_model, model_save_path)
            break
    logger.info(
        f"End of training metrics: Train Loss: {train_loss: .4f}, "
        f"Val Loss: {val_loss: .4f}"
    )

    # Evaluate on unseen test data at the end of training
    test_loss = evaluate(model, criterion, testloader, device=device)
    # Log the test loss to mlflow;
    mlflow.log_metric(
        "test_loss",
        test_loss,
    )
    logger.info(f"Test Loss: {test_loss: .4f}")

    # Register the model in the MLFlow registry
    model_uri = mlflow_utils.log_torch_model(
        model,
        valloader,
        mlflow.active_run(),
        config["model"]["model_name"],
    )
    logger.info(f"Model registered in MLFlow. uri - {model_uri}")
