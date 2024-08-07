"""NN model training loop with logging."""

import os
import pickle
import sys
from pathlib import Path

import mlflow
import torch
from torch import nn

from src import mlflow_utils
from src.evaluate import evaluate
from src.model import EarlyStopping, NNModel, save_model
from src.preprocess import create_dataloader
from src.utils import get_device, logger


def train(
    model,
    trainloader,
    valloader,
    config,
    model_save_path,
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
        6. At the end log the best model to MLFlow
    """
    # Selecting device- 'cpu' or 'gpu'
    device = get_device(config)

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

    # Training for given number of epochs
    n_epochs = config["model"]["n_epochs"]
    # Training loop
    train_loss = 0.0
    val_loss = 0.0
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

    # Register the model in the MLFlow registry
    model_uri = mlflow_utils.log_torch_model(
        model,
        valloader,
        mlflow.active_run(),
        config["model"]["model_name"],
    )
    logger.info(f"Model registered in MLFlow. uri - {model_uri}")


def train_model(config):
    """Load the data and train the model."""
    # Load preprocessed test data and prepare dataloader
    with open(
        Path(config["dvc"]["train_data_path"]).with_suffix(".pkl"), "rb"
    ) as f:
        train_x, train_y = pickle.load(f)

    train_dataloader = create_dataloader(
        train_x,
        train_y,
        batch_size=config["model"]["train_batch_size"],
        shuffle=True,
    )

    # Load preprocessed val data and prepare dataloader
    with open(
        Path(config["dvc"]["val_data_path"]).with_suffix(".pkl"), "rb"
    ) as f:
        val_x, val_y = pickle.load(f)
    val_dataloader = create_dataloader(
        val_x, val_y, batch_size=config["model"]["test_batch_size"]
    )

    logger.info("data loaders created.")

    # 5. Create/Initialise a model object
    regression_model = NNModel(in_feats=train_x.shape[1])

    # 6. start MLFLow run and log the parameters
    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
    )
    mlflow_utils.start_run(mlflow_tracking_uri, config["mlflow"]["expt_name"])
    run_id = mlflow_utils.get_activ_run_id()
    logger.info(f"started MLFlow run with run id: {run_id}")
    mlflow_utils.log_param_dict(mlflow_utils.flatten_dict(config))

    # 7. Train the model with metric logging
    logger.info("starting the training.")

    # Model save path
    model_save_path = f'{config["model"]["model_name"]}.pth'

    try:
        # Set the active run to the previously started run
        with mlflow.start_run(run_id=run_id, nested=True):
            train(
                regression_model,
                train_dataloader,
                val_dataloader,
                config,
                model_save_path,
                verbose=True,
            )
            logger.info("Training completed.")
    except Exception as err:
        mlflow.end_run("FAILED")
        mlflow.log_param("error", err)
        logger.info(f"Training task failed with error - {err}")
        raise Exception(err)
    else:
        # 8. Ensure the MLFlow run has ended properly
        if mlflow.active_run():
            mlflow.end_run("FINISHED")
    return run_id
