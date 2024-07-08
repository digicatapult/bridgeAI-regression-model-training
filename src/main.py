"""Main training pipeline."""

import os

import joblib
import mlflow

from src import mlflow_utils, utils
from src.model import NNModel
from src.preprocess import create_dataloader, load_and_split_data, preprocess
from src.train import train
from src.utils import logger


def main():
    """Main training pipeline.

    Steps involved:
        1. Split the raw data into train, validation and test data
        2. preprocess train data and save the preprocessing steps/pipeline
        3. preprocess validation and test data
        4. create data loaders for train, val and test data
            (only for Torch models)
        5. Create/Initialise a model object
        6. Train the model and save/log the results
    """
    config_path = "./config.yaml"

    config = utils.load_yaml_config(config_path)
    logger.info("Config", extra=config)

    # 1. Split the raw data
    logger.info("Splitting the data into train, val, and test.")
    load_and_split_data(config)

    # 2 + 4.[Train data] preprocess train data and save the preprocessing steps
    train_x, train_y = preprocess(
        config["data"]["train_data_save_path"],
        config,
        preprocessor=None,
        save_preprocessor=True,
    )
    train_dataloader = create_dataloader(
        train_x,
        train_y,
        batch_size=config["model"]["train_batch_size"],
        shuffle=True,
    )

    # Load the preprocessor created while preparing the train data
    # This is needed to apply the same preprocessing to other data
    preprocessor = joblib.load(config["data"]["preprocessor_path"])

    # 3 + 4 [val data] preprocess and create data loader
    # preprocess val data and prepare validation data loader
    val_x, val_y = preprocess(
        config["data"]["val_data_save_path"], config, preprocessor=preprocessor
    )
    val_dataloader = create_dataloader(
        val_x, val_y, batch_size=config["model"]["test_batch_size"]
    )

    # 3 + 4 [test data] preprocess and create data loader
    test_x, test_y = preprocess(
        config["data"]["test_data_save_path"],
        config,
        preprocessor=preprocessor,
    )
    test_dataloader = create_dataloader(
        test_x, test_y, batch_size=config["model"]["test_batch_size"]
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
    try:
        # Set the active run to the previously started run
        with mlflow.start_run(run_id=run_id, nested=True):
            train(
                regression_model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                config,
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


if __name__ == "__main__":
    main()
