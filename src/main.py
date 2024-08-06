"""Main training pipeline."""

from src import utils
from src.evaluate import evaluate_on_test_data
from src.preprocess import preprocess_datasets
from src.train import train_model
from src.utils import logger


def main():
    """Main training pipeline.

    Steps involved:
        1. Fetch versioned data
        2. preprocess the data splits
        3. Load the preprocessed data as datasets and train model
        4. Evaluate the model on test data
    """

    config = utils.load_yaml_config()
    logger.info("Config", extra=config)

    # 1. fetch the data
    # fetch_data(config)

    # 2. preprocess the datasets
    preprocess_datasets(config)

    # 3. train model with mlflow tracking
    run_id, model_save_path = train_model(config)

    # 4. evaluate the model and update the result on mlflow
    # TODO:
    evaluate_on_test_data(config, run_id, model_save_path)


if __name__ == "__main__":
    main()
