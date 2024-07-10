"""Integration test for airflow logging."""

import subprocess
import time
from unittest.mock import MagicMock, patch

import mlflow
import pytest
import requests
from mlflow.tracking import MlflowClient

DOCKER_COMPOSE_FILE = "./tests/docker-compose-integration-test-airflow.yaml"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Start the docker compose services."""
    subprocess.run(
        ["docker-compose", "-f", DOCKER_COMPOSE_FILE, "up", "--build", "-d"],
        check=True,
    )

    # Wait for the MLflow server to be up and running
    time.sleep(30)

    yield

    # Tear down the docker compose services
    subprocess.run(
        ["docker-compose", "-f", DOCKER_COMPOSE_FILE, "down"], check=True
    )


@patch("src.main.load_and_split_data")
@patch("src.main.preprocess")
@patch("src.main.create_dataloader")
@patch("src.main.joblib.load")
@patch("src.main.NNModel")
@patch("src.main.utils.load_yaml_config")
def test_training_logs_to_mlflow(
    mock_config,
    mock_NNModel,
    mock_joblib_load,
    mock_create_dataloader,
    mock_preprocess,
    mock_load_and_split_data,
):
    """Integration test for airflow.

    This function tests:
        1. The integration to airflow
        2. Logging of parameters
        3. Logging of metrics
        4. Logging of model and registering in MLFlow registry
    """
    mlflow_uri = "http://localhost:5000"
    expt_name = "test_expt"
    model_name = "test-model"
    mock_config.return_value = {
        "data": {
            "raw_data": "data.csv",
            "train_data_save_path": "train.csv",
            "val_data_save_path": "val.csv",
            "test_data_save_path": "test.csv",
            "preprocessor_path": "preprocessor.joblib",
        },
        "model": {
            "model_name": model_name,
            "train_batch_size": 4,
            "test_batch_size": 4,
        },
        "mlflow": {"tracking_uri": mlflow_uri, "expt_name": expt_name},
    }
    mock_load_and_split_data.return_value = MagicMock()
    mock_preprocess.side_effect = [
        (MagicMock(), MagicMock()),  # train_x, train_y
        (MagicMock(), MagicMock()),  # val_x, val_y
        (MagicMock(), MagicMock()),  # test_x, test_y
    ]
    mock_create_dataloader.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    # train_dataloader, val_dataloader, test_dataloader
    mock_joblib_load.return_value = MagicMock()
    mock_NNModel.return_value = MagicMock()

    result = subprocess.run(
        [
            "docker-compose",
            "-f",
            DOCKER_COMPOSE_FILE,
            "exec",
            "training",
            "poetry",
            "run",
            "pytest",
            "./tests/mock_main.py",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    # check if experiment is logged
    mlflow.set_tracking_uri(mlflow_uri)
    experiment = mlflow.get_experiment(experiment_id="0")
    assert experiment is not None

    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) == 1

    # 1. Check if the MLFlow connection worked as expected
    # Check the status of the run
    run_id = runs["run_id"][0]
    run_data = requests.get(
        f"{mlflow_uri}/api/2.0/mlflow/runs/get?run_id={run_id}"
    ).json()
    assert run_data["run"]["info"]["status"] == "FINISHED"

    # 2. Check if parameters are logged
    expected_param = {"key": "mlflow.expt_name", "value": expt_name}
    assert expected_param in run_data["run"]["data"]["params"]

    # 3. Check the reported test_loss metrics is present
    assert len(run_data["run"]["data"]["metrics"]) > 0

    # 4. Check if the model is registered properly
    client = MlflowClient()

    # Check the latest version of the model
    registered_model = dict(
        client.search_model_versions(f"name='{model_name}'")[0]
    )
    assert registered_model["name"] == model_name
    assert registered_model["version"] == "1"
    assert registered_model["status"] == "READY"
