"""Dummy mocked main for MLFlow integration test."""

import mlflow
import requests
import torch
from mlflow.tracking import MlflowClient

from src import mlflow_utils


class SimpleNN(torch.nn.Module):
    """A simple one-layer neural network."""

    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


def test_integration_mlflow():
    # Setup mock return values
    mlflow_uri = "http://localhost:5000"
    expt_name = "test-expt"
    model_name = "test-model"

    config = {
        "data": {
            "label_col": "price",
            "categorical_cols": ["col1", "col2"],
            "numeric_cols": ["col3", "col4", "col5"],
            "preprocessor_path": "./artefacts/test-preprocessor.joblib",
        },
        "model": {
            "model_name": "test-model",
            "save_path": "./artefacts/",
            "train_batch_size": 4,
            "test_batch_size": 4,
            "n_epochs": 0,
            "learning_rate": 0.01,
            "es_patience": 10,
            "es_delta": 0,
            "use_gpu": False,
        },
        "mlflow": {
            "tracking_uri": mlflow_uri,
            "expt_name": expt_name,
        },
        "dvc": {
            "git_repo_url": "https://sample_url",
        },
    }

    # Start a run and log some values
    mlflow_utils.start_run(mlflow_uri, config["mlflow"]["expt_name"])
    mlflow_utils.log_param_dict(mlflow_utils.flatten_dict(config))
    mlflow.log_metric("loss", 1.1, step=1)

    # log model and get the model uri
    input_dim = 8
    output_dim = 1

    model = SimpleNN(input_dim, output_dim)

    # Get the active run id and end run
    run_id = mlflow_utils.get_activ_run_id()
    if mlflow.active_run():
        mlflow.end_run("FINISHED")

    mlflow.pytorch.log_model(model, "model")
    mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

    # check if experiment is logged
    mlflow.set_tracking_uri(mlflow_uri)
    experiment = mlflow.get_experiment(experiment_id="0")
    assert experiment is not None

    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) >= 1

    # 1. Check if the MLFlow connection worked as expected
    # Check the status of the run
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
    assert int(registered_model["version"]) >= 1
    assert registered_model["status"] == "READY"
