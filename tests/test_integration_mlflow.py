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
    mlflow_uri = "http://localhost:5001"
    expt_name = "test-expt"
    model_register_name = "registered_name"
    model_alias = "champion"

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

    # Get the active run id
    run_id = mlflow_utils.get_activ_run_id()

    mlflow.pytorch.log_model(model, "model")

    # end run
    if mlflow.active_run():
        mlflow.end_run("FINISHED")

    model_uri = f"runs:/{run_id}/model"

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

    # 5. Test model promotion
    mlflow_utils.promote_model(model_uri, model_register_name, model_alias)

    # Check the latest version of the registered model
    latest_versions = client.get_latest_versions(model_register_name)
    assert len(latest_versions) == 1
    latest_version = latest_versions[0].version
    model_version_details = client.get_model_version(
        model_register_name, latest_version
    )
    # Check if the alias exists
    assert (
        model_alias in model_version_details.aliases
    ), f"Alias '{model_alias}' not found for model {model_register_name}."
