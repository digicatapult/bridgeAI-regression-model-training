"""MLFlow utility functions."""

import time
from types import SimpleNamespace

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import pandas as pd
import toml
import torch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.preprocess import schema
from src.servable_model import ServableModel
from src.utils import get_device


def start_run(tracking_uri: str, experiment_name: str):
    """Start run of mlflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def get_activ_run_id():
    """Get mlflow active run id."""
    active_run = mlflow.active_run()
    return active_run.info.run_id


def log_param_dict(config: dict):
    """Load parameters that are in dict format."""
    for key, value in config.items():
        if isinstance(value, dict):
            for k, v in value.items():
                mlflow.log_param(k, v)
        else:
            mlflow.log_param(key, value)


def log_torch_model(
    run,
    model_save_path,
    config,
):
    """Logs the given PyTorch model to MLflow Model Registry."""

    feature_cols = (
        config["data"]["categorical_cols"] + config["data"]["numeric_cols"]
    )

    sample_data = pd.read_csv(
        config["dvc"]["val_data_path"], usecols=feature_cols, nrows=4
    )
    sample_data = schema.validate(sample_data)
    preprocessor_save_path = config["data"]["preprocessor_path"]

    # Use custom model to predict on the sample data to infer signature
    device = get_device(config)

    model = ServableModel(device)
    # Create a simulated context object
    context = SimpleNamespace(
        artifacts={
            "torch_model": model_save_path,
            "preprocessor": preprocessor_save_path,
        }
    )
    # Update the context of the custom model
    model.load_context(context)
    # Do inference on sample data to get sample output
    with torch.no_grad():
        sample_output = model.predict(context=context, model_input=sample_data)
    signature = infer_signature(sample_data, sample_output)

    # Log the PyTorch model with dependancies
    mlflow.pyfunc.log_model(
        artifact_path="custom_pytorch_model",
        python_model=ServableModel(device),
        artifacts={
            "torch_model": str(model_save_path),
            "preprocessor": preprocessor_save_path,
        },
        signature=signature,
        conda_env=get_conda_env(),
        infer_code_paths=False,
        code_paths=["./src"],
    )
    model_uri = f"runs:/{run.info.run_id}/{config['model']['model_name']}"
    return model_uri


def flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_conda_env(toml_path: str = "./pyproject.toml") -> dict:
    """Get the conda environment details from the pyproject."""
    # Load the pyproject.toml file
    pyproject = toml.load(toml_path)

    # Extract dependencies from the [tool.poetry] of the pyproject file
    dependencies = pyproject["tool"]["poetry"].get("dependencies", [])
    # Remove the redundant ones
    dependencies.pop("python", None)
    dependencies.pop("mlflow", None)

    # Convert the gathered dependancies into proper foramt
    dependencies = [
        f"{pkg}>={ver.lstrip('^')}" for pkg, ver in dependencies.items()
    ]

    conda_env = mlflow.pyfunc.get_default_conda_env()
    conda_env["dependencies"][-1]["pip"].extend(dependencies)
    return conda_env


def promote_model(model_uri, model_register_name, model_alias):
    """Register model and add alias - for deployment."""
    # Initialise MLflow client
    client = MlflowClient()

    # Register the model in the registry if it doesn't already exist
    result = mlflow.register_model(model_uri, model_register_name)

    # Wait for the registration process to complete
    while True:
        model_version_details = client.get_model_version(
            name=model_register_name, version=result.version
        )
        status = model_version_details.status
        if status == "READY":
            break
        time.sleep(1)

    # Wait until the model version is ready
    model_version_number = result.version

    status = None
    while status != "READY":
        model_version_details = client.get_model_version(
            name=model_register_name, version=model_version_number
        )
        status = model_version_details.status
        if status == "READY":
            print(
                f"Model version {model_register_name}-"
                f"{model_version_number} is ready."
            )
            break
        elif status == "FAILED_REGISTRATION":
            raise Exception(
                f"Model {model_register_name} - "
                f"failed to register new version."
            )
        else:
            print(f"Current status: {status}. Waiting...")
            time.sleep(5)

    # Assign alias to this model version
    client.set_registered_model_alias(
        name=model_register_name,
        alias=model_alias,
        version=model_version_number,
    )
    print(
        f"Alias '{model_alias}' set to model '{model_register_name}' "
        f"version {model_version_number}."
    )
