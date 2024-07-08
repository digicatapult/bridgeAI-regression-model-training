import mlflow
import mlflow.pytorch
import torch
from mlflow.models.signature import infer_signature


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
    model,
    dataloader,
    run,
    model_name,
):
    """Logs the given PyTorch model to MLflow Model Registry."""
    # Infer the signature from the sample data
    model.eval()
    sample_x, _ = next(iter(dataloader))
    with torch.no_grad():
        example_output = model(sample_x)
    signature = infer_signature(sample_x.numpy(), example_output.numpy())

    # Log the PyTorch model with the inferred signature
    mlflow.pytorch.log_model(
        model,
        model_name,
        signature=signature,
    )
    model_uri = f"runs:/{run.info.run_id}/{model_name}"

    # Register the model in Model Registry for versioning and stage management
    mlflow.register_model(model_uri=model_uri, name=model_name)
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
