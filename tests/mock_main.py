"""Dummy mocked main for MLFlow integration test."""

from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.main import main


class SimpleNN(torch.nn.Module):
    """A simple one-layer neural network."""

    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


@patch("src.main.utils.load_yaml_config")
@patch("src.main.load_and_split_data")
@patch("src.main.preprocess")
@patch("src.main.create_dataloader")
@patch("src.main.joblib.load")
@patch("src.main.NNModel")
@patch("src.train.torch.optim.Adam")
@patch("src.train.evaluate")
def test_main(
    mock_eval,
    mock_opt,
    mock_NNModel,
    mock_joblib_load,
    mock_create_dataloader,
    mock_preprocess,
    mock_load_and_split_data,
    mock_load_yaml_config,
):
    # Setup mock return values
    mlflow_uri = "http://localhost:5000"
    expt_name = "test_expt"

    mock_load_yaml_config.return_value = {
        "data": {
            "raw_data": "data.csv",
            "train_data_save_path": "train.csv",
            "val_data_save_path": "val.csv",
            "test_data_save_path": "test.csv",
            "preprocessor_path": "preprocessor.joblib",
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
    }
    mock_eval.return_value = 1.1
    mock_opt.return_value = MagicMock()
    mock_load_and_split_data.return_value = None
    mock_preprocess.return_value = (MagicMock(), MagicMock())

    input_dim = 8
    output_dim = 1
    mock_NNModel.return_value = SimpleNN(input_dim, output_dim)

    # Create a TensorDataset from the tensor
    dataset = TensorDataset(
        torch.randn(4, input_dim), torch.randn(4, output_dim)
    )
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=4)
    mock_create_dataloader.return_value = dataloader

    # train_dataloader, val_dataloader, test_dataloader
    mock_joblib_load.return_value = MagicMock()

    # call main with the mocks
    main()
