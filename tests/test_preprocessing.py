"""Unit tests for data preprocessing."""

from unittest.mock import ANY, patch

import pandas as pd
import pytest
import torch

from src.preprocess import create_dataloader, preprocess


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "data": {
            "raw_data": "data.csv",
            "label_col": "target",
            "train_data_save_path": "train.csv",
            "val_data_save_path": "val.csv",
            "test_data_save_path": "test.csv",
            "preprocessor_path": "preprocessor.joblib",
            "numeric_cols": ["num1", "num2"],
            "categorical_cols": ["cat1", "cat2"],
        }
    }


@pytest.fixture
def mock_data():
    """Create mock dataframe."""
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [6, 7, 8, 9, 10],
            "cat1": ["A", "B", "A", "B", "C"],
            "cat2": ["X", "Y", "X", "Y", "Z"],
            "target": [0, 1, 0, 1, 0],
        }
    )


@patch("src.preprocess.pd.read_csv")
@patch("src.preprocess.joblib.dump")
def test_preprocess(mock_joblib_dump, mock_read_csv, mock_config, mock_data):
    """Test preprocess function."""
    mock_read_csv.return_value = mock_data
    mock_joblib_dump.return_value = None

    features, labels = preprocess(
        mock_config["data"]["train_data_save_path"],
        mock_config,
        save_preprocessor=True,
    )

    # Ensure read_csv is called with the correct path
    mock_read_csv.assert_called_once_with(
        mock_config["data"]["train_data_save_path"]
    )

    # Ensure joblib.dump is called once to save the preprocessor
    mock_joblib_dump.assert_called_once_with(
        ANY, mock_config["data"]["preprocessor_path"]
    )

    # Ensure the returned features and labels have the correct shapes
    assert features.shape[0] == mock_data.shape[0]
    assert labels.shape[0] == mock_data.shape[0]


@patch("src.preprocess.pd.read_csv")
@patch("src.preprocess.joblib.dump")
def test_create_dataloader(
    mock_joblib_dump, mock_read_csv, mock_config, mock_data
):
    """Test create_dataloader function."""
    batch_size = 2
    mock_read_csv.return_value = mock_data
    mock_joblib_dump.return_value = None

    features, labels = preprocess(
        mock_config["data"]["train_data_save_path"],
        mock_config,
        save_preprocessor=True,
    )

    dataloader = create_dataloader(
        features, labels, batch_size=batch_size, shuffle=True
    )

    # Ensure the dataloader is an instance of DataLoader
    assert isinstance(dataloader, torch.utils.data.DataLoader)

    # Ensure the dataloader contains the correct number of batches
    # using ceil division
    assert len(dataloader) == -(-mock_data.shape[0] // batch_size)

    # Ensure the first batch has the correct shape
    features, labels = next(iter(dataloader))
    number_of_features_after_oh_encoding = 8
    assert features.shape == torch.Size(
        [2, number_of_features_after_oh_encoding]
    )
    assert labels.shape == torch.Size([2])
