"""Unit test for evaluate."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate


@pytest.fixture
def mock_model():
    """Mock model prediction."""
    model = MagicMock()
    model.return_value = torch.tensor([1.0, 2.0])
    return model


@pytest.fixture
def mock_criterion():
    """Mock loss value returned."""
    criterion = MagicMock()
    criterion.return_value = torch.tensor(0.5)
    return criterion


@pytest.fixture
def mock_batch_size():
    """Evaluation batch size."""
    return 2


@pytest.fixture
def mock_testloader(mock_batch_size):
    """Mock test data loader."""
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
    )
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    dataset = TensorDataset(features, labels)
    testloader = DataLoader(dataset, batch_size=mock_batch_size)
    return testloader


@pytest.fixture
def mock_device():
    """Mock torch device.

    Only cpu is supported.
    """
    return torch.device("cpu")


@patch("src.evaluate.torch.no_grad")
def test_evaluate(
    mock_no_grad,
    mock_model,
    mock_criterion,
    mock_testloader,
    mock_device,
    mock_batch_size,
):
    """Test the evaluate function."""
    # Run the evaluate function
    loss = evaluate(mock_model, mock_criterion, mock_testloader, mock_device)

    # Verify that model.eval() was called once
    mock_model.eval.assert_called_once()

    # Verify that torch.no_grad() was called
    assert mock_no_grad.called

    # Verify that the loss calculation is correct
    expected_loss = (mock_batch_size * mock_criterion()) / len(
        mock_testloader.dataset
    )
    assert loss == expected_loss
