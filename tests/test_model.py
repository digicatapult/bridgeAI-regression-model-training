"""Unit tests for neural network model."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import NNModel


@pytest.fixture
def input_dim():
    return 10


@pytest.fixture
def model(input_dim):
    return NNModel(input_dim)


def test_forward_pass(input_dim, model):
    """Test if the forward pass works correctly."""
    # A dummy batch of data (batch_size=4) with `input_dim` features
    input_data = torch.randn(4, input_dim)
    output_data = model(input_data)
    assert output_data.shape == (4, 1)


def test_forward_pass_fail(model, input_dim):
    """Test if the forward pass fails when incorrect number of features.

    Number of features in the input should always match
    to what it is initialised to.
    """
    # A batch of data (batch_size=4) with incorrect number of features
    input_data = torch.randn(4, input_dim + 1)
    with pytest.raises(RuntimeError):
        model(input_data)


def test_training_step():
    """Tests a single training step.

    Check if it generates a valid loss tensor.
    """
    input_dim = 8
    model = NNModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Batch size of 5
    input_data = torch.randn(5, input_dim)
    # Batch size of 5, output dimension of 1
    target_data = torch.randn(5, 1)

    # Forward pass
    output_data = model(input_data)
    loss = criterion(output_data, target_data)

    # Backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if the loss is a valid tensor
    assert isinstance(loss, torch.Tensor)
