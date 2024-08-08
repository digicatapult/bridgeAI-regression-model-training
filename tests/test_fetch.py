import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from src.fetch_data import fetch_data

TEMP_DIR = "./repo"


@pytest.fixture
def config():
    """Fixture to provide configuration for the tests."""
    return {
        "dvc": {
            "git_repo_url": "https://github.com/user/repo",
        }
    }


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: Create temp dir
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    yield

    # Teardown - Remove the temp dir
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


@patch("src.fetch_data.get_authenticated_github_url")
@patch("src.fetch_data.Repo.clone_from")
@patch("src.fetch_data.Repo")
@patch("src.fetch_data.checkout_data")
@patch("src.fetch_data.dvc_pull")
@patch("src.fetch_data.move_dvc_data")
def test_fetch_data(
    mock_move_dvc_data,
    mock_dvc_pull,
    mock_checkout_data,
    mock_Repo,
    mock_repo_clone_from,
    mock_get_authenticated_github_url,
    config,
    setup_and_teardown,
):
    """Test the fetch_data function."""
    # Set up mocks
    authed_repo_url = "https://authed_url/repo.git"
    mock_get_authenticated_github_url.return_value = authed_repo_url

    mock_repo = MagicMock()
    mock_repo.bare = False
    mock_Repo.return_value = mock_repo

    # Run the function
    fetch_data(config)

    # Assertions
    mock_get_authenticated_github_url.assert_called_once_with(
        config["dvc"]["git_repo_url"]
    )
    mock_repo_clone_from.assert_called_once_with(authed_repo_url, TEMP_DIR)
    mock_checkout_data.assert_called_once_with(mock_repo, config)
    mock_dvc_pull.assert_called_once_with(config)
    mock_move_dvc_data.assert_called_once_with(config)
