"""Fetch the data from featurestore."""

import os
import shutil
from pathlib import Path

from dvc.cli import main as dvc_main
from git import GitCommandError, Repo

from src import utils
from src.utils import logger


def checkout_data(repo, config):
    """Checkout to the data branch."""
    repo.git.fetch()
    data_version = os.getenv("DATA_VERSION", config["dvc"]["data_version"])
    try:
        try:
            repo.git.checkout(data_version)
        except GitCommandError:
            # If branch does not exist, create it
            repo.git.checkout("HEAD", b=config["dvc"]["git_branch"])
    except Exception as e:
        logger.error(f"Git data version checkout failed with error: {e}")
        raise e


def get_authenticated_github_url(base_url):
    """From the base git http url, generate an authenticated url."""
    username = os.getenv("GITHUB_USERNAME")
    password = os.getenv("GITHUB_PASSWORD")

    if not username or not password:
        logger.error(
            "GITHUB_USERNAME or GITHUB_PASSWORD environment variables not set"
        )
        raise ValueError(
            "GITHUB_USERNAME or GITHUB_PASSWORD environment variables not set"
        )

    # Separate protocol and the rest of the URL
    protocol, rest_of_url = base_url.split("://")

    # Construct the new URL with credentials
    new_url = f"{protocol}://{username}:{password}@{rest_of_url}"

    return new_url


def delete_file_if_exists(file_path):
    """Delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)


def dvc_pull(config):
    """DVC pull."""
    # first remove if older data exists
    delete_file_if_exists(config["dvc"]["train_data_path"])
    delete_file_if_exists(config["dvc"]["test_data_path"])
    delete_file_if_exists(config["dvc"]["val_data_path"])
    dvc_remote = os.getenv("DVC_REMOTE", config["dvc"]["dvc_remote"])
    try:
        dvc_remote_add(config)
        dvc_main(["pull", "-r", dvc_remote])
    except Exception as e:
        logger.error(f"DVC pull failed with error: {e}")
        raise e


def dvc_remote_add(config):
    """Set the dvc remote."""
    access_key_id = os.getenv("DVC_ACCESS_KEY_ID")
    secret_access_key = os.getenv("DVC_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION")
    try:
        dvc_remote_name = os.getenv(
            "DVC_REMOTE_NAME", config["dvc"]["dvc_remote_name"]
        )
        dvc_remote = os.getenv("DVC_REMOTE", config["dvc"]["dvc_remote"])
        dvc_endpoint_url = os.getenv(
            "DVC_ENDPOINT_URL", config["dvc"]["dvc_endpoint_url"]
        )

        dvc_main(["remote", "add", "-f", dvc_remote_name, dvc_remote])
        dvc_main(
            [
                "remote",
                "modify",
                dvc_remote_name,
                "endpointurl",
                dvc_endpoint_url,
            ]
        )
        if secret_access_key is None or secret_access_key == "":
            # Set dvc remote credentials
            # only when a valid secret access key is present
            logger.warning(
                "AWS credentials `dvc_secret_access_key` is missing "
                "in the Airflow connection."
            )
        else:
            dvc_main(
                [
                    "remote",
                    "modify",
                    dvc_remote_name,
                    "access_key_id",
                    access_key_id,
                ]
            )
            dvc_main(
                [
                    "remote",
                    "modify",
                    dvc_remote_name,
                    "secret_access_key",
                    secret_access_key,
                ]
            )
        # Minio does not enforce regions but DVC requires it
        dvc_main(["remote", "modify", dvc_remote_name, "region", region])
    except Exception as e:
        logger.error(f"DVC remote add failed with error: {e}")
        raise e


def move_dvc_data(config):
    """Move pulled dvc data to where it is expected to be."""
    # First delete if the destination has files with same name
    destination = Path("../artefacts/.")
    delete_file_if_exists(
        destination / Path(config["dvc"]["train_data_path"]).name
    )
    delete_file_if_exists(
        destination / Path(config["dvc"]["test_data_path"]).name
    )
    delete_file_if_exists(
        destination / Path(config["dvc"]["val_data_path"]).name
    )

    # Now move the data
    try:
        shutil.move(config["dvc"]["train_data_path"], destination)
        shutil.move(config["dvc"]["test_data_path"], destination)
        shutil.move(config["dvc"]["val_data_path"], destination)
    except Exception as e:
        logger.error(f"Copying dvc data failed with error {e}")
        raise e


def fetch_data(config):
    """Fetch the versioned data from dvc."""
    # 1. Authenticate, clone, and update git repo
    authenticated_git_url = get_authenticated_github_url(
        config["dvc"]["git_repo_url"]
    )
    repo_temp_path = "./repo"
    Repo.clone_from(authenticated_git_url, repo_temp_path)

    os.chdir(repo_temp_path)

    # 2. Initialise git and dvc
    repo = Repo("./")
    assert not repo.bare

    # 3. Checkout to the data version
    checkout_data(repo, config)

    # 4. DVC pull
    dvc_pull(config)

    # 5. move the pulled data to expected location
    move_dvc_data(config)
    os.chdir("../")


if __name__ == "__main__":
    config = utils.load_yaml_config()
    fetch_data(config)
