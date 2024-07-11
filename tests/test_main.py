"""Unit test for main pipeline."""

from unittest.mock import MagicMock, patch

from src.main import main


@patch("src.main.mlflow_utils")
@patch("src.main.mlflow")
@patch("src.main.utils.load_yaml_config")
@patch("src.main.load_and_split_data")
@patch("src.main.preprocess")
@patch("src.main.create_dataloader")
@patch("src.main.joblib.load")
@patch("src.main.NNModel")
@patch("src.main.train")
def test_main(
    mock_train,
    mock_NNModel,
    mock_joblib_load,
    mock_create_dataloader,
    mock_preprocess,
    mock_load_and_split_data,
    mock_load_yaml_config,
    mock_mlflow,
    mock_mlflow_utils,
):
    # Setup mock return values
    mock_load_yaml_config.return_value = {
        "data": {
            "raw_data": "data.csv",
            "train_data_save_path": "train.csv",
            "val_data_save_path": "val.csv",
            "test_data_save_path": "test.csv",
            "preprocessor_path": "preprocessor.joblib",
        },
        "model": {"train_batch_size": 16, "test_batch_size": 32},
        "mlflow": {"tracking_uri": "tracking_uri", "expt_name": "test_name"},
    }
    mock_load_and_split_data.return_value = None
    mock_preprocess.side_effect = [
        (MagicMock(), MagicMock()),  # train_x, train_y
        (MagicMock(), MagicMock()),  # val_x, val_y
        (MagicMock(), MagicMock()),  # test_x, test_y
    ]
    mock_create_dataloader.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    # train_dataloader, val_dataloader, test_dataloader
    mock_joblib_load.return_value = MagicMock()
    mock_NNModel.return_value = MagicMock()

    mock_mlflow_utils.return_value = MagicMock()
    mock_mlflow.return_value = MagicMock()

    # call main with the mocks
    main()

    # Asserting conditions
    mock_load_yaml_config.assert_called_once_with("./config.yaml")
    mock_load_and_split_data.assert_called_once_with(
        mock_load_yaml_config.return_value
    )
    assert mock_preprocess.call_count == 3
    assert mock_create_dataloader.call_count == 3
    mock_joblib_load.assert_called_once_with(
        mock_load_yaml_config.return_value["data"]["preprocessor_path"]
    )
    mock_NNModel.assert_called_once()
    mock_train.assert_called_once()
