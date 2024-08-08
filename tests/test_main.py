"""Unit test for main pipeline."""

from unittest.mock import patch

from src.main import main


@patch("src.main.evaluate_on_test_data")
@patch("src.main.fetch_data")
@patch("src.main.utils.load_yaml_config")
@patch("src.main.preprocess_datasets")
@patch("src.main.train_model")
def test_main(
    mock_train_model,
    mock_preprocess_datasets,
    mock_load_yaml_config,
    mock_fetch_data,
    mock_evaluate,
):
    # call main with the mocks
    main()

    # Asserting all the steps are called
    mock_load_yaml_config.assert_called_once()
    mock_fetch_data.assert_called_once()
    mock_preprocess_datasets.assert_called_once()
    mock_train_model.assert_called_once()
    mock_evaluate.assert_called_once()
