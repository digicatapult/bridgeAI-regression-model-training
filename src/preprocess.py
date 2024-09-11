"""Data split, preprocess and other data utilities."""

import pickle
from pathlib import Path

import joblib
import pandas as pd
import torch
from pandera import Column, DataFrameSchema
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src import utils

schema = DataFrameSchema(
    {
        "area": Column(float),
        "bedrooms": Column(int),
        "bathrooms": Column(int),
        "stories": Column(int),
        "mainroad": Column(str),
        "guestroom": Column(str),
        "basement": Column(str),
        "hotwaterheating": Column(str),
        "airconditioning": Column(str),
        "parking": Column(int),
        "prefarea": Column(str),
        "furnishingstatus": Column(str),
    },
    coerce=True,
)


def preprocess(
    data_path: str,
    config: dict,
    preprocessor=None,
    save_preprocessor: bool = False,
):
    """Preprocess the data and return a tuple of features and labels."""
    # Load the data from csv files with the correct column names
    data = pd.read_csv(data_path)
    data = schema.validate(data)

    # Define the target, numerical and categorical features
    label_col = config["data"]["label_col"]
    numeric_features = config["data"]["numeric_cols"]
    categorical_features = config["data"]["categorical_cols"]

    # defining labels and features
    labels = data[label_col].values
    features = data.columns.drop(label_col)

    if preprocessor is None:
        # Define preprocessing steps for numerical and categorical features
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine the preprocessing steps to form the final pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Fit the preprocessor on the data
        preprocessor.fit(data[features])

    # Transform the data using the preprocessor
    data_preprocessed = preprocessor.transform(data[features])

    # Save the preprocessor
    if save_preprocessor:
        joblib.dump(preprocessor, config["data"]["preprocessor_path"])

    return data_preprocessed, labels


def preprocess_datasets(config):
    """Preprocess the split of data and save them."""
    # [Train data] preprocess train data and save the preprocessing steps
    train_x, train_y = preprocess(
        config["dvc"]["train_data_path"],
        config,
        preprocessor=None,
        save_preprocessor=True,
    )

    with open(
        Path(config["dvc"]["train_data_path"]).with_suffix(".pkl"), "wb"
    ) as f:
        pickle.dump((train_x, train_y), f)

    # Load the preprocessor created while preparing the train data
    # This is needed to apply the same preprocessing to other data
    preprocessor = joblib.load(config["data"]["preprocessor_path"])

    # [val data] preprocess and create data loader
    # preprocess val data and prepare validation data loader
    val_x, val_y = preprocess(
        config["dvc"]["val_data_path"], config, preprocessor=preprocessor
    )
    with open(
        Path(config["dvc"]["val_data_path"]).with_suffix(".pkl"), "wb"
    ) as f:
        pickle.dump((val_x, val_y), f)

    # [test data] preprocess and create data loader
    test_x, test_y = preprocess(
        config["dvc"]["test_data_path"],
        config,
        preprocessor=preprocessor,
    )
    with open(
        Path(config["dvc"]["test_data_path"]).with_suffix(".pkl"), "wb"
    ) as f:
        pickle.dump((test_x, test_y), f)


def create_dataloader(features, labels, batch_size: int = 32, shuffle=False):
    """Create pytorch dataloader for the data."""
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    config = utils.load_yaml_config()
    preprocess_datasets(config)
