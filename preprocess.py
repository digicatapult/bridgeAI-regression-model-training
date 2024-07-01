"""Data split, preprocess and other data utilities."""

import joblib
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_and_split_data(config: dict, seed: int = 42) -> None:
    """Load a single data source and split it into train, test and val."""
    test_size = 0.2
    val_size = 0.2
    data = pd.read_csv(config["data"]["raw_data"])

    # Split features(X) and target(y) variable
    label_column = config["data"]["label_col"]
    X_all = data.drop(label_column, axis=1)
    y_all = data[label_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed
    )

    # Split the training data further into train and val data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed
    )

    # Combine features and labels into a dataframe
    X_train[config["data"]["label_col"]] = y_train
    X_test[config["data"]["label_col"]] = y_test
    X_val[config["data"]["label_col"]] = y_val

    # Save the split dataframes into csvs
    X_train.to_csv(config["data"]["train_data_save_path"], index=False)
    X_val.to_csv(config["data"]["val_data_save_path"], index=False)
    X_test.to_csv(config["data"]["test_data_save_path"], index=False)


def preprocess(
    data_path: str,
    config: dict,
    preprocessor=None,
    save_preprocessor: bool = False,
):
    """Preprocess the data and return a tuple of features and labels."""
    # Load the data from csv files with the correct column names
    data = pd.read_csv(data_path)

    # Define the target, numerical and categorical features
    label_col = config["data"]["label_col"]
    numeric_features = config["data"]["numeric_cols"]
    categorical_features = config["data"]["categoric_cols"]

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


def create_dataloader(features, labels, batch_size: int = 32, shuffle=False):
    """Create pytorch dataloader for the data."""
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
