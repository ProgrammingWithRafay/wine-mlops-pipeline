"""Data loading and preprocessing module."""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Wine dataset into a pandas DataFrame and Series."""
    wine = load_wine(as_frame=True)
    return wine.data, wine.target


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate that the dataset meets expected criteria."""
    if X.isnull().values.any():
        raise ValueError("Dataset contains null values in features.")
    if y.isnull().values.any():
        raise ValueError("Dataset contains null values in target.")
    if X.shape[1] != 13:
        raise ValueError(f"Expected 13 features, got {X.shape[1]}")


def get_splits():
    """Load, validate, and split the data into train and test sets."""
    X, y = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
