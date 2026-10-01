"""Tests for the data module."""

import pytest
import pandas as pd
import numpy as np

from src.data import load_data, validate_data, get_splits


def test_split_sizes():
    """Test if the splits have the correct number of samples."""
    X_train, X_test, y_train, y_test = get_splits()
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_train) == 142
    assert len(y_test) == 36


def test_no_nulls():
    """Test that the loaded data has no null values."""
    X, y = load_data()
    assert not X.isnull().values.any()
    assert not y.isnull().values.any()


def test_feature_count():
    """Test that the loaded data has exactly 13 features."""
    X, y = load_data()
    assert X.shape[1] == 13


def test_stratification():
    """Test that the class proportions in train and test are similar to the full dataset."""
    X_train, X_test, y_train, y_test = get_splits()
    _, y_full = load_data()

    full_props = y_full.value_counts(normalize=True).sort_index()
    train_props = y_train.value_counts(normalize=True).sort_index()
    test_props = y_test.value_counts(normalize=True).sort_index()

    # Proportions should be close
    np.testing.assert_allclose(full_props, train_props, atol=0.05)
    np.testing.assert_allclose(full_props, test_props, atol=0.05)


def test_reproducibility():
    """Test that two calls to get_splits return identical results."""
    X_train1, X_test1, y_train1, y_test1 = get_splits()
    X_train2, X_test2, y_train2, y_test2 = get_splits()

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_validate_data_raises_on_bad_data():
    """Test that validate_data raises ValueError on invalid inputs."""
    X, y = load_data()

    # Test wrong feature count
    X_bad_features = X.iloc[:, :-1]
    with pytest.raises(ValueError, match="Expected 13 features"):
        validate_data(X_bad_features, y)

    # Test nulls in X
    X_nulls = X.copy()
    X_nulls.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="null values"):
        validate_data(X_nulls, y)

    # Test nulls in y
    y_nulls = y.copy()
    y_nulls.iloc[0] = np.nan
    with pytest.raises(ValueError, match="null values"):
        validate_data(X, y_nulls)
