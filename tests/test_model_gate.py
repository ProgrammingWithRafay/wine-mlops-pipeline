"""Model quality gate tests for CI/CD."""

import time
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data import get_splits
from src.train import run_cv, BEST_PARAMS

MIN_F1 = 0.88
MAX_LATENCY_MS = 30


@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train the model once for the whole module and provide splits."""
    X_train, X_test, y_train, y_test = get_splits()

    model = RandomForestClassifier(**BEST_PARAMS, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


def test_metric_gate():
    """
    Metric gate protects against deploying models with degraded predictive performance.
    Validates that the chosen champion configuration hits our target macro F1 score.
    """
    X_train, _, y_train, _ = get_splits()
    model = RandomForestClassifier(**BEST_PARAMS, random_state=42, n_jobs=1)

    metrics = run_cv(model, X_train, y_train)
    val_f1 = metrics["val_f1_macro"]

    assert val_f1 >= MIN_F1, f"Validation F1 {val_f1:.4f} is below threshold {MIN_F1}"


def test_latency_gate(trained_model_and_data):
    """
    Latency gate protects against releasing models that are too slow to run.
    Ensures inference time is within acceptable SLA bounds.
    """
    model, _, _, X_test, _ = trained_model_and_data

    # Warm up
    model.predict(X_test)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(X_test)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    median_latency = np.median(times)
    assert median_latency <= MAX_LATENCY_MS, \
        f"Median latency {median_latency:.2f}ms exceeds {MAX_LATENCY_MS}ms"


def test_schema_gate(trained_model_and_data):
    """
    Output schema gate protects against downstream consumer crashes.
    Ensures prediction formats and values remain stable.
    """
    model, _, _, X_test, _ = trained_model_and_data

    predictions = model.predict(X_test)

    # Check shape
    assert predictions.shape[0] == X_test.shape[0], "Must return one prediction per input row"

    # Check type
    assert np.issubdtype(predictions.dtype, np.integer), "Predictions must have an integer dtype"

    # Check classes
    valid_classes = {0, 1, 2}
    unique_preds = set(np.unique(predictions))
    assert unique_preds.issubset(valid_classes), \
        f"Predicted values {unique_preds} are not in {valid_classes}"
