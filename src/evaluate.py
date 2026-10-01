"""Evaluate the champion model on the test split."""

import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, accuracy_score, log_loss
from src.data import get_splits


def main():
    """Load model, evaluate on test set and print metrics."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    _, X_test, _, y_test = get_splits()

    model_uri = "models:/WineClassifier@champion"
    model = mlflow.sklearn.load_model(model_uri)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    f1 = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)
    loss = log_loss(y_test, prob, labels=[0, 1, 2])

    print("Test Evaluation Results:")
    print(f"Macro F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
