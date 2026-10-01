"""Model training, evaluation via CV, and tracking with MLflow."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.client import MlflowClient
from src.data import get_splits

RF_CONFIGS = [
    {"n_estimators": 50, "max_depth": None},
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 5},
]

BEST_PARAMS = {"n_estimators": 100, "max_depth": None}

GB_CONFIGS = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
]


def run_cv(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Run 5-fold StratifiedKFold CV and return averaged metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "train_f1_macro": [], "train_accuracy": [], "train_log_loss": [],
        "val_f1_macro": [], "val_accuracy": [], "val_log_loss": []
    }

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)

        pred_tr = model.predict(X_tr)
        prob_tr = model.predict_proba(X_tr)

        pred_val = model.predict(X_val)
        prob_val = model.predict_proba(X_val)

        metrics["train_f1_macro"].append(f1_score(y_tr, pred_tr, average="macro"))
        metrics["train_accuracy"].append(accuracy_score(y_tr, pred_tr))
        metrics["train_log_loss"].append(log_loss(y_tr, prob_tr, labels=[0, 1, 2]))

        metrics["val_f1_macro"].append(f1_score(y_val, pred_val, average="macro"))
        metrics["val_accuracy"].append(accuracy_score(y_val, pred_val))
        metrics["val_log_loss"].append(log_loss(y_val, prob_val, labels=[0, 1, 2]))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train_and_log(config, family, run_name, X_train, y_train):
    """Run CV, refit, log to MLflow and return metrics & model URI."""
    with mlflow.start_run(run_name=run_name) as run:
        if family == "RandomForest":
            model = RandomForestClassifier(random_state=42, **config)
        else:
            model = GradientBoostingClassifier(random_state=42, **config)

        metrics = run_cv(model, X_train, y_train)

        mlflow.log_param("model_family", family)
        mlflow.log_params(config)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_family", family)
        mlflow.set_tag("config_name", run_name)

        # Refit on full train set
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "family": family,
            "config": config,
            "metrics": metrics,
            "model_uri": model_info.model_uri
        }


def main():
    """Main training routine."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Wine-Cultivar-Classification")

    X_train, _, y_train, _ = get_splits()

    results = []

    for conf in RF_CONFIGS:
        name = f"rf_n{conf['n_estimators']}_d{conf['max_depth']}"
        res = train_and_log(conf, "RandomForest", name, X_train, y_train)
        results.append(res)

    for conf in GB_CONFIGS:
        name = f"gb_n{conf['n_estimators']}_lr{conf['learning_rate']}_d{conf['max_depth']}"
        res = train_and_log(conf, "GradientBoosting", name, X_train, y_train)
        results.append(res)

    best_run = max(results, key=lambda x: x["metrics"]["val_f1_macro"])

    mv = mlflow.register_model(best_run["model_uri"], "WineClassifier")
    client = MlflowClient()
    client.set_registered_model_alias("WineClassifier", "champion", mv.version)

    print("| Family | Config | Train F1 | Val F1 | Train Acc | Val Acc | "
          "Train LogLoss | Val LogLoss |")
    print("|--------|--------|----------|--------|-----------|---------|-"
          "--------------|-------------|")
    for r in results:
        fam = r["family"]
        conf = str(r["config"])
        m = r["metrics"]
        print(
            f"| {fam} | {conf} | {m['train_f1_macro']:.4f} | {m['val_f1_macro']:.4f} "
            f"| {m['train_accuracy']:.4f} | {m['val_accuracy']:.4f} | "
            f"{m['train_log_loss']:.4f} | {m['val_log_loss']:.4f} |"
        )

    print(f"\nRegistered champion model: {best_run['run_name']} "
          f"(Val F1: {best_run['metrics']['val_f1_macro']:.4f})")


if __name__ == "__main__":
    main()
