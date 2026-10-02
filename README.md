# Wine MLOps Pipeline

![CI](https://github.com/ProgrammingWithRafay/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

## Project Overview
This project implements an MLOps pipeline for a Wine Classifier. It automates data processing, model training, evaluation, and logging using MLflow. The pipeline is designed with continuous integration (CI) via GitHub Actions to maintain code quality and ensure only high-performing models pass strict evaluation criteria.

## Repository Structure
- `.github/workflows/`: Contains GitHub Actions workflows for continuous integration (CI).
- `data/`: Directory for storing the dataset.
- `src/`: Source code for the pipeline operations.
  - `config.py`: Configuration settings and logging setup.
  - `data.py`: Data loading and preprocessing scripts.
  - `evaluate.py`: Model evaluation logic.
  - `train.py`: Model training logic and MLflow tracking integration.
- `tests/`: Unit and integration tests to validate pipeline functionality.
- `Makefile`: Make commands for quick project setup, linting, testing, and pipeline execution.
- `requirements.txt`: Python package dependencies.

## Setup

### Windows (PowerShell)
```powershell
# Create a Python 3.10 virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Execute pipeline steps via Makefile
make install
make lint
make test
make train
make evaluate
```

### Linux
```bash
# Create a Python 3.10 virtual environment
python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Execute pipeline steps via Makefile
make install
make lint
make test
make train
make evaluate
```

## MLflow UI
To view the logged runs, metrics, and models, you can open the MLflow UI by running the following command:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` in your web browser.

## MLflow Experiment & WineClassifier
The pipeline tracks all model training and evaluation runs within an MLflow experiment. The models are logged under the name `WineClassifier`. The best performing model that passes all validation checks is registered and assigned the `champion` alias in the MLflow Model Registry.

## Quality Gates
To ensure model robustness and performance, the pipeline enforces three quality gates:
- **Macro F1 Score:** Must be `>= 0.88` to guarantee high classification accuracy across all classes.
- **Batch Latency:** Must be `<= 30 ms` to meet performance requirements for inference speed.
- **Class Indices:** Model predictions must strictly only contain indices `0, 1, or 2`.