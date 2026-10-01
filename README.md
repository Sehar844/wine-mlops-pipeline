# Wine MLOps Pipeline

An end-to-end MLOps pipeline for Wine Cultivar Classification using
scikit-learn, MLflow, pytest, Make, Git, and GitHub Actions.

## Project Overview

This project builds a reproducible machine learning pipeline using the
scikit-learn Wine dataset.

The pipeline includes:

- Data loading and validation
- Stratified train-test splitting
- Multiple Random Forest and Gradient Boosting configurations
- 5-fold Stratified Cross-Validation
- MLflow experiment tracking
- MLflow model logging and registry
- Model evaluation on a held-out test set
- Automated quality gates
- GitHub Actions CI
- Git branching and conflict-resolution workflow

All stochastic operations use `random_state=42` for reproducibility.

## Dataset

The project uses the Wine dataset from `sklearn.datasets.load_wine`.

Dataset characteristics:

- Samples: 178
- Features: 13
- Classes: 3
- Missing feature values: None
- Train/test split: 80/20
- Split strategy: Stratified
- Random state: 42

## Project Structure

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model_gate.py
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md

