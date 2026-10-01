
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_wine_data():
    dataset = load_wine()

    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    validate_data(X, y)

    return X, y


def validate_data(X, y):
    if X.shape[1] != 13:
        raise ValueError("Wine dataset must have exactly 13 features.")

    if X.isnull().any().any():
        raise ValueError("Feature data contains null values.")

    if y.isnull().any():
        raise ValueError("Target data contains null values.")


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
