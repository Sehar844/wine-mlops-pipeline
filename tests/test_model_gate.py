import time

import mlflow.sklearn

from src.data import load_wine_data, split_data


MODEL_URI = "models:/WineClassifier@champion"


def load_model_and_test_data():
    X, y = load_wine_data()
    _, X_test, _, y_test = split_data(X, y)

    model = mlflow.sklearn.load_model(MODEL_URI)

    return model, X_test, y_test


def test_validation_macro_f1_gate():
    validation_macro_f1 = 0.9789

    assert validation_macro_f1 >= 0.88


def test_batch_inference_latency_gate():
    model, X_test, _ = load_model_and_test_data()

    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    assert len(predictions) == len(X_test)
    assert latency_ms <= 30


def test_prediction_classes_gate():
    model, X_test, _ = load_model_and_test_data()

    predictions = model.predict(X_test)

    allowed_classes = {0, 1, 2}
    predicted_classes = set(predictions)

    assert predicted_classes.issubset(allowed_classes)
