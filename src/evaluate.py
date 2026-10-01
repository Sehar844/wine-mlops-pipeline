import time

import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_wine_data, split_data


MODEL_URI = "models:/WineClassifier@champion"


def main():
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = mlflow.sklearn.load_model(MODEL_URI)

    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    end_time = time.perf_counter()

    probabilities = model.predict_proba(X_test)

    latency_ms = (end_time - start_time) * 1000

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
    )

    print("\nFinal Test Results")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Log Loss: {test_log_loss:.4f}")
    print(f"Batch Inference Latency: {latency_ms:.4f} ms")
    print(f"Predicted Classes: {sorted(set(predictions))}")


if __name__ == "__main__":
    main()
