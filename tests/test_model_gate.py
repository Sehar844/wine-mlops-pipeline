import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import load_wine_data, split_data


RANDOM_STATE = 42


def build_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE,
    )


def test_validation_macro_f1_gate():
    X, y = load_wine_data()
    X_train, _, y_train, _ = split_data(X, y)

    model = build_model()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict",
    )

    validation_macro_f1 = f1_score(
        y_train,
        predictions,
        average="macro",
    )

    assert validation_macro_f1 >= 0.88


def test_batch_inference_latency_gate():
    X, y = load_wine_data()
    X_train, X_test, y_train, _ = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    assert len(predictions) == len(X_test)
    assert latency_ms <= 30


def test_prediction_classes_gate():
    X, y = load_wine_data()
    X_train, X_test, y_train, _ = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    allowed_classes = {0, 1, 2}
    predicted_classes = set(predictions)

    assert predicted_classes.issubset(allowed_classes)
