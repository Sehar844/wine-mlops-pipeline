import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import load_wine_data, split_data


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
RANDOM_STATE = 42


def get_models():
    return {
        "rf_1": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "rf_2": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "rf_3": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
        ),
        "gb_1": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "gb_2": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "gb_3": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "accuracy": accuracy_score(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities),
    }


def train_model(name, model, X_train, y_train):
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

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )

    metrics = calculate_metrics(y_train, predictions, probabilities)

    model.fit(X_train, y_train)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", 5)

        for parameter, value in model.get_params().items():
            if value is not None:
                mlflow.log_param(parameter, value)

        mlflow.log_metric("validation_macro_f1", metrics["macro_f1"])
        mlflow.log_metric("validation_accuracy", metrics["accuracy"])
        mlflow.log_metric("validation_log_loss", metrics["log_loss"])

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(1),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    print(f"\n{name}")
    print(f"Validation Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Log Loss: {metrics['log_loss']:.4f}")

    return metrics


def main():
    X, y = load_wine_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    mlflow.set_experiment(EXPERIMENT_NAME)

    models = get_models()
    results = {}

    for name, model in models.items():
        results[name] = train_model(
            name,
            model,
            X_train,
            y_train,
        )

    best_model = max(
        results,
        key=lambda model_name: results[model_name]["macro_f1"],
    )

    print("\nBest Model:")
    print(best_model)
    print(
        f"Best Validation Macro F1: "
        f"{results[best_model]['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
