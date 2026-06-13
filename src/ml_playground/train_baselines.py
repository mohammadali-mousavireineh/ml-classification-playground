from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import Pipeline

from ml_playground.data import load_dataset, split_dataset
from ml_playground.evaluate import classification_metrics, save_confusion_matrix, save_metrics_table
from ml_playground.features import build_preprocessor
from ml_playground.models import get_models


def run_baseline_experiment(
    data_path: str | Path,
    target: str = "label",
    output_dir: str | Path = "reports",
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Train a few baseline models and save their results."""
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state,
    )

    output_dir = Path(output_dir)
    rows = []
    fitted_models = {}

    for model_name, model in get_models(random_state=random_state).items():
        scale_numeric = model_name == "logistic_regression"
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train, scale_numeric=scale_numeric)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = classification_metrics(y_test, predictions)
        rows.append({"model": model_name, **metrics})
        fitted_models[model_name] = pipeline

    results = save_metrics_table(rows, output_dir / "baseline_metrics.csv")
    best_model_name = results.iloc[0]["model"]
    save_confusion_matrix(
        fitted_models[best_model_name],
        X_test,
        y_test,
        output_dir / f"confusion_matrix_{best_model_name}.png",
    )

    return results
