from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate the main metrics used in early classification experiments."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_metrics_table(rows: list[dict], output_path: str | Path) -> pd.DataFrame:
    """Save model comparison metrics as a CSV table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    results.to_csv(output_path, index=False)
    return results


def save_confusion_matrix(model, X_test, y_test, output_path: str | Path) -> None:
    """Save a confusion matrix figure for the selected model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    display.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
