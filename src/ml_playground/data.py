from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and do a few basic checks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target with stratification when possible."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
