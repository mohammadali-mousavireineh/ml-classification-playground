from pathlib import Path

from ml_playground.data import load_dataset, split_dataset


def test_sample_dataset_can_be_loaded():
    df = load_dataset(Path("data/sample_iot_windows.csv"))
    assert not df.empty
    assert "label" in df.columns


def test_dataset_can_be_split():
    df = load_dataset(Path("data/sample_iot_windows.csv"))
    X_train, X_test, y_train, y_test = split_dataset(df, target="label")
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
