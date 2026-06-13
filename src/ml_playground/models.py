from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(random_state: int = 42):
    """Return a small set of baseline classification models."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            random_state=random_state,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }
