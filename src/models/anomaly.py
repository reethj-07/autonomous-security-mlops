import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow


class SecurityIsolationForest:
    """
    Isolation Forest for security anomaly detection.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores.
        Higher = more anomalous
        """
        # IsolationForest: lower = more anomalous → invert
        raw_scores = self.model.score_samples(X)
        return -raw_scores

    def predict(self, X: pd.DataFrame, threshold: float = None):
        """
        Returns binary anomaly flags
        """
        scores = self.score(X)

        if threshold is None:
            # Default: use model decision function
            preds = self.model.predict(X)
            return (preds == -1).astype(int)

        return (scores >= threshold).astype(int)

    def log_to_mlflow(self, X: pd.DataFrame, prefix: str = "anomaly"):
        scores = self.score(X)

        mlflow.log_metric(f"{prefix}_mean_score", float(np.mean(scores)))
        mlflow.log_metric(f"{prefix}_p95_score", float(np.percentile(scores, 95)))
        mlflow.log_metric(f"{prefix}_p99_score", float(np.percentile(scores, 99)))
