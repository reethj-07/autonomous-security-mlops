import numpy as np
from typing import Dict


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize anomaly scores to [0, 1]
    """
    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s < 1e-6:
        return np.zeros_like(scores)

    return (scores - min_s) / (max_s - min_s)


def compute_context_risk(features: Dict[str, float]) -> float:
    """
    Simple rule-based context risk
    """

    risk = 0.0

    if features.get("is_admin_path", 0) == 1:
        risk += 0.4

    if features.get("has_sql_keywords", 0) == 1:
        risk += 0.4

    if features.get("request_length", 0) > 200:
        risk += 0.2

    return min(risk, 1.0)


def compute_hybrid_risk(
    classifier_prob: float,
    anomaly_score: float,
    anomaly_score_norm: float,
    context_features: Dict[str, float],
    weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Computes final hybrid risk score
    """

    if weights is None:
        weights = {
            "classifier": 0.5,
            "anomaly": 0.4,
            "context": 0.1,
        }

    context_risk = compute_context_risk(context_features)

    risk_score = (
        weights["classifier"] * classifier_prob
        + weights["anomaly"] * anomaly_score_norm
        + weights["context"] * context_risk
    )

    return {
        "risk_score": round(float(risk_score), 4),
        "classifier_prob": round(float(classifier_prob), 4),
        "anomaly_score": round(float(anomaly_score), 4),
        "context_risk": round(float(context_risk), 4),
    }
