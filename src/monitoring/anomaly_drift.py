import numpy as np
from typing import Dict


def detect_anomaly_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    percentile: float = 99.5,
    rate_multiplier: float = 2.0,
    score_delta_threshold: float = 0.05,
) -> Dict[str, object]:
    """
    Detects anomaly-driven retraining signals.

    Parameters
    ----------
    baseline_scores : np.ndarray
        Anomaly scores from reference / training period
    current_scores : np.ndarray
        Anomaly scores from live / recent data
    percentile : float
        Percentile used for thresholding
    rate_multiplier : float
        Allowed increase in anomaly rate
    score_delta_threshold : float
        Minimum increase in pXX score to trigger drift

    Returns
    -------
    dict
        Structured anomaly drift decision
    """

    # Percentile thresholds
    baseline_thr = np.percentile(baseline_scores, percentile)
    current_thr = np.percentile(current_scores, percentile)

    # Anomaly rates
    baseline_rate = (baseline_scores >= baseline_thr).mean()
    current_rate = (current_scores >= current_thr).mean()

    # Drift flags
    rate_drift = current_rate > baseline_rate * rate_multiplier
    score_drift = (current_thr - baseline_thr) > score_delta_threshold

    retrain = rate_drift or score_drift

    return {
        "retrain": retrain,
        "baseline_threshold": round(float(baseline_thr), 6),
        "current_threshold": round(float(current_thr), 6),
        "baseline_anomaly_rate": round(float(baseline_rate), 6),
        "current_anomaly_rate": round(float(current_rate), 6),
        "rate_drift": rate_drift,
        "score_drift": score_drift,
        "percentile": percentile,
    }
