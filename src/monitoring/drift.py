from typing import Dict, Optional
from src.monitoring.drift_metrics import is_drift_severe


def should_retrain(
    feature_drift: Dict[str, float],
    prediction_drift_score: Optional[float] = None,
    psi_threshold: float = 0.25,
    prediction_drift_threshold: float = 0.15,
) -> Dict[str, object]:
    """
    Central retraining decision engine.

    Parameters
    ----------
    feature_drift : dict
        Feature-wise PSI or drift scores
    prediction_drift_score : float, optional
        Aggregate prediction drift score
    psi_threshold : float
        Threshold above which feature drift is considered severe
    prediction_drift_threshold : float
        Threshold above which prediction drift is severe

    Returns
    -------
    dict
        Structured retraining decision (audit-friendly)
    """

    # -------------------------------
    # Feature drift decision
    # -------------------------------
    feature_drift_flag = is_drift_severe(
        feature_drift,
        threshold=psi_threshold
    )

    # -------------------------------
    # Prediction drift decision
    # -------------------------------
    prediction_drift_flag = False
    if prediction_drift_score is not None:
        prediction_drift_flag = (
            prediction_drift_score > prediction_drift_threshold
        )

    # -------------------------------
    # Final verdict
    # -------------------------------
    retrain = feature_drift_flag or prediction_drift_flag

    return {
        "retrain": retrain,
        "feature_drift": feature_drift_flag,
        "prediction_drift": prediction_drift_flag,
        "psi_threshold": psi_threshold,
        "prediction_drift_threshold": prediction_drift_threshold,
    }
