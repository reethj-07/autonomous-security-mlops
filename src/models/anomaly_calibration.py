import numpy as np
import pandas as pd
from typing import Dict


def percentile_threshold(
    scores: np.ndarray,
    percentile: float = 99.5
) -> float:
    """
    Computes anomaly threshold using score percentile.
    """
    return float(np.percentile(scores, percentile))


def evaluate_anomalies(
    scores: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluates anomaly score distribution relative to threshold.
    """
    flags = scores >= threshold

    return {
        "threshold": round(float(threshold), 6),
        "anomaly_rate": round(float(flags.mean()), 6),
        "mean_score": round(float(scores.mean()), 6),
        "p95_score": round(float(np.percentile(scores, 95)), 6),
        "p99_score": round(float(np.percentile(scores, 99)), 6),
        "max_score": round(float(scores.max()), 6),
    }
