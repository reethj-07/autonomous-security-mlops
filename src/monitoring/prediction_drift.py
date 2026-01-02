import numpy as np


def prediction_confidence_drift(probs: np.ndarray):
    """
    Detects degradation in model confidence.
    """

    mean_confidence = np.mean(probs)
    entropy = -np.mean(
        probs * np.log(probs + 1e-6) + (1 - probs) * np.log(1 - probs + 1e-6)
    )

    return {
        "mean_confidence": round(float(mean_confidence), 4),
        "prediction_entropy": round(float(entropy), 4),
    }
