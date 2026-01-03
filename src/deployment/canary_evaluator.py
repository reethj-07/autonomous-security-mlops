from datetime import datetime
from typing import Dict, List


def evaluate_canary(
    canary_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, object]:
    """
    Evaluates canary model safety against baseline.

    Returns one of:
    - PROMOTE
    - ROLLBACK
    - EXTEND
    """

    reasons: List[str] = []

    # ----------------------------
    # Guardrail thresholds
    # ----------------------------
    ALERT_MULTIPLIER_LIMIT = 2.0
    ENTROPY_DELTA_LIMIT = 0.15
    LATENCY_MULTIPLIER_LIMIT = 1.1

    # ----------------------------
    # Required metrics
    # ----------------------------
    required_keys = [
        "alerts_per_min",
        "prediction_entropy",
        "p95_latency_ms",
    ]

    for key in required_keys:
        if key not in canary_metrics or key not in baseline_metrics:
            return {
                "decision": "EXTEND",
                "reasons": [f"Missing metric: {key}"],
                "timestamp": datetime.utcnow().isoformat(),
            }

    # ----------------------------
    # Alert rate guard
    # ----------------------------
    if (
        canary_metrics["alerts_per_min"]
        > baseline_metrics["alerts_per_min"] * ALERT_MULTIPLIER_LIMIT
    ):
        reasons.append("Alert rate spike detected")

    # ----------------------------
    # Entropy guard
    # ----------------------------
    if (
        canary_metrics["prediction_entropy"]
        > baseline_metrics["prediction_entropy"] + ENTROPY_DELTA_LIMIT
    ):
        reasons.append("Prediction entropy increased")

    # ----------------------------
    # Latency guard
    # ----------------------------
    if (
        canary_metrics["p95_latency_ms"]
        > baseline_metrics["p95_latency_ms"] * LATENCY_MULTIPLIER_LIMIT
    ):
        reasons.append("Latency regression detected")

    # ----------------------------
    # Final decision
    # ----------------------------
    if reasons:
        return {
            "decision": "ROLLBACK",
            "reasons": reasons,
            "timestamp": datetime.utcnow().isoformat(),
        }

    return {
        "decision": "PROMOTE",
        "reasons": ["Canary stable"],
        "timestamp": datetime.utcnow().isoformat(),
    }
