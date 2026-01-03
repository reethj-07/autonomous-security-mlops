from typing import Dict


ALERT_THRESHOLDS = {
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 0.8,
}


def determine_alert_level(risk_score: float) -> str:
    """
    Maps a risk score to SOC alert level
    """
    if risk_score >= ALERT_THRESHOLDS["HIGH"]:
        return "CRITICAL"
    elif risk_score >= ALERT_THRESHOLDS["MEDIUM"]:
        return "HIGH"
    elif risk_score >= ALERT_THRESHOLDS["LOW"]:
        return "MEDIUM"
    else:
        return "LOW"


def generate_alert(
    risk_output: Dict[str, float],
    metadata: Dict[str, object] = None,
) -> Dict[str, object]:
    """
    Creates a SOC-friendly alert payload
    """

    if metadata is None:
        metadata = {}

    risk_score = risk_output["risk_score"]
    level = determine_alert_level(risk_score)

    alert = {
        "alert_level": level,
        "risk_score": risk_score,
        "signals": {
            "classifier_prob": risk_output.get("classifier_prob"),
            "anomaly_score": risk_output.get("anomaly_score"),
            "context_risk": risk_output.get("context_risk"),
        },
        "metadata": metadata,
    }

    return alert
