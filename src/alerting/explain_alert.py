from typing import Dict, List, Tuple


RISK_WEIGHTS = {
    "classifier_prob": 0.4,
    "anomaly_score": 0.4,
    "context_risk": 0.2,
}


def compute_risk_contributions(
    risk_output: Dict[str, float]
) -> Dict[str, float]:
    """
    Computes weighted contribution of each signal
    """

    contributions = {}

    for signal, weight in RISK_WEIGHTS.items():
        value = risk_output.get(signal)
        if value is not None:
            contributions[signal] = round(value * weight, 4)

    return contributions


def generate_explanation(
    risk_output: Dict[str, float],
    top_k: int = 3
) -> Dict[str, object]:
    """
    Generates human-readable explanation for SOC analysts
    """

    contributions = compute_risk_contributions(risk_output)

    ranked = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    reasons = []
    for signal, score in ranked[:top_k]:
        reasons.append(
            f"{signal} contributed {score:.3f} to overall risk"
        )

    explanation = {
        "risk_score": risk_output["risk_score"],
        "top_contributors": ranked[:top_k],
        "reasoning": reasons,
    }

    return explanation
