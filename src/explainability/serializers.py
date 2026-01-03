from typing import Dict
from src.explainability.context_schema import IncidentContext


def serialize_for_llm(context: IncidentContext) -> Dict[str, object]:
    """
    Converts IncidentContext into a strictly bounded,
    hallucination-safe payload for LLM consumption.
    """

    return {
        "incident_id": context.incident_id,
        "timestamp": context.timestamp,
        "environment": context.environment,

        "model": {
            "name": context.model_name,
            "version": context.model_version,
            "stage": context.model_stage,
        },

        "trigger": {
            "reason": context.trigger_reason,
            "retrain_triggered": context.retrain_triggered,
        },

        "evidence": {
            "feature_drift": context.feature_drift,
            "prediction_drift": context.prediction_drift,
            "anomaly_score_stats": context.anomaly_score_stats,
            "canary_metrics": context.canary_metrics,
        },

        "safety": {
            "safe_mode": context.safe_mode,
            "kill_switch_active": context.kill_switch_active,
        },

        "system_summary": context.summary,

        "instructions": (
            "You are a security ML explanation assistant. "
            "Explain the incident strictly using the provided data. "
            "Do NOT speculate. Do NOT invent causes. "
            "Do NOT suggest actions unless explicitly supported by the data."
        ),
    }
