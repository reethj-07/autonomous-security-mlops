from typing import Dict

from src.explainability.serializers import serialize_for_llm
from src.explainability.context_schema import IncidentContext
from src.explainability.llm_client import LLMClient


def generate_explanation(context: IncidentContext) -> Dict[str, object]:
    """
    Generates a grounded LLM explanation for an incident.
    """

    payload = serialize_for_llm(context)

    llm = LLMClient()
    explanation_text = llm.generate(payload)

    return {
        "incident_id": context.incident_id,
        "timestamp": context.timestamp,
        "model": payload["model"],
        "environment": payload["environment"],
        "explanation": explanation_text,
    }
