from typing import List, Dict
from datetime import datetime


def build_timeline(
    incidents: List[Dict],
    explanations: Dict[str, Dict]
) -> List[Dict]:
    """
    Merges incidents and explanations into a time-ordered timeline.
    """

    timeline = []

    for incident in incidents:
        entry = {
            "incident_id": incident["incident_id"],
            "timestamp": incident["timestamp"],
            "environment": incident["environment"],
            "model": {
                "name": incident["model_name"],
                "version": incident["model_version"],
                "stage": incident["model_stage"],
            },
            "trigger_reason": incident["trigger_reason"],
            "retrain_triggered": incident["retrain_triggered"],
            "summary": incident["summary"],
            "explanation": explanations.get(
                incident["incident_id"], {}
            ).get("explanation", "No explanation available"),
        }

        timeline.append(entry)

    timeline.sort(
        key=lambda x: datetime.fromisoformat(x["timestamp"]),
        reverse=True,
    )

    return timeline
