import json
from pathlib import Path
from typing import List, Dict


INCIDENT_DIR = Path("artifacts/incidents")
EXPLANATION_DIR = Path("artifacts/explanations")


def load_incidents() -> List[Dict]:
    """
    Loads all incident context artifacts.
    """
    incidents = []

    for file in INCIDENT_DIR.glob("*.json"):
        with open(file) as f:
            incidents.append(json.load(f))

    return incidents


def load_explanations() -> Dict[str, Dict]:
    """
    Loads all LLM explanation artifacts indexed by incident_id.
    """
    explanations = {}

    for file in EXPLANATION_DIR.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            explanations[data["incident_id"]] = data

    return explanations
