import json
from pathlib import Path
from typing import Dict


EXPLANATION_DIR = Path("artifacts/explanations")


def write_explanation(explanation: Dict[str, object]) -> Path:
    """
    Writes LLM explanation artifact to disk.
    """

    EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)

    path = EXPLANATION_DIR / f"{explanation['incident_id']}.json"

    with open(path, "w") as f:
        json.dump(explanation, f, indent=2)

    print(f"🧠 Explanation written to {path}")
    return path
