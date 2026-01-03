import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from src.feedback.feedback_schema import AnalystFeedback


FEEDBACK_STORE = Path("artifacts/analyst_feedback.jsonl")


def store_feedback(feedback: AnalystFeedback) -> None:
    """
    Append analyst feedback to feedback store.
    """

    FEEDBACK_STORE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        **feedback,
        "timestamp": datetime.utcnow().isoformat()
    }

    with FEEDBACK_STORE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print("📝 Analyst feedback recorded:", record)
