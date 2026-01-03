import json
from pathlib import Path
from collections import Counter


FEEDBACK_STORE = Path("artifacts/analyst_feedback.jsonl")


def compute_feedback_stats():
    """
    Computes analyst feedback statistics.
    """

    if not FEEDBACK_STORE.exists():
        return {}

    verdicts = []

    with FEEDBACK_STORE.open() as f:
        for line in f:
            verdicts.append(json.loads(line)["verdict"])

    counts = Counter(verdicts)

    total = sum(counts.values())

    return {
        "total_feedback": total,
        "true_positive_rate": counts.get("true_positive", 0) / total if total else 0,
        "false_positive_rate": counts.get("false_positive", 0) / total if total else 0,
        "needs_review_rate": counts.get("needs_review", 0) / total if total else 0,
    }
