from typing import TypedDict
from datetime import datetime


class AnalystFeedback(TypedDict):
    alert_id: str
    analyst_id: str
    verdict: str  # true_positive | false_positive | benign_expected | needs_review
    comment: str
    timestamp: str
