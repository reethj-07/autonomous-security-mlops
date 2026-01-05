import pandas as pd
from typing import Dict
from src.monitoring.psi import calculate_psi


DRIFT_FEATURES = [
    "request_length",
    "path_rarity_score",
    "method_entropy",
    "sql_keyword_rarity",
]


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Computes PSI for critical security features
    """

    drift_report = {}

    for feature in DRIFT_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        psi = calculate_psi(
            reference_df[feature],
            current_df[feature]
        )

        drift_report[feature] = psi

    return drift_report


def is_drift_severe(drift_report: Dict[str, float], threshold: float = 0.25) -> bool:
    """
    Returns True if ANY feature exceeds drift threshold
    """

    return any(psi > threshold for psi in drift_report.values())
