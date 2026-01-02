import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI)
    """

    def _scale(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-6)

    expected = _scale(expected)
    actual = _scale(actual)

    breakpoints = np.linspace(0, 1, bins + 1)

    expected_bins = pd.cut(expected, breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, breakpoints, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True).sort_index()
    actual_dist = actual_bins.value_counts(normalize=True).sort_index()

    psi = np.sum(
        (actual_dist - expected_dist)
        * np.log((actual_dist + 1e-6) / (expected_dist + 1e-6))
    )

    return float(psi)


def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list,
    psi_threshold: float = 0.2,
):
    """
    Computes PSI per feature and flags drift.
    """

    report = {}

    for feature in feature_columns:
        psi = calculate_psi(reference_df[feature], current_df[feature])

        report[feature] = {
            "psi": round(psi, 4),
            "drift_detected": psi > psi_threshold,
        }

    return report
