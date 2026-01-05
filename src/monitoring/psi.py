import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI)

    expected = training distribution
    actual   = current / new distribution
    """

    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (expected_perc - actual_perc)
        * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )

    return round(float(psi), 4)
