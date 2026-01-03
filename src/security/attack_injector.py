import random
import pandas as pd
from typing import Tuple

from src.security.attack_templates import sample_attack


def inject_attacks(
    df: pd.DataFrame,
    injection_rate: float = 0.01,
    seed: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Injects synthetic attacks into raw log dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw log dataframe
    injection_rate : float
        Fraction of rows to inject attacks into
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Modified dataframe + number of injected attacks
    """

    random.seed(seed)

    df = df.copy()
    n_rows = len(df)
    n_inject = max(1, int(n_rows * injection_rate))

    inject_indices = random.sample(range(n_rows), n_inject)

    for idx in inject_indices:
        attack = sample_attack()

        df.at[idx, "method"] = attack["method"]
        df.at[idx, "path"] = attack["path"]
        df.at[idx, "request"] = attack["request"]

        # Internal audit tag (NOT used for training)
        df.at[idx, "_injected_attack"] = attack["attack_type"]

    return df, n_inject
