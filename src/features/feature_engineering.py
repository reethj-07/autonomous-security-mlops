import pandas as pd
import numpy as np
from src.security.attack_injector import inject_attacks


from src.features.rarity import (
    add_path_rarity,
    add_method_entropy,
    add_sql_keyword_rarity
)

from src.features.sequences import (
    add_path_transition_risk,
    add_repeated_request_count,
    add_method_transition_flag,
)


def generate_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Security-grade feature engineering with controlled attack injection.
    """

    # ----------------------------------
    # Inject synthetic attacks BEFORE features
    # ----------------------------------
    df, injected_count = inject_attacks(
        df,
        injection_rate=0.02  # 2% controlled injection
    )

    print(f"🚨 Injected {injected_count} synthetic attacks")

    # ----------------------------------
    # Feature engineering
    # ----------------------------------
    df["request_length"] = df["request"].str.len()

    df["has_sql_keywords"] = df["request"].str.contains(
        r"(select|union|drop|insert|or 1=1)",
        case=False,
        regex=True
    ).astype(int)

    df["is_admin_path"] = df["path"].str.contains(
        r"(admin|login|wp-login)",
        case=False,
        regex=True
    ).astype(int)

    # ----------------------------------
    # Heuristic risk score (NO leakage)
    # ----------------------------------
    risk_score = (
        (df["request_length"] > 120).astype(int)
        + df["has_sql_keywords"]
        + df["is_admin_path"]
    )

    # ----------------------------------
    # Final label (weak supervision)
    # ----------------------------------
    df["label"] = (risk_score >= 2).astype(int)

    return df
