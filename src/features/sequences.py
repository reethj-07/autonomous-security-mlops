import pandas as pd
import numpy as np

from src.features.rarity import (
    add_path_rarity,
    add_method_entropy,
    add_sql_keyword_rarity
)


def generate_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Security-grade feature engineering + heuristic labeling.
    """

    # -----------------------------
    # BASIC FEATURE ENGINEERING
    # -----------------------------
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

    # -----------------------------
    # RARITY & ENTROPY FEATURES (Module 3.3.2)
    # -----------------------------
    df = add_path_rarity(df)
    df = add_method_entropy(df)
    df = add_sql_keyword_rarity(df)

    # -----------------------------
    # HEURISTIC LABELING (TEMPORARY)
    # -----------------------------
    risk_score = (
        (df["request_length"] > 120).astype(int)
        + df["has_sql_keywords"]
        + df["is_admin_path"]
    )

    df["label"] = (risk_score >= 2).astype(int)

    return df
