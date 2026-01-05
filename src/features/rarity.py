import numpy as np
import pandas as pd

# --------------------------------------------------
# PATH RARITY
# --------------------------------------------------
def add_path_rarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes inverse-frequency rarity score for request paths
    """
    path_counts = df["path"].value_counts()
    path_prob = path_counts / path_counts.sum()

    df["path_rarity_score"] = df["path"].map(
        lambda p: -np.log(path_prob.get(p, 1e-6))
    )

    return df


# --------------------------------------------------
# METHOD ENTROPY (PER IP)
# --------------------------------------------------
def add_method_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures how random HTTP method usage is per IP
    """
    entropy_scores = {}

    for ip, group in df.groupby("ip_address"):
        probs = group["method"].value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        entropy_scores[ip] = entropy

    df["method_entropy"] = df["ip_address"].map(entropy_scores)
    return df


# --------------------------------------------------
# SQL KEYWORD RARITY
# --------------------------------------------------
def add_sql_keyword_rarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Penalizes rare presence of SQL keywords
    """
    if "has_sql_keywords" not in df.columns:
        df["sql_keyword_rarity"] = 0.0
        return df

    prob_sql = df["has_sql_keywords"].mean()
    rarity = -np.log(prob_sql + 1e-6)

    df["sql_keyword_rarity"] = df["has_sql_keywords"] * rarity
    return df
