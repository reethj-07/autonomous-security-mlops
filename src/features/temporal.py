import pandas as pd

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")

    # Ensure timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -----------------------------
    # IP-based burst features
    # -----------------------------
    df["ip_req_count_1m"] = (
        df.groupby("ip_address")["timestamp"]
        .rolling("1min")
        .count()
        .reset_index(level=0, drop=True)
    )

    df["ip_req_count_5m"] = (
        df.groupby("ip_address")["timestamp"]
        .rolling("5min")
        .count()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # User login failure bursts
    # -----------------------------
    df["user_login_fail_5m"] = (
        df[df["is_login_failure"] == 1]
        .groupby("user_id")["timestamp"]
        .rolling("5min")
        .count()
        .reset_index(level=0, drop=True)
    )

    df["user_login_fail_5m"] = df["user_login_fail_5m"].fillna(0)

    # -----------------------------
    # Privilege change bursts
    # -----------------------------
    df["session_priv_change_10m"] = (
        df[df["is_privilege_change"] == 1]
        .groupby("session_id")["timestamp"]
        .rolling("10min")
        .count()
        .reset_index(level=0, drop=True)
    )

    df["session_priv_change_10m"] = df["session_priv_change_10m"].fillna(0)

    return df
