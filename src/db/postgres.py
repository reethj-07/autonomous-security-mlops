import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

DB_URL = "postgresql://user:password@localhost:5432/security_ml"

engine = create_engine(DB_URL)


def write_features_to_db(execution_time):
    feature_path = Path("data/features") / f"features_{execution_time.isoformat()}.parquet"

    if not feature_path.exists():
        raise FileNotFoundError("Feature file not found")

    df = pd.read_parquet(feature_path)

    df.to_sql(
        "security_features",
        engine,
        if_exists="append",
        index=False,
        method="multi",
    )
