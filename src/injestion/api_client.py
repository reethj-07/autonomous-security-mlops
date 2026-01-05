import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from configs import settings

RAW_DATA_DIR = Path("data/raw")


def fetch_security_logs(execution_time: datetime) -> List[dict]:
    """
    Fetch logs from external security APIs for a fixed time window.
    This function is deterministic and idempotent.
    """
    start_time = execution_time - timedelta(minutes=5)
    end_time = execution_time

    all_logs = []

    for source in settings.LOG_SOURCES:
        response = requests.get(
            source["url"],
            params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
            timeout=5,
        )
        response.raise_for_status()
        all_logs.extend(response.json())

    return all_logs


def persist_raw_logs(execution_time: datetime):
    """
    Persist raw logs to immutable, time-partitioned storage.
    """
    logs = fetch_security_logs(execution_time)

    date_path = RAW_DATA_DIR / execution_time.strftime("%Y-%m-%d")
    date_path.mkdir(parents=True, exist_ok=True)

    file_path = date_path / f"logs_{execution_time.isoformat()}.json"

    with open(file_path, "w") as f:
        json.dump(logs, f)

    return file_path
