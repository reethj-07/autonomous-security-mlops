from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal
from datetime import datetime
import json
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


class SecurityLog(BaseModel):
    event_id: str
    timestamp: datetime
    source: Literal["auth", "api", "app"]
    user_id: Optional[str]
    ip_address: str
    geo: Optional[str]
    event_type: str
    raw_message: str

def validate_logs(execution_time: datetime):
    date_path = RAW_DATA_DIR / execution_time.strftime("%Y-%m-%d")
    file_path = date_path / f"logs_{execution_time.isoformat()}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"No raw log file found for {execution_time}")

    with open(file_path) as f:
        raw_logs = json.load(f)

    valid_logs = []
    invalid_logs = []

    for record in raw_logs:
        try:
            validated = SecurityLog(**record)
            valid_logs.append(validated.dict())
        except ValidationError as e:
            invalid_logs.append({"record": record, "error": str(e)})

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    valid_path = PROCESSED_DATA_DIR / f"validated_{execution_time.isoformat()}.json"
    invalid_path = PROCESSED_DATA_DIR / f"invalid_{execution_time.isoformat()}.json"

    with open(valid_path, "w") as f:
        json.dump(valid_logs, f)

    with open(invalid_path, "w") as f:
        json.dump(invalid_logs, f)

    return valid_path
