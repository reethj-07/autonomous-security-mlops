import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_security_logs():
    """Generate sample security logs for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="1min"),
        "user_id": np.random.randint(1, 20, 100),
        "path": np.random.choice(["/admin", "/api/users", "/login", "/dashboard"], 100),
        "method": np.random.choice(["GET", "POST", "PUT", "DELETE"], 100),
        "request": [f"request_{i}" for i in range(100)],
        "status_code": np.random.choice([200, 400, 401, 403, 500], 100),
        "latency_ms": np.random.uniform(10, 1000, 100),
        "ip_address": [f"192.168.1.{i % 255}" for i in range(100)],
    })


@pytest.fixture
def sample_features():
    """Generate sample engineered features."""
    np.random.seed(42)
    return pd.DataFrame({
        "request_length": np.random.uniform(10, 500, 100),
        "has_sql_keywords": np.random.randint(0, 2, 100),
        "is_admin_path": np.random.randint(0, 2, 100),
        "failures_last_5min": np.random.randint(0, 10, 100),
        "latency_p95": np.random.uniform(100, 1000, 100),
        "path_rarity": np.random.uniform(0.01, 1.0, 100),
        "method_entropy": np.random.uniform(0, 2, 100),
        "label": np.random.randint(0, 2, 100),
    })


@pytest.fixture
def sample_predictions():
    """Generate sample model predictions."""
    np.random.seed(42)
    return pd.DataFrame({
        "prediction": np.random.randint(0, 2, 50),
        "probability": np.random.uniform(0, 1, 50),
        "anomaly_score": np.random.uniform(0, 1, 50),
        "risk_score": np.random.uniform(0, 1, 50),
    })


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    """Create temporary artifacts directory."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir
