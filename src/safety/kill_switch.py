from pathlib import Path
import yaml


CONFIG_PATH = Path("configs/runtime_config.yaml")

ALLOWED_STATES = {"ENABLED", "SAFE_MODE", "LOCKDOWN"}


class SafetyException(Exception):
    pass


def load_runtime_config() -> dict:
    """
    Loads runtime safety configuration.
    Fails closed (SAFE_MODE) if config is invalid.
    """
    if not CONFIG_PATH.exists():
        return {"system_state": "SAFE_MODE"}

    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        state = config.get("system_state", "SAFE_MODE")

        if state not in ALLOWED_STATES:
            raise SafetyException(f"Invalid system_state: {state}")

        return config

    except Exception:
        # Fail closed
        return {"system_state": "SAFE_MODE"}


def is_ml_enabled() -> bool:
    """
    Returns True only if ML is fully enabled.
    """
    config = load_runtime_config()
    return config["system_state"] == "ENABLED"


def is_safe_mode() -> bool:
    """
    Returns True if system is in SAFE_MODE.
    """
    config = load_runtime_config()
    return config["system_state"] == "SAFE_MODE"


def is_lockdown() -> bool:
    """
    Returns True if system is in LOCKDOWN.
    """
    config = load_runtime_config()
    return config["system_state"] == "LOCKDOWN"
