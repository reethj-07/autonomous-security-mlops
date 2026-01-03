import json
import subprocess
from pathlib import Path


SIGNAL_PATH = Path("airflow/signals/retrain_signal.json")


def run_retraining():
    """
    Executes retraining only if retrain_signal.json exists
    """
    if not SIGNAL_PATH.exists():
        print("✅ No retraining signal found. Skipping retraining.")
        return

    print("🚨 Retraining signal detected. Starting retraining pipeline...")

    # Optional: read signal metadata
    with open(SIGNAL_PATH) as f:
        signal = json.load(f)
        print("📡 Signal details:", signal)

    # Run training pipeline
    result = subprocess.run(
        ["python", "training/train.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("❌ Retraining failed")

    # Cleanup signal after successful retraining
    SIGNAL_PATH.unlink()
    print("🧹 Retraining signal consumed and removed")
