import os
import yaml
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)

# ======================================================
# CI-FRIENDLY MLFLOW SETUP
# ======================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    MLFLOW_TRACKING_URI = "https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("security-log-detection")

# ======================================================
# UTILS
# ======================================================
def load_config():
    with open("training/config.yaml") as f:
        return yaml.safe_load(f)

def load_features(path):
    if not os.path.exists(path) and os.path.exists(f"../{path}"):
        path = f"../{path}"

    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    dfs = [
        pd.read_parquet(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".parquet")
    ]

    if not dfs:
        raise ValueError("No parquet feature files found")

    return pd.concat(dfs, ignore_index=True)

# ======================================================
# MODULE 3.2 — SECURITY-AWARE TRAINING
# ======================================================
def main():
    config = load_config()

    print("📦 Loading features...")
    df = load_features(config["data"]["features_path"])

    # --------------------------------------------------
    # LABELING (Leakage-safe)
    # --------------------------------------------------
    label_conditions = []

    if "failures_last_5min" in df.columns:
        label_conditions.append(df["failures_last_5min"] > 3)

    if "latency_p95" in df.columns:
        label_conditions.append(df["latency_p95"] > 800)

    if not label_conditions:
        raise ValueError("❌ No valid columns for label creation")

    df["label"] = pd.concat(label_conditions, axis=1).any(axis=1).astype(int)

    # --------------------------------------------------
    # FEATURE SELECTION (No leakage)
    # --------------------------------------------------
    LEAKAGE_COLUMNS = {
        "label",
        "failures_last_5min",
        "latency_p95",
        "error_rate",
    }

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLUMNS]

    assert feature_cols, "❌ No valid features after leakage removal"

    X = df[feature_cols]
    y = df["label"]

    print("✅ Training features:", feature_cols)
    print("⚠️ Positive class ratio:", round(y.mean(), 4))

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🆔 Run ID: {run_id}")

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=config["model"]["max_iter"],
                    class_weight=config["model"]["class_weight"]
                ))
            ]
        )

        model.fit(X_train, y_train)

        # --------------------------------------------------
        # THRESHOLD SEARCH (F2 OPTIMIZATION)
        # --------------------------------------------------
        probs = model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.01, 0.91, 0.01)

        best = {
            "threshold": None,
            "f2": -1
        }

        for t in thresholds:
            preds = (probs >= t).astype(int)

            f2 = fbeta_score(
                y_test,
                preds,
                beta=2,
                zero_division=0
            )

            if f2 > best["f2"]:
                best.update({
                    "threshold": t,
                    "f2": f2,
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                    "f1": f1_score(y_test, preds, zero_division=0),
                    "cm": confusion_matrix(y_test, preds)
                })

        # --------------------------------------------------
        # LOG RESULTS
        # --------------------------------------------------
        print("🎯 Best Threshold:", round(best["threshold"], 3))
        print("📊 Confusion Matrix:\n", best["cm"])
        print(
            f"📈 Metrics @ threshold={best['threshold']:.2f} | "
            f"P={best['precision']:.3f} "
            f"R={best['recall']:.3f} "
            f"F1={best['f1']:.3f} "
            f"F2={best['f2']:.3f}"
        )

        mlflow.log_param("best_threshold", best["threshold"])
        mlflow.log_metric("precision", best["precision"])
        mlflow.log_metric("recall", best["recall"])
        mlflow.log_metric("f1", best["f1"])
        mlflow.log_metric("f2", best["f2"])

        # --------------------------------------------------
        # LOG & REGISTER MODEL
        # --------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="security-log-model"
        )

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)

        print("✅ Training complete & model registered")

if __name__ == "__main__":
    main()
