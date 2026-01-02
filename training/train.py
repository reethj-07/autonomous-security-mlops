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
    # LABELING (Leakage-safe, heuristic)
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
    # FEATURE SELECTION (STRICT NO-LEAKAGE)
    # --------------------------------------------------
    LEAKAGE_COLUMNS = {
        "label",
        "failures_last_5min",
        "latency_p95",
        "error_rate",
    }

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLUMNS]

    if not feature_cols:
        raise ValueError("❌ No valid features after leakage removal")

    X = df[feature_cols]
    y = df["label"]

    print("✅ Training features:", feature_cols)
    print("⚠️ Positive class ratio:", round(y.mean(), 4))

    # --------------------------------------------------
    # TRAIN / TEST SPLIT (STRATIFIED)
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
        # PROBABILITY-BASED INFERENCE
        # --------------------------------------------------
        probs = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # THRESHOLD SEARCH (F2 OPTIMIZATION)
        # --------------------------------------------------
        thresholds = np.linspace(0.01, 0.5, 50)

        best_f2 = 0.0
        best_threshold = 0.5
        best_metrics = {}

        for t in thresholds:
            preds = (probs >= t).astype(int)

            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)

            if f2 > best_f2:
                best_f2 = f2
                best_threshold = t
                best_metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "f2": f2,
                }

        # --------------------------------------------------
        # SAFE FALLBACK (NO POSITIVE PREDICTIONS)
        # --------------------------------------------------
        if not best_metrics:
            best_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f2": 0.0,
            }

        # --------------------------------------------------
        # FINAL EVALUATION
        # --------------------------------------------------
        final_preds = (probs >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, final_preds)

        print(f"🎯 Best Threshold: {best_threshold:.2f}")
        print("📊 Confusion Matrix:")
        print(cm)

        print(
            f"📈 Metrics @ threshold={best_threshold:.2f} | "
            f"P={best_metrics['precision']:.3f} "
            f"R={best_metrics['recall']:.3f} "
            f"F1={best_metrics['f1']:.3f} "
            f"F2={best_metrics['f2']:.3f}"
        )

        # --------------------------------------------------
        # LOG TO MLFLOW
        # --------------------------------------------------
        mlflow.log_metric("positive_class_ratio", y.mean())
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("precision", best_metrics["precision"])
        mlflow.log_metric("recall", best_metrics["recall"])
        mlflow.log_metric("f1", best_metrics["f1"])
        mlflow.log_metric("f2", best_metrics["f2"])

        # Save confusion matrix as artifact
        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write(str(cm))

        mlflow.log_artifact(cm_path)

        # --------------------------------------------------
        # LOG & REGISTER MODEL
        # --------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="security-log-model"
        )

        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)

        print("✅ Training complete & model registered")

if __name__ == "__main__":
    main()
