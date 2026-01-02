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

# ----------------------------
# CI-FRIENDLY MLFLOW SETUP
# ----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    MLFLOW_TRACKING_URI = "https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("security-log-detection")

# ----------------------------
# UTILS
# ----------------------------
def load_config():
    with open("training/config.yaml") as f:
        return yaml.safe_load(f)

def load_features(path):
    dfs = []

    if not os.path.exists(path):
        if os.path.exists(f"../{path}"):
            path = f"../{path}"
        else:
            raise ValueError(f"Path does not exist: {path}")

    for file in os.listdir(path):
        if file.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(path, file)))

    if not dfs:
        raise ValueError(f"No parquet feature files found in {path}")

    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# TRAINING ENTRYPOINT
# ----------------------------
def main():
    config = load_config()
    print("📦 Loading features...")
    df = load_features(config["data"]["features_path"])

    # ======================================================
    # MODULE 3.1 — SECURITY-GRADE LABELING (SCHEMA-AWARE)
    # ======================================================
    label_conditions = []

    if "failures_last_5min" in df.columns:
        label_conditions.append(df["failures_last_5min"] > 3)

    if "latency_p95" in df.columns:
        label_conditions.append(df["latency_p95"] > 800)

    if not label_conditions:
        raise ValueError(
            "❌ No valid columns available to create labels. "
            "Check feature engineering pipeline."
        )

    df["label"] = pd.concat(label_conditions, axis=1).any(axis=1).astype(int)

    # ======================================================
    # LEAKAGE-PROOF FEATURE SELECTION
    # ======================================================
    LEAKAGE_COLUMNS = {
        "label",
        "failures_last_5min",
        "latency_p95",
        "error_rate",   # future-proof
    }

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLUMNS]

    if not feature_cols:
        raise ValueError("❌ No valid features left after leakage removal")

    X = df[feature_cols]
    y = df["label"]

    print("✅ Training features:", feature_cols)
    print("⚠️ Positive class ratio:", round(y.mean(), 4))

    # ======================================================
    # TRAIN / TEST SPLIT (STRATIFIED)
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )

    print("🚀 Starting MLflow run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🆔 Run ID: {run_id}")

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=config["model"]["max_iter"],
                    class_weight=config["model"]["class_weight"]
                ))
            ]
        )

        model.fit(X_train, y_train)

        # ======================================================
        # MODULE 3.2 — F2 OPTIMIZATION & THRESHOLDING
        # ======================================================
        probs = model.predict_proba(X_test)[:, 1]

        thresholds = np.linspace(0.01, 0.99, 50)
        best_f2 = 0.0
        best_threshold = 0.5

        for t in thresholds:
            preds_t = (probs >= t).astype(int)
            f2_t = fbeta_score(y_test, preds_t, beta=2, zero_division=0)
            if f2_t > best_f2:
                best_f2 = f2_t
                best_threshold = t

        final_preds = (probs >= best_threshold).astype(int)

        precision = precision_score(y_test, final_preds, zero_division=0)
        recall = recall_score(y_test, final_preds, zero_division=0)
        f1 = f1_score(y_test, final_preds, zero_division=0)
        f2 = fbeta_score(y_test, final_preds, beta=2, zero_division=0)

        cm = confusion_matrix(y_test, final_preds)

        print("📊 Confusion Matrix:")
        print(cm)

        print(
            f"📈 Metrics @ threshold={best_threshold:.2f} | "
            f"Precision={precision:.3f} Recall={recall:.3f} "
            f"F1={f1:.3f} F2={f2:.3f}"
        )

        # ======================================================
        # LOGGING
        # ======================================================
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("max_iter", config["model"]["max_iter"])
        mlflow.log_param("optimal_threshold", best_threshold)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("f2", f2)

        mlflow.log_metric("tn", cm[0, 0])
        mlflow.log_metric("fp", cm[0, 1])
        mlflow.log_metric("fn", cm[1, 0])
        mlflow.log_metric("tp", cm[1, 1])

        # ======================================================
        # LOG + REGISTER MODEL
        # ======================================================
        print("📦 Logging & registering model to DagsHub...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="security-log-model"
        )

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)

        print("✅ Training complete. Run ID saved to artifacts/run_id.txt")

if __name__ == "__main__":
    main()
