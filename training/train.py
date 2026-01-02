import os
import yaml
import mlflow
import pandas as pd
# import dagshub  <-- COMMENTED OUT TO PREVENT CI FREEZE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

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
    # MODULE 3.1 — SECURITY-GRADE LABELING (NO LEAKAGE)
    # ======================================================
    df["label"] = (
        (df["failures_last_5min"] > 3) |
        (df["error_rate"] > 0.25) |
        (df["latency_p95"] > 800)
    ).astype(int)

    # ------------------------------------------------------
    # LEAKAGE-PROOF FEATURE SELECTION
    # ------------------------------------------------------
    LEAKAGE_COLUMNS = {
        "label",
        "failures_last_5min",
        "error_rate",
        "latency_p95",
    }

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLUMNS]

    assert len(feature_cols) > 0, "❌ No valid features left after leakage removal"

    X = df[feature_cols]
    y = df["label"]

    print("✅ Training features:", feature_cols)
    print("⚠️ Positive class ratio:", round(y.mean(), 4))

    # ------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------------
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

        preds = model.predict(X_test)

        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        # ----------------------------
        # LOGGING
        # ----------------------------
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("max_iter", config["model"]["max_iter"])
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        print(f"📊 Metrics — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # ----------------------------
        # LOG + REGISTER MODEL
        # ----------------------------
        print("📦 Logging & registering model to DagsHub...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="security-log-model"
        )

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)

        print(f"✅ Training complete. Run ID saved to artifacts/run_id.txt")

if __name__ == "__main__":
    main()
