import os
import yaml
import mlflow
import dagshub  # ✅ NEW IMPORT
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# ✅ 1. Initialize DagsHub (Configures MLflow + Artifact Storage automatically)
dagshub.init(repo_owner='reethj-07', repo_name='autonomous-security-mlops', mlflow=True)

def load_config():
    with open("training/config.yaml") as f:
        return yaml.safe_load(f)

def load_features(path):
    dfs = []
    # Check if path exists
    if not os.path.exists(path):
         raise ValueError(f"Path does not exist: {path}")
         
    for file in os.listdir(path):
        if file.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(path, file)))
    if not dfs:
        raise ValueError(f"No parquet feature files found in {path}")
    return pd.concat(dfs, ignore_index=True)

def main():
    config = load_config()
    print("Loading features...")
    df = load_features(config["data"]["features_path"])

    # Create Label
    df["label"] = (df["failures_last_5min"] > 3).astype(int)

    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

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

        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("max_iter", config["model"]["max_iter"])

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        print(f"Logging model to DagsHub... (F1: {f1:.4f})")
        
        # ✅ Log the model (Boto3 + DagsHub will handle the upload now)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="security-log-model" # Optional: registers immediately
        )

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Training complete. Run ID: {run_id} saved to artifacts/run_id.txt")

if __name__ == "__main__":
    main()