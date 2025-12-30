import os
import yaml
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("security-log-detection")


def load_config():
    with open("training/config.yaml") as f:
        return yaml.safe_load(f)


def load_features(path):
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(path, file)))
    if not dfs:
        raise ValueError("No parquet feature files found")
    return pd.concat(dfs, ignore_index=True)


def main():
    config = load_config()
    df = load_features(config["data"]["features_path"])

    df["label"] = (df["failures_last_5min"] > 3).astype(int)

    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    with mlflow.start_run() as run:
    run_id = run.info.run_id

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

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model_type", config["model"]["type"])
    mlflow.log_param("max_iter", config["model"]["max_iter"])

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # ✅ THIS IS THE CRITICAL FIX
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/run_id.txt", "w") as f:
        f.write(run_id)

    print(f"Training complete. Run ID: {run_id}")



if __name__ == "__main__":
    main()
