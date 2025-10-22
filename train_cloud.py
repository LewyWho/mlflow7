from pathlib import Path
import os, shutil

# --- 0) Configure MLflow URIs before importing mlflow ---
project_dir = Path(__file__).resolve().parent
tracking_dir = project_dir / "mlruns"
tracking_dir.mkdir(parents=True, exist_ok=True)

os.environ["MLFLOW_TRACKING_URI"] = tracking_dir.as_uri()
registry_db = project_dir / "mlflow_registry.db"
os.environ["MLFLOW_REGISTRY_URI"] = f"sqlite:///{registry_db.as_posix()}"

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_registry_uri(os.environ["MLFLOW_REGISTRY_URI"])
    mlflow.set_experiment("practice07_wine")

    # --- Data ---
    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model ---
    params = {"n_estimators": 200, "max_depth": 6, "random_state": 42}
    model = RandomForestClassifier(**params).fit(X_train, y_train)

    # --- Metrics ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:2],
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        local_model_dir = Path(download_artifacts(model_uri))
        dest_dir = project_dir / "model"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(local_model_dir, dest_dir)

        print(f"[OK] accuracy={acc:.4f} f1_weighted={f1:.4f}")
        print(f"[OK] RUN_ID:        {run_id}")
        print(f"[OK] TRACKING_URI:  {mlflow.get_tracking_uri()}")
        print(f"[OK] REGISTRY_URI:  {mlflow.get_registry_uri()}")
        print(f"[OK] ARTIFACT_URI:  {mlflow.get_artifact_uri()}")
        print(f"[OK] MODEL_URI:     {model_uri}")
        print(f"[OK] Local model →  {dest_dir}")


if __name__ == "__main__":
    main()
