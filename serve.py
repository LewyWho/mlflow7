from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import mlflow.pyfunc
import pandas as pd

# Load MLflow model from local folder
model = mlflow.pyfunc.load_model("model")

# Expected wine feature names (from sklearn.datasets.load_wine(as_frame=True).feature_names)
EXPECTED_COLS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]
CLASS_LABELS = ["class_0", "class_1", "class_2"]

app = FastAPI(title="Wine RF Service", version="0.1.0")

class PredictRequest(BaseModel):
    columns: List[str] = Field(..., description="Feature names (must match training)")
    data: List[List[float]] = Field(..., description="Rows of feature values")

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    schema = getattr(getattr(model, "metadata", None), "get_input_schema", lambda: None)()
    return {
        "model_flavor": "pyfunc",
        "expected_columns": EXPECTED_COLS,
        "input_schema": str(schema) if schema is not None else None,
        "class_labels": CLASS_LABELS,
    }

@app.post("/predict")
def predict(body: PredictRequest):
    if body.columns != EXPECTED_COLS:
        raise HTTPException(
            status_code=422,
            detail={"msg": "Invalid columns/order", "expected": EXPECTED_COLS, "got": body.columns},
        )
    if not body.data:
        return {"predictions": [], "labels": []}

    df = pd.DataFrame(body.data, columns=body.columns)
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    preds = list(map(int, preds))
    labels = [CLASS_LABELS[i] for i in preds]
    return {"predictions": preds, "labels": labels}
