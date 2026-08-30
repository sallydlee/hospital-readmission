import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type="application/json"):
    """Deserialize the incoming request body into a one-row DataFrame."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    features = json.loads(request_body)
    return pd.DataFrame([features])


def predict_fn(input_df, model):
    return float(model.predict_proba(input_df)[0, 1])


def output_fn(prediction, accept="application/json"):
    """Serialize the prediction back into the {"probability": ...}
    shape api/predictors/sagemaker.py expects."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps({"probability": prediction}), accept
