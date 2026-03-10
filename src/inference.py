"""
inference.py
------------
SageMaker inference handler.
SageMaker calls these 4 functions automatically when your
endpoint receives a request.

Endpoint flow:
  POST /invocations  →  input_fn → predict_fn → output_fn
  GET  /ping         →  health check (must return 200)
"""

import os
import io
import json
import logging

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── 1. Load model once when container starts ──────────────────────────────────
def model_fn(model_dir: str):
    """
    Called once at container startup.
    Must return the loaded model object.
    """
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("✅ Model loaded successfully")
    return model


# ── 2. Deserialise incoming request body ──────────────────────────────────────
def input_fn(request_body: str, content_type: str = "application/json"):
    """
    Converts raw HTTP body into a pandas DataFrame.

    Accepts:
      application/json  →  {"features": [[v1, v2, ...]]}
      text/csv          →  v1,v2,...\n
    """
    logger.info(f"Received content_type: {content_type}")

    if content_type == "application/json":
        payload = json.loads(request_body)
        # Support both single dict and batch list
        if isinstance(payload, dict) and "features" in payload:
            return pd.DataFrame(payload["features"])
        elif isinstance(payload, list):
            return pd.DataFrame(payload)
        else:
            return pd.DataFrame([payload])

    elif content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body), header=None)

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# ── 3. Run prediction ─────────────────────────────────────────────────────────
def predict_fn(input_data: pd.DataFrame, model):
    """
    Runs model inference.
    Returns dict with prediction label + fraud probability.
    """
    logger.info(f"Running inference on {len(input_data)} row(s)")

    predictions  = model.predict(input_data).tolist()
    probabilities = model.predict_proba(input_data)[:, 1].tolist()

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "prediction":        int(pred),
            "label":             "FRAUD" if pred == 1 else "LEGIT",
            "fraud_probability": round(float(prob), 4),
            "risk_level":        _risk_level(prob),
        })

    return results


def _risk_level(prob: float) -> str:
    """Translate probability into human-readable risk tier."""
    if prob >= 0.80:
        return "HIGH"
    elif prob >= 0.50:
        return "MEDIUM"
    elif prob >= 0.20:
        return "LOW"
    else:
        return "VERY_LOW"


# ── 4. Serialise response ─────────────────────────────────────────────────────
def output_fn(predictions: list, accept: str = "application/json"):
    """
    Converts prediction list back to HTTP response body.
    """
    if accept == "application/json":
        # Return single object for single-row requests
        result = predictions[0] if len(predictions) == 1 else predictions
        return json.dumps(result), "application/json"

    raise ValueError(f"Unsupported accept type: {accept}")
