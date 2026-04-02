"""
evaluate_processing.py
---------------------
Evaluate an XGBoost model artifact produced by `src/train_processing.py`.

Inputs (provided via SageMaker Processing inputs):
- MODEL_DIR: contains model.tar.gz with `xgboost-model`
- TEST_DATA_DIR: contains X_test.csv and y_test.csv

Outputs:
- REPORT_DIR/metrics.json with at least: auc, f1

Notes:
- Runs inside the XGBoost container (guarantees xgboost is installed).
- Does NOT fail the step when AUC is below threshold; the pipeline ConditionStep
  is responsible for gating model registration.
"""

import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb


def _auc_trapezoid(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC using trapezoidal integration without sklearn."""
    y_true = y_true.astype(int)
    # Sort by score descending
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]

    pos = float(np.sum(y_true == 1))
    neg = float(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / pos
    fpr = fps / neg

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))


def _f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def _resolve_and_extract_model(model_dir: str) -> str:
    """Return path to extracted xgboost-model file."""
    extracted = os.path.join(model_dir, "xgboost-model")
    if os.path.exists(extracted):
        return extracted

    tar_path = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Expected {tar_path} to exist")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = {m.name for m in tf.getmembers()}
        if "xgboost-model" not in members:
            raise FileNotFoundError(
                f"model.tar.gz does not contain 'xgboost-model' (found: {sorted(members)[:10]})"
            )
        tf.extract("xgboost-model", path=model_dir)

    if not os.path.exists(extracted):
        raise FileNotFoundError(f"Extraction failed; {extracted} not found")

    return extracted


def main() -> None:
    model_dir = os.environ.get("MODEL_DIR", "/opt/ml/processing/model")
    test_data_dir = os.environ.get("TEST_DATA_DIR", "/opt/ml/processing/test")
    report_dir = os.environ.get("REPORT_DIR", "/opt/ml/processing/evaluation")

    print("Loading model and data...")
    model_path = _resolve_and_extract_model(model_dir)

    X_test = pd.read_csv(os.path.join(test_data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(test_data_dir, "y_test.csv")).squeeze("columns").to_numpy()

    booster = xgb.Booster()
    booster.load_model(model_path)

    dtest = xgb.DMatrix(X_test)
    y_score = booster.predict(dtest)

    auc = _auc_trapezoid(y_test, y_score)
    f1 = _f1_at_threshold(y_test, y_score, threshold=0.5)

    metrics = {
        "auc": round(float(auc), 4) if np.isfinite(auc) else None,
        "f1": round(float(f1), 4),
    }

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote metrics:", out_path)
    print(metrics)


if __name__ == "__main__":
    main()
