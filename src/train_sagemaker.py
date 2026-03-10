"""
train_sagemaker.py
------------------
XGBoost fraud detection training script designed to run
as a SageMaker Training Job.

SageMaker injects environment variables:
  SM_MODEL_DIR       → where to save model artifacts
  SM_CHANNEL_TRAIN   → path to training data
  SM_CHANNEL_TEST    → path to test data
  SM_OUTPUT_DATA_DIR → path for any extra outputs

Run locally:  python src/train_sagemaker.py
Run on SM:    triggered via pipelines/run_training_job.py
"""

import os
import argparse
import json
import logging

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── SageMaker paths (fall back to local defaults) ────────────────────────────
SM_MODEL_DIR       = os.environ.get("SM_MODEL_DIR",       "models")
SM_CHANNEL_TRAIN   = os.environ.get("SM_CHANNEL_TRAIN",   "data/processed")
SM_CHANNEL_TEST    = os.environ.get("SM_CHANNEL_TEST",    "data/processed")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "reports")


# ── Hyper-parameters (overridden by SageMaker HPO if used) ───────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",     type=int,   default=300)
    parser.add_argument("--max-depth",        type=int,   default=6)
    parser.add_argument("--learning-rate",    type=float, default=0.05)
    parser.add_argument("--subsample",        type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--scale-pos-weight", type=float, default=10.0)
    parser.add_argument("--auc-threshold",    type=float, default=0.70)
    parser.add_argument("--mlflow-tracking-uri", type=str, default="")
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(train_dir: str, test_dir: str):
    logger.info(f"Loading train data from {train_dir}")
    X_train = pd.read_csv(os.path.join(train_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(train_dir, "y_train.csv")).squeeze()

    logger.info(f"Loading test data from {test_dir}")
    X_test  = pd.read_csv(os.path.join(test_dir, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(test_dir, "y_test.csv")).squeeze()

    logger.info(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
    logger.info(f"Fraud rate (train): {y_train.mean():.4%}")
    return X_train, X_test, y_train, y_test


# ── SMOTE oversampling ────────────────────────────────────────────────────────
def apply_smote(X, y):
    logger.info("Applying SMOTE for class imbalance...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(
        f"Post-SMOTE → Fraud: {int(y_res.sum())}  "
        f"Non-fraud: {int((y_res == 0).sum())}"
    )
    return X_res, y_res


# ── Training ──────────────────────────────────────────────────────────────────
def train(X_train, y_train, args):
    params = {
        "n_estimators":     args.n_estimators,
        "max_depth":        args.max_depth,
        "learning_rate":    args.learning_rate,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": args.scale_pos_weight,
        "eval_metric":      "auc",
        "use_label_encoder": False,
        "random_state":     42,
        "n_jobs":           -1,
    }
    logger.info(f"Training XGBoost with params: {params}")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=50)
    return model, params


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc":       round(float(roc_auc_score(y_test, y_pred_prob)), 4),
        "f1":        round(float(f1_score(y_test, y_pred)),           4),
        "precision": round(float(precision_score(y_test, y_pred)),    4),
        "recall":    round(float(recall_score(y_test, y_pred)),       4),
    }

    logger.info("\n" + classification_report(y_test, y_pred,
                                             target_names=["Legit", "Fraud"]))
    logger.info(f"Metrics: {metrics}")
    return metrics


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model, metrics, model_dir, output_dir):
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Model pickle (SageMaker expects model.tar.gz in SM_MODEL_DIR)
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved → {model_path}")

    # Metrics JSON (picked up by SageMaker Model Registry)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # Feature importance
    fi = pd.Series(model.feature_importances_).sort_values(ascending=False)
    fi.to_csv(os.path.join(output_dir, "feature_importance.csv"))
    logger.info("Feature importance saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Optional: connect to remote MLflow (e.g. on EC2 or SageMaker Studio)
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection-sagemaker")

    X_train, X_test, y_train, y_test = load_data(
        SM_CHANNEL_TRAIN, SM_CHANNEL_TEST
    )
    X_res, y_res = apply_smote(X_train, y_train)

    with mlflow.start_run(run_name="xgboost-sagemaker-v2"):
        model, params = train(X_res, y_res, args)
        mlflow.log_params(params)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        save_artifacts(model, metrics, SM_MODEL_DIR, SM_OUTPUT_DATA_DIR)
        mlflow.xgboost.log_model(model, "fraud-xgboost-sagemaker")

        # ── Quality gate ────────────────────────────────────────────────────
        if metrics["auc"] < args.auc_threshold:
            raise ValueError(
                f"❌  AUC {metrics['auc']} is below threshold "
                f"{args.auc_threshold}. Stopping pipeline."
            )
        logger.info(
            f"✅  Quality gate passed — AUC {metrics['auc']} "
            f"≥ {args.auc_threshold}"
        )


if __name__ == "__main__":
    main()
