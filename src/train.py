"""
train.py
--------
Trains XGBoost fraud detection model with MLflow tracking.
Handles class imbalance (fraud is rare ~3.5% of transactions).
"""

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import logging
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────
DATA_DIR   = "data/processed"
MODEL_DIR  = "models"
EXPERIMENT = "fraud-detection-phase1"

PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,   # handles class imbalance
    "use_label_encoder": False,
    "eval_metric":     "auc",
    "random_state":    42,
    "n_jobs":          -1,
}
# ────────────────────────────────────────────────────────


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Oversample minority fraud class."""
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE — Fraud: {y_res.sum()}, Non-fraud: {(y_res==0).sum()}")
    return X_res, y_res


def train_model(X_train, y_train):
    model = XGBClassifier(**PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50
    )
    return model


def evaluate_model(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "auc":       roc_auc_score(y_test, y_pred_prob),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
    }
    
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"AUC Score: {metrics['auc']:.4f}")
    return metrics


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = f"{MODEL_DIR}/fraud_model.pkl"
    joblib.dump(model, path)
    logger.info(f"✅ Model saved to {path}")
    return path


def main():
    mlflow.set_experiment(EXPERIMENT)
    
    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    with mlflow.start_run(run_name="xgboost-fraud-v1"):
        # Log params
        mlflow.log_params(PARAMS)
        
        # Train
        logger.info("Training XGBoost model...")
        model = train_model(X_train_res, y_train_res)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Save & log model
        model_path = save_model(model)
        mlflow.xgboost.log_model(model, "fraud-xgboost-model")
        
        # Gate: fail pipeline if AUC too low
        if metrics["auc"] < 0.70:
            raise ValueError(f"❌ AUC {metrics['auc']:.4f} below threshold 0.70!")
        
        logger.info(f"✅ Training complete! AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()