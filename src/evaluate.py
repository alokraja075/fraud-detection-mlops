"""
evaluate.py
-----------
Standalone evaluation script — loads saved model and 
runs full evaluation report. Use after training.
"""

import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

MODEL_PATH = "models/fraud_model.pkl"
DATA_DIR   = "data/processed"
REPORT_DIR = "reports"


def plot_roc_curve(y_test, y_scores):
    os.makedirs(REPORT_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange", lw=2)
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection Model")
    plt.legend(loc="lower right")
    plt.savefig(f"{REPORT_DIR}/roc_curve.png")
    plt.close()
    print(f"✅ ROC curve saved to {REPORT_DIR}/roc_curve.png")


def main():
    print("Loading model and data...")
    model  = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "auc":       round(roc_auc_score(y_test, y_pred_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }
    
    print("\n📊 Evaluation Report")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"AUC Score: {metrics['auc']}")
    
    # Save metrics JSON
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot ROC
    plot_roc_curve(y_test, y_pred_prob)
    
    # Gate check
    if metrics["auc"] >= 0.70:
        print(f"\n✅ Model PASSED quality gate (AUC {metrics['auc']} ≥ 0.70)")
    else:
        print(f"\n❌ Model FAILED quality gate (AUC {metrics['auc']} < 0.70)")
        exit(1)


if __name__ == "__main__":
    main()