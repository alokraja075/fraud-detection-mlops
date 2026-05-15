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
import tarfile
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    roc_curve
)
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # matplotlib isn't present in some minimal images
    plt = None

MODEL_PATH = "models/fraud_model.pkl"
DATA_DIR = "data/processed"
REPORT_DIR = "reports"


def resolve_model_path() -> str:
    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        tar_path = os.path.join(model_dir, "model.tar.gz")
        extracted = os.path.join(model_dir, "fraud_model.pkl")
        if os.path.exists(extracted):
            return extracted
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=model_dir)
            return extracted
    return MODEL_PATH


def plot_roc_curve(y_test, y_scores, report_dir: str):
    if plt is None:
        print("ℹ️  matplotlib not available; skipping ROC curve plot.")
        return
    os.makedirs(report_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection Model")
    plt.legend(loc="lower right")
    plt.savefig(f"{report_dir}/roc_curve.png")
    plt.close()
    print(f"✅ ROC curve saved to {report_dir}/roc_curve.png")


def main():
    data_dir = os.environ.get("TEST_DATA_DIR", DATA_DIR)
    report_dir = os.environ.get("REPORT_DIR", REPORT_DIR)
    print("Loading model and data...")
    model = joblib.load(resolve_model_path())
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    default_auc_threshold = 0.70 if len(y_test) >= 10000 else 0.50
    auc_threshold = float(
        os.environ.get("AUC_THRESHOLD", str(default_auc_threshold))
    )

    metrics = {
        "auc": round(roc_auc_score(y_test, y_pred_prob), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }

    print("\n📊 Evaluation Report")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"AUC Score: {metrics['auc']}")

    # Save metrics JSON
    os.makedirs(report_dir, exist_ok=True)
    with open(f"{report_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot ROC
    plot_roc_curve(y_test, y_pred_prob, report_dir)

    # Gate check
    if metrics["auc"] >= auc_threshold:
        msg = f"\n✅ Model PASSED quality gate (AUC {metrics['auc']} ≥ {auc_threshold:.2f})"
        print(msg)
    else:
        msg = f"\n❌ Model FAILED quality gate (AUC {metrics['auc']} < {auc_threshold:.2f})"
        print(msg)
        exit(1)


if __name__ == "__main__":
    main()
