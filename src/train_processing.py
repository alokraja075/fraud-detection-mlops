"""
train_processing.py
------------------
Train an XGBoost model inside a SageMaker *Processing* job.

Why this exists:
- Some AWS accounts have 0 quota for SageMaker *Training Jobs*
- But do have quota for SageMaker *Processing Jobs* (e.g., ml.t3.*)

This script:
- Reads pre-split CSVs (X_train/y_train and X_test/y_test)
- Trains a small XGBoost booster
- Writes a SageMaker-compatible model artifact: model.tar.gz containing `xgboost-model`

Expected input files:
  {train_dir}/X_train.csv
  {train_dir}/y_train.csv
  {test_dir}/X_test.csv
  {test_dir}/y_test.csv

Output:
  {model_dir}/model.tar.gz
"""

import argparse
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model-dir", required=True)

    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--num-round", type=int, default=200)
    return parser.parse_args()


def _load_split(split_dir):
    x_path = os.path.join(split_dir, "X_train.csv")
    y_path = os.path.join(split_dir, "y_train.csv")
    if not os.path.exists(x_path):
        x_path = os.path.join(split_dir, "X_test.csv")
    if not os.path.exists(y_path):
        y_path = os.path.join(split_dir, "y_test.csv")

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns")
    return X, y


def main():
    args = parse_args()

    X_train, y_train = _load_split(args.train_dir)
    X_test, y_test = _load_split(args.test_dir)

    # Handle class imbalance.
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "scale_pos_weight": scale_pos_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "verbosity": 1,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False,
    )

    os.makedirs(args.model_dir, exist_ok=True)

    # SageMaker XGBoost inference container expects `xgboost-model` inside the tar.gz
    model_file = os.path.join(args.model_dir, "xgboost-model")
    booster.save_model(model_file)

    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(model_file, arcname="xgboost-model")

    # Keep the directory clean-ish (optional)
    try:
        os.remove(model_file)
    except OSError:
        pass

    print(f"Wrote model artifact: {tar_path}")


if __name__ == "__main__":
    main()
