"""
preprocess.py
-------------
Downloads Kaggle IEEE fraud dataset, cleans it,
and saves processed train/test splits to data/processed/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load raw transaction data."""
    train_transaction = pd.read_csv(f"{data_dir}/train_transaction.csv")
    train_identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    
    logger.info(f"Transactions shape: {train_transaction.shape}")
    logger.info(f"Identity shape: {train_identity.shape}")
    
    # Merge on TransactionID
    df = train_transaction.merge(train_identity, on="TransactionID", how="left")
    logger.info(f"Merged shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and handle missing values."""
    logger.info("Cleaning data...")
    
    # Drop columns with >50% missing
    threshold = 0.5
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} high-missing columns")
    
    # Fill numeric NaN with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill categorical NaN with 'unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("unknown")
    
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical features."""
    logger.info("Encoding categorical features...")
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def split_and_save(df: pd.DataFrame, output_dir: str = "data/processed"):
    """Split into train/test and save."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Features and target
    target = "isFraud"
    feature_cols = [c for c in df.columns if c not in [target, "TransactionID"]]
    
    X = df[feature_cols]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save splits
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    logger.info(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")
    logger.info(f"✅ Fraud rate (train): {y_train.mean():.4f}")
    logger.info(f"Saved to {output_dir}/")


if __name__ == "__main__":
    df = load_data("data/raw")
    df = clean_data(df)
    df = encode_features(df)
    split_and_save(df)