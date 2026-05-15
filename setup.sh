#!/bin/bash
# =============================================
# Phase 1 Setup Script - Fraud Detection MLOps
# Run: bash setup.sh
# =============================================

echo "🚀 Setting up Fraud Detection MLOps Pipeline..."

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize Git
git init
git add .
git commit -m "Initial project structure"

# 4. Initialize DVC
dvc init
git add .dvc
git commit -m "Initialize DVC"

# 5. Configure DVC remote to S3
DVC_REMOTE_BUCKET="s3://fraud-detection/fraud-detection-dvc"

dvc remote add -d awsremote $DVC_REMOTE_BUCKET
dvc remote modify awsremote region us-east-1   # Change to your region

git add .dvc/config
git commit -m "Configure DVC S3 remote"

echo " Setup complete! Next: run python src/preprocess.py"