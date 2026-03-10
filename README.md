# 🏦 Fraud Detection MLOps — Phase 2: SageMaker

Full enterprise-grade MLOps pipeline using AWS SageMaker.

---

## 📁 Project Structure

```
fraud-detection-mlops/
├── src/
│   ├── preprocess.py          ← Phase 1: data cleaning & splitting
│   ├── train.py               ← Phase 1: local training
│   ├── train_sagemaker.py     ← Phase 2: SageMaker training job ⭐
│   ├── inference.py           ← Phase 2: SageMaker endpoint handler ⭐
│   └── evaluate.py            ← evaluation & ROC curve
├── pipelines/
│   ├── sagemaker_pipeline.py  ← Phase 2: full SM pipeline definition ⭐
│   └── deploy_endpoint.py     ← Phase 2: deploy & test endpoint ⭐
├── infra/
│   └── setup_aws.py           ← Phase 2: one-time AWS setup ⭐
├── .github/workflows/
│   ├── ml-pipeline.yml        ← Phase 1: basic CI/CD
│   └── sagemaker-pipeline.yml ← Phase 2: SageMaker CI/CD ⭐
└── requirements.txt
```

---

## 🚀 Quick Start — Phase 2

### Step 1: AWS Setup (run once)
```bash
pip install -r requirements.txt
python infra/setup_aws.py
```
This creates your S3 bucket, IAM role, and Model Registry group.
Copy the printed `SAGEMAKER_ROLE_ARN` to GitHub Secrets.

### Step 2: Add GitHub Secrets
Go to: **GitHub repo → Settings → Secrets → Actions**
```
AWS_ACCESS_KEY_ID       → your IAM user access key
AWS_SECRET_ACCESS_KEY   → your IAM user secret key
SAGEMAKER_ROLE_ARN      → printed by setup_aws.py
```

### Step 3: Update config values
In `pipelines/sagemaker_pipeline.py` update `CONFIG`:
```python
"bucket":    "your-actual-bucket-name",
"role_arn":  "arn:aws:iam::YOUR_ACCOUNT:role/FraudDetectionSageMakerRole",
"region":    "us-east-1",
```

### Step 4: Create the SageMaker Pipeline
```bash
python pipelines/sagemaker_pipeline.py --action create
```

### Step 5: Trigger a run
```bash
python pipelines/sagemaker_pipeline.py --action run
```

### Step 6: Deploy the model
```bash
python pipelines/deploy_endpoint.py --action deploy
python pipelines/deploy_endpoint.py --action test
```

---

## 🏗️ Pipeline Architecture

```
GitHub Push
    ↓
GitHub Actions (sagemaker-pipeline.yml)
    ↓
SageMaker Pipeline
  ├── Step 1: ProcessingJob  →  preprocess.py  (ml.m5.xlarge)
  ├── Step 2: TrainingJob    →  train_sagemaker.py (ml.m5.xlarge)
  ├── Step 3: EvaluationJob  →  evaluate.py
  ├── Step 4: ConditionStep  →  AUC >= 0.70?
  └── Step 5: ModelRegistry  →  register approved model
      ↓
SageMaker Endpoint (real-time inference)
    ↓
REST API  →  {"prediction": 1, "label": "FRAUD", "fraud_probability": 0.94}
```

---

## 💰 Estimated AWS Cost

| Service              | Usage               | Cost/month |
|----------------------|---------------------|------------|
| S3                   | ~5GB data+models    | ~$0.12     |
| SageMaker Training   | ml.m5.xlarge ~1hr   | ~$0.23     |
| SageMaker Processing | ml.m5.xlarge ~30min | ~$0.12     |
| SageMaker Endpoint   | ml.m5.large 24/7    | ~$70       |
| **Endpoint (dev)**   | Turn off when done  | ~$0        |

> 💡 **Tip**: Delete the endpoint when not testing to save credits:
> ```bash
> python pipelines/deploy_endpoint.py --action delete
> ```

---

## 📊 Monitoring the Pipeline

- **SageMaker Pipelines**: https://console.aws.amazon.com/sagemaker/pipelines
- **Model Registry**: https://console.aws.amazon.com/sagemaker/model-registry
- **Endpoints**: https://console.aws.amazon.com/sagemaker/endpoints
- **S3 artifacts**: https://s3.console.aws.amazon.com/s3/buckets/your-bucket

---

## 🔜 Phase 3 Preview

- Auto-trigger pipeline when new data lands in S3
- CloudWatch alarms for model drift
- A/B testing between model versions
- SageMaker Model Monitor (data quality checks)
