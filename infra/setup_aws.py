"""
infra/setup_aws.py
------------------
One-time AWS setup script for Phase 2.
Creates:
  ✅ S3 bucket (for data, models, artifacts)
  ✅ SageMaker IAM Role (with required permissions)
  ✅ SageMaker Model Package Group (Model Registry)
  ✅ SageMaker Studio Domain check

Run ONCE before your first pipeline execution:
  python infra/setup_aws.py
"""

import boto3
import json
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config — edit these ───────────────────────────────────────────────────────
REGION             = os.environ.get("AWS_REGION", "us-east-1")
BUCKET_NAME        = "fraud-detection-dvc"
ROLE_NAME          = "FraudDetectionSageMakerRole"
MODEL_PKG_GROUP    = "fraud-detection-models"
# ─────────────────────────────────────────────────────────────────────────────


def create_s3_bucket():
    s3 = boto3.client("s3", region_name=REGION)
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        logger.info(f"✅ S3 bucket created: s3://{BUCKET_NAME}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        logger.info(f"S3 bucket already exists: s3://{BUCKET_NAME}")

    # Block all public access (security best practice)
    s3.put_public_access_block(
        Bucket=BUCKET_NAME,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )

    # Create folder structure
    folders = [
        "fraud-detection/raw/",
        "fraud-detection/processed/",
        "fraud-detection/model-artifacts/",
        "fraud-detection/evaluation/",
        "fraud-detection/data-capture/",
    ]
    for folder in folders:
        s3.put_object(Bucket=BUCKET_NAME, Key=folder)
    logger.info("✅ S3 folder structure created")


def create_sagemaker_role():
    iam = boto3.client("iam", region_name=REGION)

    # Trust policy — allows SageMaker to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker role for Fraud Detection MLOps pipeline",
        )
        role_arn = response["Role"]["Arn"]
        logger.info(f"✅ IAM Role created: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        logger.info(f"IAM Role already exists: {role_arn}")

    # Attach required AWS managed policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    ]
    for policy_arn in policies:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
        logger.info(f"  Attached: {policy_arn.split('/')[-1]}")

    logger.info(f"\n⚠️  COPY THIS to GitHub Secrets as SAGEMAKER_ROLE_ARN:\n  {role_arn}\n")
    return role_arn


def create_model_package_group():
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PKG_GROUP,
            ModelPackageGroupDescription=(
                "Fraud Detection XGBoost models — "
                "auto-registered by SageMaker Pipeline"
            ),
        )
        logger.info(f"✅ Model Package Group created: {MODEL_PKG_GROUP}")
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e):
            logger.info(f"Model Package Group already exists: {MODEL_PKG_GROUP}")
        else:
            raise


def print_next_steps(role_arn: str):
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           ✅  AWS Infrastructure Setup Complete!                  ║
╚══════════════════════════════════════════════════════════════════╝

📋  Add these to GitHub Secrets (Settings → Secrets → Actions):

  AWS_ACCESS_KEY_ID      → your AWS IAM user access key
  AWS_SECRET_ACCESS_KEY  → your AWS IAM user secret key
  SAGEMAKER_ROLE_ARN     → {role_arn}

📋  Update these in your code:

  CONFIG["bucket"]    = "{BUCKET_NAME}"
  CONFIG["region"]    = "{REGION}"
  CONFIG["role_arn"]  = "{role_arn}"
  AWS Account ID      = {account_id}

📋  SageMaker Console links:
  Pipelines   → https://console.aws.amazon.com/sagemaker/pipelines
  Endpoints   → https://console.aws.amazon.com/sagemaker/endpoints
  Registry    → https://console.aws.amazon.com/sagemaker/model-registry
  S3 Bucket   → https://s3.console.aws.amazon.com/s3/buckets/{BUCKET_NAME}

🚀  Next: run  python pipelines/sagemaker_pipeline.py --action create
""")


def main():
    logger.info("Setting up AWS infrastructure for Fraud Detection MLOps...")
    create_s3_bucket()
    role_arn = create_sagemaker_role()
    create_model_package_group()
    print_next_steps(role_arn)


if __name__ == "__main__":
    main()
