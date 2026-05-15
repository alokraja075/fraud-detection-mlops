"""
deploy_endpoint.py
------------------
Deploys the latest Approved model from SageMaker Model Registry
to a real-time inference endpoint.

Usage:
  python pipelines/deploy_endpoint.py --action deploy    # create/update endpoint
  python pipelines/deploy_endpoint.py --action test      # send a test request
  python pipelines/deploy_endpoint.py --action delete    # tear down endpoint
"""

import argparse
import json
import logging
import os

import boto3
import sagemaker
from sagemaker import ModelPackage
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_REGION", "us-east-1")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")  # Must be set via env var
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "fraud-detection-endpoint")
MODEL_PKG_GRP = os.environ.get("MODEL_PACKAGE_GROUP", "fraud-detection-models")
INSTANCE_TYPE = os.environ.get("ENDPOINT_INSTANCE", "ml.t2.medium")
S3_BUCKET = os.environ.get("S3_BUCKET", "fraud-detection-dvc")
# ─────────────────────────────────────────────────────────────────────────────


def get_latest_approved_model_arn() -> str:
    """Fetch the latest Approved model package ARN from the registry."""
    sm = boto3.client("sagemaker", region_name=REGION)
    response = sm.list_model_packages(
        ModelPackageGroupName=MODEL_PKG_GRP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(
            f"No Approved models found in group '{MODEL_PKG_GRP}'. "
            "Run the pipeline first and ensure AUC >= threshold."
        )
    arn = packages[0]["ModelPackageArn"]
    logger.info(f"Latest approved model ARN: {arn}")
    return arn


def deploy_endpoint(model_arn: str):
    """Create or update the SageMaker real-time endpoint."""
    if not ROLE_ARN:
        raise RuntimeError("Missing SAGEMAKER_ROLE_ARN environment variable.")

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )

    # Some SDK versions forward unknown kwargs to `Model` which can raise
    # TypeError: Model.__init__() got an unexpected keyword argument 'model_package_arn'.
    # Guard against that by trying `ModelPackage(...)` and falling back to
    # creating a `Model` from the model package description if it fails.
    try:
        model = ModelPackage(
            model_package_arn=model_arn,   # loaded from Model Registry
            role=ROLE_ARN,
            sagemaker_session=session,
        )
    except TypeError:
        logger.warning(
            "ModelPackage(...) failed; falling back to creating a Model from the model package"
        )
        sm = boto3.client("sagemaker", region_name=REGION)
        pkg = sm.describe_model_package(ModelPackageNameOrArn=model_arn)
        # Inference specification contains container image and model artifacts
        inf = pkg.get("InferenceSpecification", {})
        containers = inf.get("Containers", [])
        if not containers:
            raise RuntimeError("No container info found in model package.")
        image_uri = containers[0].get("Image")
        # Model data URI might be under ModelArtifacts or ModelPackageVersion
        model_data_url = None
        if "ModelArtifacts" in pkg:
            model_data_url = pkg["ModelArtifacts"].get("S3ModelArtifacts")
        elif containers[0].get("ModelDataUrl"):
            model_data_url = containers[0].get("ModelDataUrl")
        if not model_data_url:
            # Some packages embed model data in inference spec properties
            model_data_url = pkg.get("ModelPackageDescription")
        model = sagemaker.model.Model(
            image_uri=image_uri,
            model_data=model_data_url,
            role=ROLE_ARN,
            sagemaker_session=session,
        )

    logger.info(f"Deploying to endpoint '{ENDPOINT_NAME}' on {INSTANCE_TYPE}...")
    # Determine supported real-time instance types from the model package, and
    # fall back to a supported instance if the configured one isn't compatible.
    sm = boto3.client("sagemaker", region_name=REGION)
    chosen_instance = INSTANCE_TYPE
    try:
        pkg_desc = sm.describe_model_package(ModelPackageNameOrArn=model_arn)
        supported = pkg_desc.get("InferenceSpecification", {}).get(
            "SupportedRealtimeInferenceInstanceTypes", []
        )
        if supported:
            if INSTANCE_TYPE not in supported:
                logger.warning(
                    "Configured instance '%s' not in model package supported types %s;"
                    " falling back to '%s'",
                    INSTANCE_TYPE,
                    supported,
                    supported[0],
                )
                chosen_instance = supported[0]
    except Exception:
        # If we cannot inspect the package, continue with configured instance
        logger.info("Could not determine supported instance types from model package; using configured instance.")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=chosen_instance,
        endpoint_name=ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        # Enable data capture for monitoring (Phase 3 will use this)
        data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,
            destination_s3_uri=f"s3://{S3_BUCKET}/fraud-detection/data-capture",
        ),
    )
    logger.info(f"✅ Endpoint '{ENDPOINT_NAME}' is LIVE!")
    logger.info(f"   Invoke URL: https://runtime.sagemaker.{REGION}.amazonaws.com"
                f"/endpoints/{ENDPOINT_NAME}/invocations")
    return predictor


def test_endpoint():
    """Send a sample transaction to the live endpoint."""
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )
    predictor = Predictor(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    # Sample transaction (replace feature values with real ones from your dataset)
    sample_transaction = {
        "features": [[
            1234.56,   # TransactionAmt
            0,         # ProductCD encoded
            117.0,     # card1
            0,         # card2
            0,         # addr1
            87,        # dist1
            0.0,       # P_emaildomain encoded
            0.0,       # R_emaildomain encoded
            # ... add remaining features to match your training columns
        ]]
    }

    logger.info("Sending test transaction to endpoint...")
    result = predictor.predict(sample_transaction)
    logger.info(f"✅ Prediction result: {json.dumps(result, indent=2)}")
    return result


def delete_endpoint():
    """Tear down the endpoint to stop incurring costs."""
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info(f"✅ Endpoint '{ENDPOINT_NAME}' deleted.")
    except sm.exceptions.ClientError as e:
        logger.warning(f"Could not delete: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["deploy", "test", "delete"],
        default="deploy",
    )
    args = parser.parse_args()

    if args.action == "deploy":
        model_arn = get_latest_approved_model_arn()
        deploy_endpoint(model_arn)

    elif args.action == "test":
        test_endpoint()

    elif args.action == "delete":
        delete_endpoint()


if __name__ == "__main__":
    main()
