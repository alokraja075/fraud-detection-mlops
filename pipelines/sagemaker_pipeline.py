"""
sagemaker_pipeline.py
---------------------
Defines the full SageMaker Pipeline:

  Step 1 → ProcessingJob   (data preprocessing via SKLearn Processor)
  Step 2 → TrainingJob     (XGBoost on ml.m5.large)
  Step 3 → EvaluationJob   (compute AUC + metrics.json)
  Step 4 → ConditionStep   (only proceed if AUC >= 0.70)
  Step 5 → ModelRegister   (push to SageMaker Model Registry)
  Step 6 → DeployStep      (create/update real-time endpoint)

Run:
  python pipelines/sagemaker_pipeline.py --action create   # build pipeline
  python pipelines/sagemaker_pipeline.py --action run      # trigger execution
  python pipelines/sagemaker_pipeline.py --action status   # check last run
"""

import argparse
import logging
import os
import boto3
import sagemaker
from botocore.exceptions import NoCredentialsError
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterString,
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.model_metrics import (
    ModelMetrics,
    MetricsSource,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration — edit these ────────────────────────────────────────────────
CONFIG = {
    
    "region":           os.environ.get("AWS_REGION", "us-east-1"),
    "bucket":           os.environ.get("S3_BUCKET",  "fraud-detection-dvc"),
    "role_arn":         os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::779466390141:role/FraudDetectionSageMakerRole"),
    "pipeline_name":    "fraud-detection-pipeline",
    "model_package_group": "fraud-detection-models",
    "endpoint_name":    "fraud-detection-endpoint",

    "processing_instance": "ml.t3.medium",

    "xgboost_image":    sagemaker.image_uris.retrieve(
                            "xgboost", os.environ.get("AWS_REGION", "us-east-1"),
                            version="1.7-1"
                        ) if False else None,   # resolved lazily below
}
# ─────────────────────────────────────────────────────────────────────────────


def _list_nonzero_processing_instances(region: str) -> list[str]:
    """Return instance types with non-zero *processing job usage* quota."""
    sq = boto3.client("service-quotas", region_name=region)
    token = None
    instances: list[str] = []
    while True:
        kwargs = {"ServiceCode": "sagemaker", "MaxResults": 100}
        if token:
            kwargs["NextToken"] = token
        resp = sq.list_service_quotas(**kwargs)
        for q in resp.get("Quotas", []):
            name = (q.get("QuotaName") or "").lower()
            if "for processing job usage" in name and float(q.get("Value", 0.0)) > 0:
                # QuotaName format: "ml.t3.medium for processing job usage"
                instances.append((q.get("QuotaName") or "").split(" ")[0])
        token = resp.get("NextToken")
        if not token:
            break
    # Stable order: smaller first when possible.
    preferred = ["ml.t3.medium", "ml.t3.large", "ml.t3.xlarge"]
    ordered = [i for i in preferred if i in instances] + [i for i in instances if i not in preferred]
    return ordered


def _pick_processing_instance(region: str, desired: str) -> str:
    available = _list_nonzero_processing_instances(region)
    if desired in available:
        return desired
    if available:
        logger.warning(
            "Processing instance '%s' has no quota; falling back to '%s' (available=%s)",
            desired,
            available[0],
            available,
        )
        return available[0]
    logger.warning(
        "No non-zero processing instance quotas detected; keeping desired '%s'", desired
    )
    return desired


def _ensure_raw_prefix_has_object(region: str, bucket: str, prefix: str) -> None:
    """SageMaker Processing fails to download when an input S3 prefix is empty."""
    try:
        s3 = boto3.client("s3", region_name=region)
        raw_prefix = f"{prefix}/raw/"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=raw_prefix, MaxKeys=1)
        if resp.get("KeyCount", 0) > 0:
            return
        key = f"{raw_prefix}.keep"
        logger.warning("Seeding empty S3 prefix s3://%s/%s", bucket, raw_prefix)
        s3.put_object(Bucket=bucket, Key=key, Body=b"placeholder")
    except NoCredentialsError:
        logger.warning(
            "AWS credentials not configured; skipping S3 raw prefix seeding. "
            "Pipeline create/run will also fail until credentials are set."
        )


def get_session():
    boto_session = boto3.Session(region_name=CONFIG["region"])
    return sagemaker.Session(boto_session=boto_session)


def build_pipeline() -> Pipeline:
    session = get_session()
    role    = CONFIG["role_arn"]
    bucket  = CONFIG["bucket"]
    prefix  = "fraud-detection"

    processing_instance = _pick_processing_instance(CONFIG["region"], CONFIG["processing_instance"])

    # ── Pipeline parameters (can be overridden per-run) ──────────────────────
    # NOTE: ProcessingStep `job_arguments` are strings, so these are strings.
    # The training script parses them into numeric types via argparse.
    p_n_estimators  = ParameterString(name="NEstimators",    default_value="300")
    p_max_depth     = ParameterString(name="MaxDepth",       default_value="6")
    p_learning_rate = ParameterString(name="LearningRate",   default_value="0.05")
    p_auc_threshold = ParameterFloat(  name="AucThreshold",   default_value=0.70)
    # ── Step 1: Processing (preprocess raw CSVs) ─────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance,
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    processing_step = ProcessingStep(
        name="FraudPreprocessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/{prefix}/raw",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/{prefix}/processed/train",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/{prefix}/processed/test",
            ),
        ],
        job_arguments=[
            "--input-dir", "/opt/ml/processing/input",
            "--train-output-dir", "/opt/ml/processing/output/train",
            "--test-output-dir", "/opt/ml/processing/output/test",
        ],
        code="src/preprocess.py",
    )

    # ── Step 2: Training (as ProcessingJob to avoid TrainingJob quotas) ───────
    xgb_image = sagemaker.image_uris.retrieve("xgboost", CONFIG["region"], version="1.7-1")

    train_processor = ScriptProcessor(
        image_uri=xgb_image,
        command=["python3"],
        instance_type=processing_instance,
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    train_step = ProcessingStep(
        name="FraudModelTraining",
        processor=train_processor,
        inputs=[
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/model",
                destination=f"s3://{bucket}/{prefix}/model-artifacts",
            ),
        ],
        job_arguments=[
            "--train-dir",
            "/opt/ml/processing/train",
            "--test-dir",
            "/opt/ml/processing/test",
            "--model-dir",
            "/opt/ml/processing/model",
            "--max-depth",
            p_max_depth,
            "--eta",
            p_learning_rate,
            "--num-round",
            p_n_estimators,
        ],
        code="src/train_processing.py",
    )

    # ── Step 3: Evaluation Job ────────────────────────────────────────────────
    # Use the XGBoost image so we can reliably load `xgboost-model`.
    eval_processor = ScriptProcessor(
        image_uri=xgb_image,
        command=["python3"],
        instance_type=processing_instance,
        instance_count=1,
        role=role,
        sagemaker_session=session,
        env={
            "MODEL_DIR": "/opt/ml/processing/model",
            "TEST_DATA_DIR": "/opt/ml/processing/test",
            "REPORT_DIR": "/opt/ml/processing/evaluation",
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="metrics.json",
    )

    evaluation_step = ProcessingStep(
        name="FraudModelEvaluation",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig
                       .Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/{prefix}/evaluation",
            ),
        ],
        code="src/evaluate_processing.py",
        property_files=[evaluation_report],
    )

    # ── Step 4: Condition — only register if AUC >= threshold ────────────────
    auc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="auc",
        ),
        right=p_auc_threshold,
    )

    # ── Step 5: Register model in Model Registry ──────────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{bucket}/{prefix}/evaluation/metrics.json",
            content_type="application/json",
        )
    )

    model_data = Join(
        on="/",
        values=[
            train_step.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
            "model.tar.gz",
        ],
    )

    xgb_model = sagemaker.model.Model(
        image_uri=xgb_image,
        model_data=model_data,
        role=role,
        sagemaker_session=session,
    )

    register_step = RegisterModel(
        name="RegisterFraudModel",
        model=xgb_model,
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge", "ml.c5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=CONFIG["model_package_group"],
        approval_status="Approved",   # auto-approve; change to "PendingManualApproval" for prod
        model_metrics=model_metrics,
    )

    # ── Step 6: Condition step (gate) ─────────────────────────────────────────
    condition_step = ConditionStep(
        name="CheckAUCQualityGate",
        conditions=[auc_condition],
        if_steps=[register_step],
        else_steps=[],   # pipeline just stops gracefully if AUC too low
    )

    # ── Assemble pipeline ─────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=CONFIG["pipeline_name"],
        parameters=[
            p_n_estimators,
            p_max_depth,
            p_learning_rate,
            p_auc_threshold,
        ],
        steps=[
            processing_step,
            train_step,
            evaluation_step,
            condition_step,
        ],
        sagemaker_session=session,
    )

    return pipeline


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["create", "run", "status", "delete"],
        default="create",
        help="create=upsert pipeline definition | run=start execution | status=last run",
    )
    parser.add_argument(
        "--auc-threshold", type=float, default=0.70,
        help="Minimum AUC to register model"
    )
    args = parser.parse_args()

    # Ensure the input prefix exists so ProcessingInput download doesn't fail.
    if args.action in {"create", "run"}:
        _ensure_raw_prefix_has_object(CONFIG["region"], CONFIG["bucket"], "fraud-detection")

    pipeline = build_pipeline()

    if args.action == "create":
        logger.info("Upserting pipeline definition to SageMaker...")
        try:
            pipeline.upsert(role_arn=CONFIG["role_arn"])
        except NoCredentialsError as e:
            raise SystemExit(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "(and AWS_SESSION_TOKEN if applicable) plus AWS_REGION, then retry."
            ) from e
        logger.info(f"✅ Pipeline '{CONFIG['pipeline_name']}' created/updated.")

    elif args.action == "run":
        logger.info("Starting pipeline execution...")
        try:
            execution = pipeline.start(parameters={"AucThreshold": args.auc_threshold})
        except NoCredentialsError as e:
            raise SystemExit(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "(and AWS_SESSION_TOKEN if applicable) plus AWS_REGION, then retry."
            ) from e
        logger.info(f"✅ Execution started: {execution.arn}")
        logger.info("Monitor at: https://console.aws.amazon.com/sagemaker/pipelines")

    elif args.action == "status":
        sm = boto3.client("sagemaker", region_name=CONFIG["region"])
        resp = sm.list_pipeline_executions(
            PipelineName=CONFIG["pipeline_name"],
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        execs = resp.get("PipelineExecutionSummaries", [])
        if execs:
            ex = execs[0]
            logger.info(f"Last execution: {ex['PipelineExecutionArn']}")
            logger.info(f"Status: {ex['PipelineExecutionStatus']}")
            logger.info(f"Started: {ex['StartTime']}")
        else:
            logger.info("No executions found yet.")

    elif args.action == "delete":
        sm = boto3.client("sagemaker", region_name=CONFIG["region"])
        sm.delete_pipeline(PipelineName=CONFIG["pipeline_name"])
        logger.info(f"✅ Pipeline '{CONFIG['pipeline_name']}' deleted.")


if __name__ == "__main__":
    main()
