"""
sagemaker_pipeline.py
---------------------
Defines the full SageMaker Pipeline:

  Step 1 → ProcessingJob   (data preprocessing via SKLearn Processor)
  Step 2 → TrainingJob     (XGBoost on ml.m5.xlarge)
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
import json
import logging
import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TransformStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat,
    ParameterString,
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    ModelMetrics,
    MetricsSource,
)
from sagemaker.workflow.execution_variables import ExecutionVariables

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration — edit these ────────────────────────────────────────────────
CONFIG = {
    # ⚠️  Replace with your values
    "region":           os.environ.get("AWS_REGION", "us-east-1"),
    "bucket":           os.environ.get("S3_BUCKET",  "fraud-detection-dvc"),
    "role_arn":         os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::779466390141:role/FraudDetectionSageMakerRole"),
    "pipeline_name":    "fraud-detection-pipeline",
    "model_package_group": "fraud-detection-models",
    "endpoint_name":    "fraud-detection-endpoint",

    # Instance types  (change to ml.t3.medium for cheaper dev)
    "processing_instance": "ml.m5.xlarge",
    "training_instance":   "ml.m5.xlarge",

    # Container image — AWS managed XGBoost
    "xgboost_image":    sagemaker.image_uris.retrieve(
                            "xgboost", os.environ.get("AWS_REGION", "us-east-1"),
                            version="1.7-1"
                        ) if False else None,   # resolved lazily below
}
# ─────────────────────────────────────────────────────────────────────────────


def get_session():
    boto_session = boto3.Session(region_name=CONFIG["region"])
    return sagemaker.Session(boto_session=boto_session)


def build_pipeline() -> Pipeline:
    session = get_session()
    role    = CONFIG["role_arn"]
    bucket  = CONFIG["bucket"]
    prefix  = "fraud-detection"

    # ── Pipeline parameters (can be overridden per-run) ──────────────────────
    p_n_estimators  = ParameterInteger(name="NEstimators",    default_value=300)
    p_max_depth     = ParameterInteger(name="MaxDepth",       default_value=6)
    p_learning_rate = ParameterFloat(  name="LearningRate",   default_value=0.05)
    p_auc_threshold = ParameterFloat(  name="AucThreshold",   default_value=0.70)
    p_instance_type = ParameterString( name="TrainingInstance",
                                       default_value=CONFIG["training_instance"])

    # ── Step 1: Processing (preprocess raw CSVs) ─────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=CONFIG["processing_instance"],
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
        code="src/preprocess.py",
    )

    # ── Step 2: Training Job ──────────────────────────────────────────────────
    xgb_image = sagemaker.image_uris.retrieve(
        "xgboost", CONFIG["region"], version="1.7-1"
    )

    xgb_estimator = Estimator(
        image_uri=xgb_image,
        instance_type=p_instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/{prefix}/model-artifacts",
        role=role,
        sagemaker_session=session,
        hyperparameters={
            "n_estimators":     p_n_estimators,
            "max_depth":        p_max_depth,
            "learning_rate":    p_learning_rate,
            "scale_pos_weight": 10,
            "eval_metric":      "auc",
        },
        entry_point="src/train_sagemaker.py",
        metric_definitions=[
            {"Name": "validation:auc", "Regex": r"AUC Score: ([0-9\.]+)"},
            {"Name": "train:f1",       "Regex": r"f1: ([0-9\.]+)"},
        ],
    )

    training_step = TrainingStep(
        name="FraudModelTraining",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig
                        .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig
                        .Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ── Step 3: Evaluation Job ────────────────────────────────────────────────
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=CONFIG["processing_instance"],
        instance_count=1,
        role=role,
        sagemaker_session=session,
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
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
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
        code="src/evaluate.py",
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

    register_step = RegisterModel(
        name="RegisterFraudModel",
        estimator=xgb_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
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
            p_instance_type,
        ],
        steps=[
            processing_step,
            training_step,
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

    pipeline = build_pipeline()

    if args.action == "create":
        logger.info("Upserting pipeline definition to SageMaker...")
        pipeline.upsert(role_arn=CONFIG["role_arn"])
        logger.info(f"✅ Pipeline '{CONFIG['pipeline_name']}' created/updated.")

    elif args.action == "run":
        logger.info("Starting pipeline execution...")
        execution = pipeline.start(
            parameters={"AucThreshold": args.auc_threshold}
        )
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
