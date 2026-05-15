"""invoke_endpoint.py

Invoke a SageMaker real-time endpoint for the fraud detection model.

Defaults are wired to this repo's pipeline outputs:
- Reads test features from S3: fraud-detection/processed/test/X_test.csv
- Optionally reads test labels from S3: fraud-detection/processed/test/y_test.csv

Example:
  python src/invoke_endpoint.py --n 10

Notes:
- Payload format: one CSV record without header.
- Endpoint is expected to return a single probability per line.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from typing import List, Optional, Tuple


def _auc_trapezoid(y_true, y_score) -> Optional[float]:
    # Minimal AUC (ROC) implementation without sklearn.
    import numpy as np

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos = float((y_true == 1).sum())
    neg = float((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]

    tps = (y_true == 1).cumsum()
    fps = (y_true == 0).cumsum()

    tpr = tps / pos
    fpr = fps / neg

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))


def _f1_at_threshold(y_true, y_score, threshold: float) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def _read_csv_rows_from_s3(
    s3_client,
    bucket: str,
    key: str,
    n: int,
) -> Tuple[List[str], List[List[str]]]:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows: List[List[str]] = []
    for _, row in zip(range(n), reader):
        rows.append(row)
    return header, rows


def _read_single_column_csv_from_s3(
    s3_client,
    bucket: str,
    key: str,
    n: int,
) -> List[int]:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # skip header
    values: List[int] = []
    for _, row in zip(range(n), reader):
        if not row:
            continue
        values.append(int(float(row[0])))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--endpoint-name", default=os.environ.get("SAGEMAKER_ENDPOINT", "fraud-detection-endpoint"))
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "fraud-detection-dvc"))
    parser.add_argument("--x-test-key", default="fraud-detection/processed/test/X_test.csv")
    parser.add_argument("--y-test-key", default="fraud-detection/processed/test/y_test.csv")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import boto3  # type: ignore
    except ModuleNotFoundError as exc:
        print("Missing dependency: boto3", file=sys.stderr)
        print("\nFix:", file=sys.stderr)
        print("- If you have a venv in this repo, run:", file=sys.stderr)
        print("    ./venv/bin/pip install -r requirements.txt", file=sys.stderr)
        print("    ./venv/bin/python src/invoke_endpoint.py --n 10", file=sys.stderr)
        print("- Or activate the venv:", file=sys.stderr)
        print("    source venv/bin/activate", file=sys.stderr)
        print("    python src/invoke_endpoint.py --n 10", file=sys.stderr)
        raise SystemExit(2) from exc

    s3 = boto3.client("s3", region_name=args.region)
    rt = boto3.client("sagemaker-runtime", region_name=args.region)

    header, rows = _read_csv_rows_from_s3(s3, args.bucket, args.x_test_key, args.n)
    y_true: Optional[List[int]] = None
    try:
        y_true = _read_single_column_csv_from_s3(s3, args.bucket, args.y_test_key, len(rows))
    except Exception:
        y_true = None

    print(f"Endpoint: {args.endpoint_name} ({args.region})")
    print(f"S3 X_test: s3://{args.bucket}/{args.x_test_key}")
    if y_true is not None:
        print(f"S3 y_test: s3://{args.bucket}/{args.y_test_key}")
    print(f"Rows: {len(rows)} | Features: {len(header)}")

    scores: List[float] = []
    for i, row in enumerate(rows, start=1):
        payload = ",".join(row)
        resp = rt.invoke_endpoint(
            EndpointName=args.endpoint_name,
            ContentType="text/csv",
            Accept="text/csv",
            Body=payload.encode("utf-8"),
        )
        out = resp["Body"].read().decode("utf-8").strip()
        score = float(out.split(",")[0])
        scores.append(score)
        label = y_true[i - 1] if y_true is not None and i - 1 < len(y_true) else None
        print(f"{i:02d}. score={score:.6f}" + (f" y={label}" if label is not None else ""))

    # Summary
    if y_true is not None and len(y_true) == len(scores):
        auc = _auc_trapezoid(y_true, scores)
        f1 = _f1_at_threshold(y_true, scores, threshold=args.threshold)
        print("Summary:")
        print(f"- threshold={args.threshold:.2f}")
        print(f"- auc={auc:.4f}" if auc is not None else "- auc=N/A")
        print(f"- f1={f1:.4f}")


if __name__ == "__main__":
    main()
