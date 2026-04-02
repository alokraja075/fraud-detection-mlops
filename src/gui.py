"""Minimal GUI to upload a CSV and score it via a SageMaker endpoint.

Run:
  ./venv/bin/uvicorn src.gui:app --host 0.0.0.0 --port 8000

Then open:
  http://localhost:8000

Notes:
- CSV must include a header row.
- Each subsequent row is sent to the endpoint as a single CSV record (no header).
- Endpoint is expected to return a single numeric probability per request.
"""

from __future__ import annotations

import csv
import html
import io
import os
import sys
from typing import List, Tuple

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI(title="Fraud Detection Scoring GUI")


def _require_boto3():
    try:
        import boto3  # type: ignore

        return boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: boto3. Use the repo venv: ./venv/bin/pip install -r requirements.txt"
        ) from exc


def _parse_csv(upload_bytes: bytes) -> Tuple[List[str], List[List[str]]]:
    text = upload_bytes.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows: List[List[str]] = [row for row in reader if row]
    return header, rows


def _render_page(body: str) -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fraud Detection Scoring</title>
</head>
<body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial; margin: 24px;">
  %s
</body>
</html>"""
        % body
    )


@app.get("/", response_class=HTMLResponse)
def index():
    region = os.environ.get("AWS_REGION", "us-east-1")
    endpoint = os.environ.get("SAGEMAKER_ENDPOINT", "fraud-detection-endpoint")
    threshold = os.environ.get("FRAUD_THRESHOLD", "0.5")

    body = f"""
  <h2>Fraud Detection — Upload CSV</h2>
  <form action="/score" method="post" enctype="multipart/form-data">
    <div style="margin-bottom: 12px;">
      <label>Region:</label><br />
      <input name="region" value="{html.escape(region)}" style="width: 360px;" />
    </div>
    <div style="margin-bottom: 12px;">
      <label>Endpoint name:</label><br />
      <input name="endpoint_name" value="{html.escape(endpoint)}" style="width: 360px;" />
    </div>
    <div style="margin-bottom: 12px;">
      <label>Threshold (flag fraud if score ≥ threshold):</label><br />
      <input name="threshold" value="{html.escape(threshold)}" style="width: 120px;" />
    </div>
    <div style="margin-bottom: 12px;">
      <label>CSV file:</label><br />
      <input type="file" name="file" accept=".csv,text/csv" required />
    </div>
    <button type="submit">Score file</button>
  </form>

  <p style="margin-top: 18px; color: #444;">
    CSV must include a header row. Each row should contain only the feature columns expected by the endpoint.
  </p>
"""
    return _render_page(body)


@app.post("/score", response_class=HTMLResponse)
async def score(
    region: str = Form(...),
    endpoint_name: str = Form(...),
    threshold: float = Form(0.5),
    file: UploadFile = File(...),
):
    try:
        upload_bytes = await file.read()
        header, rows = _parse_csv(upload_bytes)

        boto3 = _require_boto3()
        rt = boto3.client("sagemaker-runtime", region_name=region)

        scores: List[float] = []
        flagged = 0
        for row in rows:
            payload = ",".join(row)
            resp = rt.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="text/csv",
                Accept="text/csv",
                Body=payload.encode("utf-8"),
            )
            out = resp["Body"].read().decode("utf-8").strip()
            score_val = float(out.split(",")[0])
            scores.append(score_val)
            if score_val >= threshold:
                flagged += 1

        # Render a small preview table (first 50 rows)
        preview_n = min(50, len(rows))
        table_rows = []
        for i in range(preview_n):
            table_rows.append(
                "<tr>"
                f"<td>{i+1}</td>"
                f"<td>{scores[i]:.6f}</td>"
                f"<td>{'FRAUD' if scores[i] >= threshold else 'OK'}</td>"
                "</tr>"
            )

        body = f"""
  <h2>Results</h2>
  <p><b>Region:</b> {html.escape(region)}<br />
     <b>Endpoint:</b> {html.escape(endpoint_name)}<br />
     <b>File:</b> {html.escape(file.filename or '(upload)')}<br />
     <b>Rows scored:</b> {len(rows)}<br />
     <b>Features:</b> {len(header)}<br />
     <b>Threshold:</b> {threshold:.2f}<br />
     <b>Flagged as fraud:</b> {flagged}</p>

  <h3>Preview (first {preview_n})</h3>
  <table border="1" cellpadding="6" cellspacing="0">
    <thead>
      <tr><th>#</th><th>Score</th><th>Decision</th></tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>

  <p style="margin-top: 16px;"><a href="/">Score another file</a></p>
"""
        return _render_page(body)

    except Exception as exc:
        body = (
            "<h2>Error</h2>"
            f"<pre>{html.escape(str(exc))}</pre>"
            "<p><a href='/'>Back</a></p>"
        )
        return _render_page(body)
