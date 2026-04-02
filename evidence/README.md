# Evidence (GUI Working)

This folder contains screenshots showing the local CSV-upload GUI working end-to-end against the deployed SageMaker endpoint.

- `gui_upload.png` — Upload form (`/`) with region/endpoint/threshold inputs.
- `gui_results.png` — Results page (`/score`) showing scored rows and decisions.

How to reproduce locally:

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn src.gui:app --host 0.0.0.0 --port 8000
```

Open:

- http://localhost:8000

Use `sample_upload.csv` for a quick test.
