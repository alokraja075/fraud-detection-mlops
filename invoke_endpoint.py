"""Repository entrypoint for invoking the SageMaker endpoint.

This is a thin wrapper around `src/invoke_endpoint.py` so you can run:

  python invoke_endpoint.py --n 10

If you hit missing dependency errors, run:
  ./venv/bin/pip install -r requirements.txt
and then:
  ./venv/bin/python invoke_endpoint.py --n 10
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).parent / "src" / "invoke_endpoint.py"
    if not script.exists():
        raise SystemExit(f"Expected script not found: {script}")

    # Execute the src script as if it were run directly.
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
