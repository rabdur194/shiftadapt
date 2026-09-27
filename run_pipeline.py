#!/usr/bin/env python
"""CLI entrypoint to run the full ShiftAdapt pipeline."""
import json
import os
import sys

os.environ.setdefault("USE_MOCK_LLM", "true")
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import run_full_pipeline


def main():
    print("Running ShiftAdapt full pipeline...\n")
    report = run_full_pipeline()
    print(json.dumps(report, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
