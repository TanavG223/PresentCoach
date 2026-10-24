#!/usr/bin/env python3
"""Evaluate one locked moving-speaker split with anonymous face tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from presentcoach.presentation_dynamic_eval import (
    DynamicEvaluationError,
    atomic_write_json,
    build_dynamic_evaluation_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "test_media" / "dynamic_presentation_manifest.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split",
        choices=("development", "model_output_holdout"),
        default="development",
        help=(
            "Corpus split to evaluate. The untouched model-output holdout runs only "
            "when named explicitly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional split-specific report path.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    default_output = ROOT / "reports" / (
        "presentcoach_dynamic_development.json"
        if arguments.split == "development"
        else "presentcoach_dynamic_holdout.json"
    )
    output_path = (arguments.output or default_output).resolve()
    try:
        report = build_dynamic_evaluation_report(
            manifest_path=arguments.manifest.resolve(),
            project_root=ROOT,
            split=arguments.split,
        )
        atomic_write_json(output_path, report)
    except DynamicEvaluationError as error:
        print(json.dumps({
            "status": "error",
            "split": arguments.split,
            "error": str(error),
            "report_written": False,
        }), flush=True)
        return 1
    gate = report["release_gate"]
    assert isinstance(gate, dict)
    print(json.dumps({
        "status": gate["status"],
        "split": report["split"],
        "case_count": report["case_count"],
        "report": str(output_path),
        "landmarks_persisted": report["landmarks_persisted"],
    }), flush=True)
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
