"""Hermetic checks for the P3 Warehouse artifact validator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1] / "sessions" / "msi" / "p3_validate_warehouse.py"
)
_SPEC = importlib.util.spec_from_file_location("p3_validate_warehouse", _VALIDATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


def _write_atlas_full_run(output_dir: Path, concurrency: int) -> list[str]:
    output_dir.mkdir()
    sample_ids = [str(index) for index in range(validator.EXPECTED_DISTANCE_COUNT)]
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for sample_id in sample_ids:
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "answer": "1.0",
                        "error": None,
                        "question_type": "distance",
                        "requested_engine": "scenegraph",
                        "used_engine": "scenegraph",
                    }
                )
                + "\n"
            )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "warehouse",
                "question_filters": ["distance"],
                "engine": "scenegraph",
                "llm_enable_thinking": False,
                "concurrency": concurrency,
                "selected_ids": sample_ids,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "results_summary.json").write_text(
        json.dumps(
            {
                "split": "val",
                "total_samples": validator.EXPECTED_DISTANCE_COUNT,
                "per_category": {"distance": {"count": validator.EXPECTED_DISTANCE_COUNT}},
                "operational": {
                    "selected_count": validator.EXPECTED_DISTANCE_COUNT,
                    "completed_count": validator.EXPECTED_DISTANCE_COUNT,
                    "error_count": 0,
                    "per_question_type": {
                        "distance": {
                            "sample_count": validator.EXPECTED_DISTANCE_COUNT,
                            "completed_count": validator.EXPECTED_DISTANCE_COUNT,
                            "error_count": 0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return sample_ids


def test_full_atlas_validator_requires_declared_concurrency(monkeypatch, tmp_path):
    output_dir = tmp_path / "full"
    sample_ids = _write_atlas_full_run(output_dir, concurrency=4)
    monkeypatch.setattr(validator, "_expected_distance_ids", lambda: sample_ids)

    validator.validate_run("atlas", output_dir, "scenegraph", 4)

    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["concurrency"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not describe the exact full run"):
        validator.validate_run("atlas", output_dir, "scenegraph", 4)
