from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_p3_table as table


@dataclass
class FakeSample:
    sample_id: str
    ground_truth: float = 1.0


def _fake_target(sample_id: str) -> float:
    return {"a": 0.25, "b": 0.5, "c": 0.75, "d": 1.0}.get(sample_id, 1.0)


class FakeBenchmark:
    def __init__(self, sample_ids=("a", "b")):
        self.data = [FakeSample(sample_id, _fake_target(sample_id)) for sample_id in sample_ids]

    def evaluate(self, predictions, output_dir=None):
        assert output_dir is None
        scores = [float(predictions[str(sample.sample_id)]) for sample in self.data]
        mean_score = sum(scores) / len(scores)
        errors = [
            abs(score - sample.ground_truth) / (sample.ground_truth + 1e-8)
            for score, sample in zip(scores, self.data, strict=True)
        ]
        return {
            "split": "validation",
            "total_samples": len(scores),
            "overall_accuracy": mean_score,
            "per_type": {"distance": {"count": len(scores), "scores": scores}},
            "per_category": {
                "distance": {
                    "count": len(scores),
                    "mean_error_rate": sum(errors) / len(errors),
                }
            },
            "detailed_results": [
                {
                    "id": sample.sample_id,
                    "category": "distance",
                    "ground_truth": str(sample.ground_truth),
                    "normalized_prediction": str(score),
                    "score": score,
                }
                for score, sample in zip(scores, self.data, strict=True)
            ],
        }


def _atlas_row(
    sample_id,
    answer,
    *,
    prompt,
    completion,
    calls,
    latency,
    error=None,
    engine="scenegraph",
    grounding=None,
):
    return {
        "sample_id": sample_id,
        "answer": answer,
        "error": error,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "num_calls": calls,
        },
        "latency_seconds": latency,
        "requested_engine": engine,
        "used_engine": engine,
        "grounding_source": grounding,
    }


def _native_row(sample_id, answer, *, prompt, completion, calls, latency, error=None):
    return {
        "sample_id": sample_id,
        "content": answer,
        "error": error,
        "usage": {
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "num_calls": calls,
        },
        "latency_sec": latency,
    }


def _official_summary(answers):
    scores = [float(answer) for answer in answers]
    mean_score = sum(scores) / len(scores)
    targets = [_fake_target(chr(ord("a") + index)) for index in range(len(scores))]
    errors = [
        abs(score - target) / (target + 1e-8) for score, target in zip(scores, targets, strict=True)
    ]
    return {
        "split": "validation",
        "total_samples": len(scores),
        "overall_accuracy": mean_score,
        "per_type": {"distance": {"count": len(scores), "scores": scores}},
        "per_category": {
            "distance": {
                "count": len(scores),
                "mean_error_rate": sum(errors) / len(errors),
            }
        },
        "operational": {"ignored": True},
    }


def _write_run(run_dir: Path, rows, summary):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / table.PREDICTIONS_FILENAME).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / table.SUMMARY_FILENAME).write_text(json.dumps(summary), encoding="utf-8")


def _write_atlas_manifest(run_dir: Path, engine: str, selected_ids=("a", "b")):
    (run_dir / table.MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "benchmark": "warehouse",
                "question_filters": ["distance"],
                "engine": engine,
                "llm_enable_thinking": False,
                "concurrency": 1,
                "model_tiers": {
                    "fast": "openai/qwen3.6-27b",
                    "standard": "openai/qwen3.6-27b",
                    "strong": "openai/qwen3.6-27b",
                    "vision": "openai/qwen3.6-27b",
                },
                "seed": 42,
                "selected_ids": list(selected_ids),
            }
        ),
        encoding="utf-8",
    )


def _write_native_metadata(run_dir: Path, endpoint_manifest: Path):
    endpoint_url = "http://compute-node.invalid:8000/v1"
    (run_dir / table.CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "benchmark": "warehouse",
                "question_type": ["distance"],
                "llm_model": "qwen3.6-27b",
                "llm_base_url": endpoint_url,
                "limit": None,
                "sample_ids": None,
                "concurrency": 1,
            }
        ),
        encoding="utf-8",
    )
    endpoint_manifest.write_text(
        json.dumps({"model": "qwen3.6-27b", "url": endpoint_url}), encoding="utf-8"
    )
    (run_dir / table.NATIVE_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "benchmark": "warehouse",
                "question_filters": ["distance"],
                "concurrency": 1,
                "endpoint_model": "qwen3.6-27b",
                "selection": {
                    "seed": 42,
                    "expected_count": 2,
                    "selected_ids": ["a", "b"],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_accounting(
    run_dir: Path,
    *,
    system_key: str,
    row_format: str,
    vllm_job_id: str,
    gpu_job_id: str | None = None,
) -> Path:
    attempt = run_dir / "accounting_attempts" / f"{vllm_job_id}-0" / "attempt_gate.json"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    predictions = run_dir / table.PREDICTIONS_FILENAME
    endpoint_url = f"http://node-{vllm_job_id}:8000/v1"
    metrics_url = f"http://node-{vllm_job_id}:8000/metrics"
    endpoint_manifest_path = attempt.parent / "endpoint_manifest.json"
    endpoint_manifest_path.write_text(
        json.dumps(
            {
                "model": "qwen3.6-27b",
                "reasoning_parser": "qwen3",
                "job_id": vllm_job_id,
                "url": endpoint_url,
            }
        ),
        encoding="utf-8",
    )

    def artifact(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": table._sha256(path),
            "size_bytes": path.stat().st_size,
        }

    endpoint_manifest = artifact(endpoint_manifest_path)
    counter_base = {
        "schema_version": 1,
        "endpoint_model": "qwen3.6-27b",
        "endpoint_reasoning_parser": "qwen3",
        "endpoint_job_id": vllm_job_id,
        "endpoint_url": endpoint_url,
        "metrics_url": metrics_url,
        "endpoint_manifest_sha256": endpoint_manifest["sha256"],
        "counter_families": ["vllm:request_success_total"],
        "terminal_reasons": ["stop", "length", "abort", "error", "repetition"],
        "gauges": {"running": 0, "waiting": 0},
    }
    before_path = attempt.parent / "counters_before.json"
    before_path.write_text(
        json.dumps(
            {
                **counter_base,
                "collected_at_utc": "2026-07-11T00:00:00+00:00",
                "counters": {
                    "stop": 0,
                    "length": 0,
                    "abort": 0,
                    "error": 0,
                    "repetition": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    after_path = attempt.parent / "counters_after.json"
    after_path.write_text(
        json.dumps(
            {
                **counter_base,
                "collected_at_utc": "2026-07-11T00:01:00+00:00",
                "counters": {
                    "stop": 2,
                    "length": 0,
                    "abort": 0,
                    "error": 0,
                    "repetition": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    service = {
        "endpoint_model": "qwen3.6-27b",
        "endpoint_reasoning_parser": "qwen3",
        "vllm_job_id": vllm_job_id,
        "gpu_tool_job_id": gpu_job_id,
        "endpoint_url": endpoint_url,
        "metrics_url": metrics_url,
        "endpoint_manifest": endpoint_manifest,
    }
    start_path = attempt.parent / "start.json"
    start_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "p3_vllm_accounting_attempt_start",
                "started_at_utc": "2026-07-11T00:00:01+00:00",
                "system_key": system_key,
                "row_format": row_format,
                "expected_count": 2,
                "concurrency": 1,
                "output_dir": str(run_dir.resolve()),
                "predictions": str(predictions.resolve()),
                "service": service,
                "counter_before": artifact(before_path),
            }
        ),
        encoding="utf-8",
    )
    nested_artifacts = {
        "start": artifact(start_path),
        "counter_before": artifact(before_path),
        "counter_after": artifact(after_path),
        "endpoint_manifest": endpoint_manifest,
    }
    attempt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "kind": "p3_vllm_accounting_attempt_gate",
                "system_key": system_key,
                "row_format": row_format,
                "expected_count": 2,
                "concurrency": 1,
                "runner_exit": 0,
                "termination": "normal",
                "output_dir": str(run_dir.resolve()),
                "predictions": str(predictions.resolve()),
                "newly_completed_count": 2,
                "newly_completed_ids": ["a", "b"],
                "journaled_call_delta": 2,
                "terminal_counter_deltas": {
                    "stop": 2,
                    "length": 0,
                    "abort": 0,
                    "error": 0,
                    "repetition": 0,
                },
                "unattributed_stop_calls": 0,
                "service": service,
                "artifacts": nested_artifacts,
            }
        ),
        encoding="utf-8",
    )
    accounting = run_dir / "full_vllm_accounting.json"
    accounting.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "kind": "p3_full_vllm_accounting",
                "system_key": system_key,
                "row_format": row_format,
                "expected_count": 2,
                "concurrency": 1,
                "output_dir": str(run_dir.resolve()),
                "predictions": {
                    "path": str(predictions.resolve()),
                    "sha256": table._sha256(predictions),
                    "size_bytes": predictions.stat().st_size,
                },
                "attempt_count": 1,
                "interrupted_attempt_count": 0,
                "attempt_statuses": ["pass"],
                "attempts": [
                    {
                        "path": str(attempt.resolve()),
                        "sha256": table._sha256(attempt),
                        "size_bytes": attempt.stat().st_size,
                        "status": "pass",
                        "nested_artifacts": nested_artifacts,
                    }
                ],
                "completed_ids": ["a", "b"],
                "vllm_job_ids": [vllm_job_id],
                "gpu_tool_job_ids": [gpu_job_id] if gpu_job_id is not None else [],
                "endpoint_urls": [endpoint_url],
                "metrics_urls": [metrics_url],
                "endpoint_manifests": [endpoint_manifest],
                "terminal_counter_totals": {
                    "stop": 2,
                    "length": 0,
                    "abort": 0,
                    "error": 0,
                    "repetition": 0,
                },
                "journaled_call_total": 2,
                "unattributed_stop_calls": 0,
            }
        ),
        encoding="utf-8",
    )
    return accounting


def _valid_systems(tmp_path: Path):
    scenegraph = tmp_path / "scenegraph"
    metric = tmp_path / "metric"
    native = tmp_path / "native"
    _write_run(
        scenegraph,
        [
            _atlas_row("a", "", prompt=1, completion=1, calls=1, latency=0.1, error="old"),
            _atlas_row("a", "0.2", prompt=10, completion=2, calls=1, latency=1.0),
            _atlas_row("b", "0.4", prompt=20, completion=4, calls=2, latency=3.0),
        ],
        _official_summary(["0.2", "0.4"]),
    )
    _write_atlas_manifest(scenegraph, "scenegraph")
    _write_run(
        metric,
        [
            _atlas_row(
                "a",
                "0.6",
                prompt=30,
                completion=6,
                calls=3,
                latency=5.0,
                engine="metric",
                grounding="rle+da3",
            ),
            _atlas_row(
                "b",
                "0.8",
                prompt=40,
                completion=8,
                calls=4,
                latency=7.0,
                engine="metric",
                grounding="rle+da3",
            ),
        ],
        _official_summary(["0.6", "0.8"]),
    )
    _write_atlas_manifest(metric, "metric")
    _write_run(
        native,
        [
            _native_row("a", "0.1", prompt=50, completion=10, calls=5, latency=9.0),
            _native_row("b", "0.3", prompt=70, completion=14, calls=7, latency=11.0),
        ],
        _official_summary(["0.1", "0.3"]),
    )
    endpoint_manifest = tmp_path / "endpoint.json"
    _write_native_metadata(native, endpoint_manifest)
    scenegraph_accounting = _write_accounting(
        scenegraph,
        system_key="scenegraph",
        row_format="atlas",
        vllm_job_id="100",
    )
    metric_accounting = _write_accounting(
        metric,
        system_key="metric",
        row_format="atlas",
        vllm_job_id="101",
        gpu_job_id="201",
    )
    native_accounting = _write_accounting(
        native,
        system_key="native",
        row_format="native",
        vllm_job_id="102",
        gpu_job_id="202",
    )
    return [
        table.SystemSpec(
            "scenegraph",
            "Spatial Atlas scenegraph",
            "atlas",
            scenegraph,
            "language scene graph",
            engine="scenegraph",
            accounting_path=scenegraph_accounting,
        ),
        table.SystemSpec(
            "metric",
            "Spatial Atlas metric",
            "atlas",
            metric,
            "rle+da3",
            engine="metric",
            accounting_path=metric_accounting,
        ),
        table.SystemSpec(
            "native",
            "SpatialClaw native",
            "native",
            native,
            "sam3+da3 code action",
            endpoint_manifest=endpoint_manifest,
            accounting_path=native_accounting,
        ),
    ]


def test_build_result_tables_happy_path_uses_latest_rows_and_actual_usage(tmp_path):
    systems = _valid_systems(tmp_path)

    markdown_path, csv_path, gate_path, results = table.build_result_tables(
        FakeBenchmark(),
        systems,
        benchmark_name="warehouse",
        expected_count=2,
        output_dir=tmp_path / "out",
        question_types=["distance"],
    )

    assert [result.official_score for result in results] == pytest.approx([0.3, 0.7, 0.2])
    expected_errors = [
        ((abs(0.2 - 0.25) / (0.25 + 1e-8)) + (abs(0.4 - 0.5) / (0.5 + 1e-8))) / 2,
        ((abs(0.6 - 0.25) / (0.25 + 1e-8)) + (abs(0.8 - 0.5) / (0.5 + 1e-8))) / 2,
        ((abs(0.1 - 0.25) / (0.25 + 1e-8)) + (abs(0.3 - 0.5) / (0.5 + 1e-8))) / 2,
    ]
    assert [result.official_distance_mean_relative_error for result in results] == pytest.approx(
        expected_errors
    )
    assert [result.distance_relative_error_count for result in results] == [2, 2, 2]
    assert results[0].mean_prompt_tokens == 15
    assert results[0].mean_completion_tokens == 3
    assert results[0].mean_total_tokens == 18
    assert results[0].mean_calls == 1.5
    assert results[0].mean_latency_seconds == 2
    assert results[2].mean_total_tokens == 72
    assert [result.spec.concurrency for result in results] == [1, 1, 1]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Benchmark: `warehouse`" in markdown
    assert (
        "| Spatial Atlas metric | rle+da3 | Qwen3.6-27B | "
        "Warehouse validation distance | 2 | 0.700000 |"
    ) in markdown
    assert str((systems[1].run_dir / table.PREDICTIONS_FILENAME).resolve()) in markdown
    assert "Mean per-row latency at declared concurrency (s)" in markdown
    assert (
        "Latency definition: mean measured per-row latency at the declared concurrency." in markdown
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["system"] for row in rows] == [
        "Spatial Atlas scenegraph",
        "Spatial Atlas metric",
        "SpatialClaw native",
    ]
    assert rows[2]["grounding"] == "sam3+da3 code action"
    assert rows[2]["backbone"] == "Qwen3.6-27B"
    assert rows[2]["question_slice"] == "Warehouse validation distance"
    assert rows[2]["mean_total_tokens"] == "72.000000"
    assert [row["concurrency"] for row in rows] == ["1", "1", "1"]
    assert rows[0]["mean_per_row_latency_seconds_at_declared_concurrency"] == "2.000000"
    assert [row["accounted_attempts"] for row in rows] == ["1", "1", "1"]
    assert [row["distance_relative_error_count"] for row in rows] == ["2", "2", "2"]
    assert [float(row["official_distance_mean_relative_error"]) for row in rows] == (
        pytest.approx(expected_errors)
    )
    assert [row["vllm_abnormal_calls"] for row in rows] == ["0", "0", "0"]
    assert [row["unattributed_stop_calls"] for row in rows] == ["0", "0", "0"]
    assert [row["vllm_stop_calls"] for row in rows] == ["2", "2", "2"]
    for reason in ("length", "abort", "error", "repetition"):
        assert {row[f"vllm_{reason}_calls"] for row in rows} == {"0"}
    assert {row["journaled_call_total"] for row in rows} == {"2"}
    assert Path(rows[0]["summary_path"]).is_absolute()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["schema_version"] == 1
    assert gate["status"] == "pass"
    assert gate["benchmark"] == "warehouse"
    assert gate["backbone"] == "Qwen3.6-27B"
    assert gate["question_slice"] == "Warehouse validation distance"
    assert gate["expected_count"] == 2
    assert gate["latency_definition"] == ("mean measured per-row latency at declared concurrency")
    assert gate["system_keys"] == ["scenegraph", "metric", "native"]
    assert [system["concurrency"] for system in gate["systems"]] == [1, 1, 1]
    assert [system["accounted_attempts"] for system in gate["systems"]] == [1, 1, 1]
    assert [system["distance_relative_error_count"] for system in gate["systems"]] == [2, 2, 2]
    assert [
        system["official_distance_mean_relative_error"] for system in gate["systems"]
    ] == pytest.approx(expected_errors)
    assert all(system["accounting"]["sha256"] for system in gate["systems"])
    assert gate["systems"][0][
        "mean_per_row_latency_seconds_at_declared_concurrency"
    ] == pytest.approx(2.0)
    assert gate["systems"][0]["terminal_counter_totals"] == {
        "stop": 2,
        "length": 0,
        "abort": 0,
        "error": 0,
        "repetition": 0,
    }
    assert gate["markdown"]["sha256"] == table._sha256(markdown_path)
    assert gate["csv"]["sha256"] == table._sha256(csv_path)
    assert not list((tmp_path / "out").glob("*.tmp-*"))


def test_tables_preserve_nondefault_concurrency_provenance(tmp_path):
    systems = _valid_systems(tmp_path)
    systems = [replace(spec, concurrency=4) for spec in systems]
    for spec in systems[:2]:
        manifest_path = spec.run_dir / table.MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["concurrency"] = 4
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    native_manifest_path = systems[2].run_dir / table.NATIVE_MANIFEST_FILENAME
    native_manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
    native_manifest["concurrency"] = 4
    native_manifest_path.write_text(json.dumps(native_manifest), encoding="utf-8")
    for spec in systems:
        assert spec.accounting_path is not None
        accounting = json.loads(spec.accounting_path.read_text(encoding="utf-8"))
        accounting["concurrency"] = 4
        attempt_path = Path(accounting["attempts"][0]["path"])
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt["concurrency"] = 4
        attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
        accounting["attempts"][0]["sha256"] = table._sha256(attempt_path)
        accounting["attempts"][0]["size_bytes"] = attempt_path.stat().st_size
        spec.accounting_path.write_text(json.dumps(accounting), encoding="utf-8")

    markdown_path, csv_path, gate_path, _results = table.build_result_tables(
        FakeBenchmark(),
        systems,
        benchmark_name="warehouse",
        expected_count=2,
        output_dir=tmp_path / "out-c4",
        question_types=["distance"],
        atlas_concurrency=4,
    )

    assert "| 4 |" in markdown_path.read_text(encoding="utf-8")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert {row["concurrency"] for row in csv.DictReader(stream)} == {"4"}
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert [system["concurrency"] for system in gate["systems"]] == [4, 4, 4]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("system_key", "metric", "full accounting mismatch"),
        ("completed_ids", ["a"], "exact sample IDs"),
        (
            "terminal_counter_totals",
            {"stop": 2, "length": 1, "abort": 0, "error": 0, "repetition": 0},
            "unreconciled vLLM calls",
        ),
        ("unattributed_stop_calls", 1, "unreconciled vLLM calls"),
    ],
)
def test_full_accounting_rejects_wrong_or_unreconciled_metadata(
    tmp_path, field, replacement, message
):
    spec = _valid_systems(tmp_path)[0]
    assert spec.accounting_path is not None
    payload = json.loads(spec.accounting_path.read_text(encoding="utf-8"))
    payload[field] = replacement
    spec.accounting_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        table._validate_full_accounting(spec, expected_count=2, expected_ids=["a", "b"])


def test_full_accounting_rejects_tampered_attempt_and_cross_system_service_reuse(tmp_path):
    systems = _valid_systems(tmp_path)
    scenegraph = systems[0]
    assert scenegraph.accounting_path is not None
    accounting = json.loads(scenegraph.accounting_path.read_text(encoding="utf-8"))
    attempt_path = Path(accounting["attempts"][0]["path"])
    attempt_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash or size changed"):
        table._validate_full_accounting(scenegraph, expected_count=2, expected_ids=["a", "b"])

    systems = _valid_systems(tmp_path / "reuse")
    metric = systems[1]
    assert metric.accounting_path is not None
    _write_accounting(
        metric.run_dir,
        system_key="metric",
        row_format="atlas",
        vllm_job_id="100",
        gpu_job_id="201",
    )
    with pytest.raises(ValueError, match="reused a vLLM service"):
        table.build_result_tables(
            FakeBenchmark(),
            systems,
            benchmark_name="warehouse",
            expected_count=2,
            output_dir=tmp_path / "reuse-out",
            question_types=["distance"],
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("\n", "is blank"),
        ("{bad}\n", "invalid JSON"),
        ("[]\n", "not a JSON object"),
        (json.dumps({"answer": "1"}) + "\n", "has no sample_id"),
        (json.dumps({"sample_id": ""}) + "\n", "has no sample_id"),
    ],
)
def test_load_latest_rows_rejects_invalid_journal_rows(tmp_path, payload, message):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / table.PREDICTIONS_FILENAME).write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        table._load_latest_rows(run_dir)


def test_load_latest_rows_requires_predictions_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="predictions journal not found"):
        table._load_latest_rows(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "latest rows, expected"),
        (lambda rows: rows[0].update(error="failed"), "has error"),
        (lambda rows: rows[0].update(answer="   "), "blank answer"),
    ],
)
def test_validated_rows_rejects_incomplete_latest_state(tmp_path, mutation, message):
    run_dir = tmp_path / "run"
    rows = [
        _atlas_row("a", "1", prompt=1, completion=1, calls=1, latency=1),
        _atlas_row("b", "1", prompt=1, completion=1, calls=1, latency=1),
    ]
    mutation(rows)
    _write_run(run_dir, rows, _official_summary(["1", "1"]))
    spec = table.SystemSpec("key", "System", "atlas", run_dir, "grounding")
    with pytest.raises(ValueError, match=message):
        table._validated_rows(spec, 2)


@pytest.mark.parametrize(
    ("spec_engine", "row_changes", "grounding", "message"),
    [
        (None, {}, "grounding", "explicit Atlas engine"),
        ("scenegraph", {"requested_engine": "metric"}, "grounding", "requested_engine"),
        ("scenegraph", {"used_engine": "metric"}, "grounding", "used_engine"),
        ("metric", {"grounding_source": "sam3+da3"}, "rle+da3", "grounding_source"),
    ],
)
def test_validated_rows_rejects_atlas_engine_or_grounding_misattribution(
    tmp_path, spec_engine, row_changes, grounding, message
):
    run_dir = tmp_path / "run"
    row = _atlas_row(
        "a",
        "1",
        prompt=1,
        completion=1,
        calls=1,
        latency=1,
        engine=spec_engine or "scenegraph",
        grounding=grounding if spec_engine == "metric" else None,
    )
    row.update(row_changes)
    _write_run(run_dir, [row], _official_summary(["1"]))
    spec = table.SystemSpec("key", "System", "atlas", run_dir, grounding, engine=spec_engine)
    with pytest.raises(ValueError, match=message):
        table._validated_rows(spec, 1)


def test_answer_supports_both_formats_and_rejects_unknown():
    assert table._answer({"answer": 4}, "atlas") == "4"
    assert table._answer({"content": " x "}, "native") == "x"
    assert table._answer({"answer": None}, "atlas") == ""
    with pytest.raises(ValueError, match="unsupported row format"):
        table._answer({}, "other")


@pytest.mark.parametrize(
    ("row", "row_format", "message"),
    [
        ({"usage": None, "latency_seconds": 1}, "atlas", "usage must be"),
        (
            {
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": 1,
                    "total_tokens": 1,
                    "num_calls": 1,
                },
                "latency_seconds": 1,
            },
            "atlas",
            "prompt_tokens must be numeric",
        ),
        (
            {
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": -1,
                    "total_tokens": 0,
                    "num_calls": 1,
                },
                "latency_seconds": 1,
            },
            "atlas",
            "completion_tokens must be finite and nonnegative",
        ),
        (
            {
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": float("inf"),
                    "num_calls": 1,
                },
                "latency_seconds": 1,
            },
            "atlas",
            "total_tokens must be finite and nonnegative",
        ),
        (
            {
                "usage": {
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 1,
                    "num_calls": True,
                },
                "latency_sec": 1,
            },
            "native",
            "num_calls must be numeric",
        ),
        (
            {
                "usage": {
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 1,
                    "num_calls": 1,
                },
                "latency_sec": "slow",
            },
            "native",
            "latency_sec must be numeric",
        ),
    ],
)
def test_row_metrics_rejects_missing_or_invalid_values(row, row_format, message):
    with pytest.raises(ValueError, match=message):
        table._row_metrics(row, row_format, "sample")


def test_row_metrics_rejects_unknown_format():
    with pytest.raises(ValueError, match="unsupported row format"):
        table._row_metrics({"usage": {}}, "other", "sample")


def test_row_metrics_rejects_inconsistent_atlas_total_tokens():
    row = _atlas_row("a", "1", prompt=3, completion=2, calls=1, latency=1)
    row["usage"]["total_tokens"] = 6
    with pytest.raises(ValueError, match="must equal prompt_tokens plus completion_tokens"):
        table._row_metrics(row, "atlas", "sample")


def test_model_identity_normalizes_provider_and_rejects_blank_values():
    assert table._model_identity("openai/Qwen3.6-27B") == "qwen3627b"
    assert table._model_identity(None) == ""
    assert table._model_identity("  ") == ""


@pytest.mark.parametrize("value", [None, True, -0.1, 1.1, float("nan")])
def test_official_score_requires_probability(value):
    with pytest.raises(ValueError, match="overall_accuracy"):
        table._official_score({"overall_accuracy": value})


@pytest.mark.parametrize(
    ("content", "error_type", "message"),
    [
        (None, FileNotFoundError, "results summary not found"),
        ("{bad}", ValueError, "invalid results summary JSON"),
        ("[]", ValueError, "not a JSON object"),
    ],
)
def test_load_summary_validates_file(tmp_path, content, error_type, message):
    if content is not None:
        (tmp_path / table.SUMMARY_FILENAME).write_text(content, encoding="utf-8")
    with pytest.raises(error_type, match=message):
        table._load_summary(tmp_path)


@pytest.mark.parametrize(
    ("expected", "stored", "message"),
    [
        ({"a": 1}, [], "expected an object"),
        ({"a": 1}, {}, "missing key"),
        ([1], [1, 2], "list shape differs"),
        (1.0, "1", "expected numeric value"),
        (1.0, 2.0, "recomputed"),
        ("a", "b", "recomputed"),
    ],
)
def test_assert_summary_matches_reports_structural_and_value_drift(expected, stored, message):
    with pytest.raises(ValueError, match=message):
        table._assert_summary_matches(expected, stored)


def test_assert_summary_matches_skips_details_and_accepts_close_numbers():
    table._assert_summary_matches(
        {"overall_accuracy": 0.3, "detailed_results": [{"x": 1}]},
        {"overall_accuracy": 0.30000000001},
    )


def test_derive_selected_ids_supports_full_seeded_and_explicit_selections():
    samples = [FakeSample(value) for value in ("a", "b", "c", "d")]
    assert table._derive_selected_ids(samples, 4, 42) == ["a", "b", "c", "d"]
    assert table._derive_selected_ids(samples, 2, 42) == ["c", "b"]
    assert table._derive_selected_ids(samples, 2, 42, ["d", "b"]) == ["b", "d"]


@pytest.mark.parametrize(
    ("samples", "expected_count", "explicit_ids", "message"),
    [
        (("a", "a"), 1, None, "duplicate sample IDs"),
        (("a", "b"), 2, ["a", "a"], "sample_ids contains duplicates"),
        (("a", "b"), 1, ["c"], "sample_ids are absent"),
        (("a", "b"), 2, ["a"], "derived selection has 1 samples"),
    ],
)
def test_derive_selected_ids_rejects_ambiguous_or_incomplete_selection(
    samples, expected_count, explicit_ids, message
):
    with pytest.raises(ValueError, match=message):
        table._derive_selected_ids(
            [FakeSample(value) for value in samples], expected_count, 42, explicit_ids
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("benchmark", "omni3d"),
        ("question_filters", ["count"]),
        ("engine", "metric"),
        ("concurrency", 4),
        ("selected_ids", ["b", "a"]),
        ("seed", 7),
    ],
)
def test_validate_atlas_manifest_rejects_run_misattribution(tmp_path, field, replacement):
    systems = _valid_systems(tmp_path)
    spec = systems[0]
    manifest_path = spec.run_dir / table.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=f"run manifest {field}"):
        table._validate_atlas_manifest(
            spec,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_ids=["a", "b"],
        )


@pytest.mark.parametrize("system_index", [0, 1])
@pytest.mark.parametrize("invalid_value", [True, None])
def test_validate_atlas_manifest_requires_disabled_thinking(tmp_path, system_index, invalid_value):
    spec = _valid_systems(tmp_path)[system_index]
    manifest_path = spec.run_dir / table.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if invalid_value is None:
        manifest.pop("llm_enable_thinking")
    else:
        manifest["llm_enable_thinking"] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run manifest llm_enable_thinking"):
        table._validate_atlas_manifest(
            spec,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_ids=["a", "b"],
        )


@pytest.mark.parametrize("model_tiers", [None, {}, {"fast": "openai/wrong-model"}])
def test_validate_atlas_manifest_requires_matching_model_tiers(tmp_path, model_tiers):
    systems = _valid_systems(tmp_path)
    spec = systems[0]
    manifest_path = spec.run_dir / table.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_tiers"] = model_tiers
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected = "no model_tiers" if not model_tiers else "do not match the backbone"
    with pytest.raises(ValueError, match=expected):
        table._validate_atlas_manifest(
            spec,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_ids=["a", "b"],
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("benchmark", "omni3d", "config benchmark"),
        ("question_type", ["count"], "config question_type"),
        ("llm_model", "wrong-model", "config llm_model"),
        ("limit", 1, "config limit"),
    ],
)
def test_validate_native_config_rejects_config_misattribution(
    tmp_path, field, replacement, message
):
    systems = _valid_systems(tmp_path)
    spec = systems[2]
    config_path = spec.run_dir / table.CONFIG_FILENAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = replacement
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _, rows = table._validated_rows(spec, 2)
    with pytest.raises(ValueError, match=message):
        table._validate_native_config(
            spec,
            rows,
            FakeBenchmark().data,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_count=2,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("benchmark", "omni3d", "native run manifest benchmark"),
        ("question_filters", ["count"], "native run manifest question_filters"),
        ("concurrency", 4, "native run manifest concurrency"),
        ("endpoint_model", "wrong-model", "native run manifest endpoint_model"),
    ],
)
def test_validate_native_manifest_rejects_misattribution(tmp_path, field, replacement, message):
    spec = _valid_systems(tmp_path)[2]
    manifest_path = spec.run_dir / table.NATIVE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, rows = table._validated_rows(spec, 2)
    with pytest.raises(ValueError, match=message):
        table._validate_native_config(
            spec,
            rows,
            FakeBenchmark().data,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_count=2,
        )


def test_validate_native_config_rejects_endpoint_misattribution_without_leaking_url(tmp_path):
    systems = _valid_systems(tmp_path)
    spec = systems[2]
    _, rows = table._validated_rows(spec, 2)

    without_endpoint = replace(spec, endpoint_manifest=None)
    with pytest.raises(ValueError, match="requires a native endpoint manifest"):
        table._validate_native_config(
            without_endpoint,
            rows,
            FakeBenchmark().data,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_count=2,
        )

    endpoint_path = spec.endpoint_manifest
    assert endpoint_path is not None
    endpoint_path.write_text(
        json.dumps({"model": "wrong-model", "url": "http://secret-node:9999/v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="endpoint manifest model"):
        table._validate_native_config(
            spec,
            rows,
            FakeBenchmark().data,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_count=2,
        )

    endpoint_path.write_text(
        json.dumps({"model": "qwen3.6-27b", "url": "http://secret-node:9999/v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="llm_base_url") as error:
        table._validate_native_config(
            spec,
            rows,
            FakeBenchmark().data,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_count=2,
        )
    assert "secret-node" not in str(error.value)


def test_validate_native_config_rejects_non_seed_selection_and_accepts_explicit_ids(tmp_path):
    systems = _valid_systems(tmp_path)
    spec = systems[2]
    _, rows = table._validated_rows(spec, 2)
    config_path = spec.run_dir / table.CONFIG_FILENAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sample_ids"] = ["a", "b"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert table._validate_native_config(
        spec,
        rows,
        FakeBenchmark().data,
        benchmark_name="warehouse",
        question_types=["distance"],
        expected_count=2,
    ) == ["a", "b"]

    with pytest.raises(ValueError, match="prediction IDs do not match"):
        table._validate_native_config(
            spec,
            {"a": rows["a"]},
            FakeBenchmark().data,
            benchmark_name="warehouse",
            question_types=["distance"],
            expected_count=2,
        )


def test_select_benchmark_samples_rejects_duplicate_missing_and_wrong_count():
    duplicate = FakeBenchmark(("a", "a"))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        table._select_benchmark_samples(duplicate, {"a"}, 1)

    missing = FakeBenchmark(("a",))
    with pytest.raises(ValueError, match="absent from benchmark"):
        table._select_benchmark_samples(missing, {"b"}, 1)

    wrong_count = FakeBenchmark(("a", "b"))
    with pytest.raises(ValueError, match="selected benchmark has 1 samples"):
        table._select_benchmark_samples(wrong_count, {"a"}, 2)


def test_build_result_tables_rejects_system_metadata_and_id_set_drift(tmp_path):
    systems = _valid_systems(tmp_path)
    with pytest.raises(ValueError, match="expected_count must be positive"):
        table.build_result_tables(
            FakeBenchmark(), systems, benchmark_name="x", expected_count=0, output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="exactly three"):
        table.build_result_tables(
            FakeBenchmark(), systems[:2], benchmark_name="x", expected_count=2, output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="keys must be unique"):
        table.build_result_tables(
            FakeBenchmark(),
            [systems[0], systems[0], systems[2]],
            benchmark_name="x",
            expected_count=2,
            output_dir=tmp_path,
        )
    blank = table.SystemSpec("metric", "Metric", "atlas", systems[1].run_dir, " ")
    with pytest.raises(ValueError, match="explicit grounding"):
        table.build_result_tables(
            FakeBenchmark(),
            [systems[0], blank, systems[2]],
            benchmark_name="x",
            expected_count=2,
            output_dir=tmp_path,
        )
    for changed, message in [
        (replace(systems[0], backbone=""), "explicit backbone"),
        (replace(systems[0], question_slice=""), "explicit question-slice"),
        (replace(systems[0], backbone="Other"), "same backbone"),
        (replace(systems[0], question_slice="Other"), "same question slice"),
        (replace(systems[0], selection_seed=7), "same selection seed"),
    ]:
        with pytest.raises(ValueError, match=message):
            table.build_result_tables(
                FakeBenchmark(),
                [changed, systems[1], systems[2]],
                benchmark_name="warehouse",
                expected_count=2,
                output_dir=tmp_path,
                question_types=["distance"],
            )

    native_rows = [
        _native_row("a", "0.1", prompt=1, completion=1, calls=1, latency=1),
        _native_row("c", "0.3", prompt=1, completion=1, calls=1, latency=1),
    ]
    _write_run(systems[2].run_dir, native_rows, _official_summary(["0.1", "0.3"]))
    with pytest.raises(ValueError, match="sample ID set differs"):
        table.build_result_tables(
            FakeBenchmark(("a", "b", "c", "d")),
            systems,
            benchmark_name="x",
            expected_count=2,
            output_dir=tmp_path,
        )


def test_build_result_tables_rejects_ids_not_derived_from_seed(tmp_path):
    systems = _valid_systems(tmp_path)
    with pytest.raises(ValueError, match="benchmark-derived selection"):
        table.build_result_tables(
            FakeBenchmark(("a", "b", "c", "d")),
            systems,
            benchmark_name="warehouse",
            expected_count=2,
            output_dir=tmp_path,
            question_types=["distance"],
        )


def test_markdown_escapes_pipe_labels_while_csv_preserves_values(tmp_path):
    systems = _valid_systems(tmp_path)
    systems[0] = replace(systems[0], label="Atlas | scenegraph", grounding="language | scene graph")
    markdown_path, csv_path, _gate_path, _results = table.build_result_tables(
        FakeBenchmark(),
        systems,
        benchmark_name="warehouse",
        expected_count=2,
        output_dir=tmp_path / "out",
        question_types=["distance"],
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Atlas \\| scenegraph" in markdown
    assert "language \\| scene graph" in markdown
    with csv_path.open(newline="", encoding="utf-8") as stream:
        first = next(csv.DictReader(stream))
    assert first["system"] == "Atlas | scenegraph"
    assert first["grounding"] == "language | scene graph"


def test_evaluate_system_rejects_nonobject_evaluator_and_summary_drift(tmp_path):
    systems = _valid_systems(tmp_path)
    _, rows = table._validated_rows(systems[0], 2)
    prediction_path = systems[0].run_dir / table.PREDICTIONS_FILENAME

    benchmark = FakeBenchmark()
    benchmark.evaluate = lambda predictions, output_dir=None: []
    with pytest.raises(ValueError, match="official evaluator returned a non-object"):
        table._evaluate_system(
            benchmark,
            systems[0],
            rows,
            2,
            prediction_path,
            table._validate_full_accounting(systems[0], expected_count=2, expected_ids=["a", "b"]),
        )

    bad_total = _official_summary(["0.2", "0.4"])
    bad_total["total_samples"] = 3
    (systems[0].run_dir / table.SUMMARY_FILENAME).write_text(
        json.dumps(bad_total), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="summary total_samples=3"):
        table._evaluate_system(
            FakeBenchmark(),
            systems[0],
            rows,
            2,
            prediction_path,
            table._validate_full_accounting(systems[0], expected_count=2, expected_ids=["a", "b"]),
        )

    nested_drift = _official_summary(["0.2", "0.4"])
    nested_drift["per_type"]["distance"]["scores"] = [0.2, 0.5]
    (systems[0].run_dir / table.SUMMARY_FILENAME).write_text(
        json.dumps(nested_drift), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"results\.per_type\.distance\.scores\[1\]"):
        table._evaluate_system(
            FakeBenchmark(),
            systems[0],
            rows,
            2,
            prediction_path,
            table._validate_full_accounting(systems[0], expected_count=2, expected_ids=["a", "b"]),
        )

    error_drift = _official_summary(["0.2", "0.4"])
    error_drift["per_category"]["distance"]["mean_error_rate"] = 9.0
    (systems[0].run_dir / table.SUMMARY_FILENAME).write_text(
        json.dumps(error_drift), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"per_category\.distance\.mean_error_rate"):
        table._evaluate_system(
            FakeBenchmark(),
            systems[0],
            rows,
            2,
            prediction_path,
            table._validate_full_accounting(systems[0], expected_count=2, expected_ids=["a", "b"]),
        )


def test_distance_mean_relative_error_is_recomputed_from_prediction_gt_rows(tmp_path):
    systems = _valid_systems(tmp_path)
    _, rows = table._validated_rows(systems[0], 2)
    prediction_path = systems[0].run_dir / table.PREDICTIONS_FILENAME
    benchmark = FakeBenchmark()
    original_evaluate = benchmark.evaluate

    def drifted_evaluate(predictions, output_dir=None):
        result = original_evaluate(predictions, output_dir=output_dir)
        result["per_category"]["distance"]["mean_error_rate"] += 0.1
        return result

    benchmark.evaluate = drifted_evaluate
    with pytest.raises(ValueError, match="does not match recomputed prediction/GT rows"):
        table._evaluate_system(
            benchmark,
            systems[0],
            rows,
            2,
            prediction_path,
            table._validate_full_accounting(systems[0], expected_count=2, expected_ids=["a", "b"]),
        )


def test_parse_args_normalizes_question_types_and_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALCLAW_ROOT", str(tmp_path / "claw"))
    args = table.parse_args(
        [
            "--benchmark",
            "warehouse",
            "--data-root",
            str(tmp_path / "data"),
            "--expected-count",
            "2",
            "--atlas-concurrency",
            "4",
            "--native-concurrency",
            "2",
            "--question-type",
            "distance,count",
            "--question-type",
            "mcq",
            "--scenegraph-dir",
            "s",
            "--metric-dir",
            "m",
            "--native-dir",
            "n",
            "--scenegraph-accounting",
            "sg-accounting.json",
            "--metric-accounting",
            "metric-accounting.json",
            "--native-accounting",
            "native-accounting.json",
            "--scenegraph-grounding",
            "scene",
            "--metric-grounding",
            "metric",
            "--native-grounding",
            "native",
            "--backbone",
            "Qwen3.6-27B",
            "--question-slice",
            "Warehouse validation distance",
            "--native-endpoint-manifest",
            str(tmp_path / "endpoint.json"),
            "--output-dir",
            "out",
        ]
    )
    assert args.question_types == ["distance", "count", "mcq"]
    assert args.spatialclaw_root == str(tmp_path / "claw")
    assert args.backbone == "Qwen3.6-27B"
    assert args.question_slice == "Warehouse validation distance"
    assert args.selection_seed == 42
    assert args.atlas_concurrency == 4
    assert args.native_concurrency == 2
    assert args.native_endpoint_manifest == str(tmp_path / "endpoint.json")


def test_parse_args_requires_positive_count_and_spatialclaw_root(monkeypatch, tmp_path):
    monkeypatch.delenv("SPATIALCLAW_ROOT", raising=False)
    common = [
        "--benchmark",
        "warehouse",
        "--data-root",
        "data",
        "--expected-count",
        "0",
        "--atlas-concurrency",
        "4",
        "--native-concurrency",
        "1",
        "--scenegraph-dir",
        "s",
        "--metric-dir",
        "m",
        "--native-dir",
        "n",
        "--scenegraph-accounting",
        "sg-accounting.json",
        "--metric-accounting",
        "metric-accounting.json",
        "--native-accounting",
        "native-accounting.json",
        "--scenegraph-grounding",
        "s",
        "--metric-grounding",
        "m",
        "--native-grounding",
        "n",
        "--backbone",
        "Qwen3.6-27B",
        "--question-slice",
        "Warehouse validation distance",
        "--native-endpoint-manifest",
        "endpoint.json",
        "--output-dir",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        table.parse_args(common)
    common[common.index("0")] = "1"
    with pytest.raises(SystemExit):
        table.parse_args(common)


def test_load_benchmark_validates_checkout_and_factory_result(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="factory not found"):
        table._load_benchmark(tmp_path, "warehouse", tmp_path, [])

    factory_path = tmp_path / "spatial_agent" / "evals" / "factory.py"
    factory_path.parent.mkdir(parents=True)
    factory_path.write_text("# fixture\n", encoding="utf-8")

    calls = []

    class Factory:
        @staticmethod
        def create_benchmark(name, **kwargs):
            calls.append((name, kwargs))
            return FakeBenchmark() if name == "warehouse" else None

    monkeypatch.setattr(
        table.importlib,
        "import_module",
        lambda name: SimpleNamespace(BenchmarkFactory=Factory),
    )
    sys.path.insert(0, str(tmp_path.resolve()))
    benchmark = table._load_benchmark(tmp_path, "warehouse", tmp_path / "data", ["distance"])
    assert isinstance(benchmark, FakeBenchmark)
    assert calls[0][1]["question_type"] == ["distance"]
    assert sys.path.count(str(tmp_path.resolve())) == 1
    with pytest.raises(ValueError, match="resolved to None"):
        table._load_benchmark(tmp_path, "none", tmp_path / "data", [])


def test_main_success_and_failure(monkeypatch, tmp_path, capsys):
    systems = _valid_systems(tmp_path)
    monkeypatch.setattr(table, "_load_benchmark", lambda *args: FakeBenchmark())
    argv = [
        "--benchmark",
        "warehouse",
        "--data-root",
        "data",
        "--expected-count",
        "2",
        "--atlas-concurrency",
        "1",
        "--native-concurrency",
        "1",
        "--question-type",
        "distance",
        "--scenegraph-dir",
        str(systems[0].run_dir),
        "--metric-dir",
        str(systems[1].run_dir),
        "--native-dir",
        str(systems[2].run_dir),
        "--scenegraph-accounting",
        str(systems[0].accounting_path),
        "--metric-accounting",
        str(systems[1].accounting_path),
        "--native-accounting",
        str(systems[2].accounting_path),
        "--scenegraph-grounding",
        "language scene graph",
        "--metric-grounding",
        "rle+da3",
        "--native-grounding",
        "sam3+da3 code action",
        "--backbone",
        "Qwen3.6-27B",
        "--question-slice",
        "Warehouse validation distance",
        "--native-endpoint-manifest",
        str(systems[2].endpoint_manifest),
        "--output-dir",
        str(tmp_path / "out"),
        "--spatialclaw-root",
        str(tmp_path / "claw"),
    ]
    assert table.main(argv) == 0
    assert "[ok] Markdown:" in capsys.readouterr().out

    monkeypatch.setattr(
        table,
        "run_from_args",
        lambda args: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    assert table.main(argv) == 1
    assert "[fatal] RuntimeError: broken" in capsys.readouterr().err


def test_run_from_args_builds_fixed_three_system_contract(monkeypatch, tmp_path):
    captured = {}
    args = argparse.Namespace(
        spatialclaw_root="claw",
        benchmark="warehouse",
        data_root="data",
        question_types=["distance"],
        scenegraph_dir="scene",
        metric_dir="metric",
        native_dir="native",
        scenegraph_accounting="scene-accounting.json",
        metric_accounting="metric-accounting.json",
        native_accounting="native-accounting.json",
        scenegraph_grounding="scene label",
        metric_grounding="metric label",
        native_grounding="native label",
        backbone="Qwen3.6-27B",
        question_slice="Warehouse validation distance",
        selection_seed=42,
        native_endpoint_manifest="endpoint.json",
        expected_count=2,
        atlas_concurrency=1,
        native_concurrency=1,
        output_dir=str(tmp_path),
    )
    benchmark = FakeBenchmark()
    monkeypatch.setattr(table, "_load_benchmark", lambda *values: benchmark)

    def fake_build(received_benchmark, systems, **kwargs):
        captured.update(benchmark=received_benchmark, systems=systems, kwargs=kwargs)
        return Path("m"), Path("c"), Path("g"), []

    monkeypatch.setattr(table, "build_result_tables", fake_build)
    assert table.run_from_args(args) == (Path("m"), Path("c"), Path("g"), [])
    assert captured["benchmark"] is benchmark
    assert [spec.row_format for spec in captured["systems"]] == ["atlas", "atlas", "native"]
    assert [spec.grounding for spec in captured["systems"]] == [
        "scene label",
        "metric label",
        "native label",
    ]
    assert [spec.backbone for spec in captured["systems"]] == ["Qwen3.6-27B"] * 3
    assert [spec.question_slice for spec in captured["systems"]] == [
        "Warehouse validation distance"
    ] * 3
    assert [spec.engine for spec in captured["systems"]] == ["scenegraph", "metric", None]
    assert captured["systems"][2].endpoint_manifest == Path("endpoint.json")
    assert [spec.accounting_path for spec in captured["systems"]] == [
        Path("scene-accounting.json"),
        Path("metric-accounting.json"),
        Path("native-accounting.json"),
    ]
    assert captured["kwargs"]["question_types"] == ["distance"]
    assert captured["kwargs"]["atlas_concurrency"] == 1
