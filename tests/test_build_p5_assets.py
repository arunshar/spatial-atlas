"""Adversarial tests for the fail-closed P5 publication asset builder."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

import build_p5_assets as builder


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _dummy(root: Path, name: str) -> Path:
    return _write_json(root / f"{name}.json", {"name": name})


def _journal_rows(arm: str, prediction: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(builder.EXPECTED_IDS):
        if arm == "native":
            usage: dict[str, object] = {
                "num_calls": 2,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 2,
                "total_reasoning_tokens": 1,
                "max_prompt_tokens": 5,
                "max_completion_tokens": 1,
            }
        else:
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "num_calls": 2,
                "estimated_cost_usd": 0.0,
            }
        rows.append(
            {
                "sample_id": sample_id,
                "cluster_id": f"c{index % builder.QSPATIAL_CLUSTERS}",
                "prediction_text": prediction,
                "latency_seconds": 10.0,
                "usage": usage,
            }
        )
    return rows


def _operational(arm: str) -> dict[str, object]:
    return builder._operational_from_journal(_journal_rows(arm, "1 meter"), arm)


def _comparison(estimate: float) -> dict[str, object]:
    return {
        "estimate": estimate,
        "confidence": builder.BOOTSTRAP_CONFIDENCE,
        "iterations": builder.BOOTSTRAP_ITERATIONS,
        "seed": builder.BOOTSTRAP_SEED,
        "quantile_method": "linear",
        "ci": {"lower": estimate, "upper": estimate},
    }


def _arm_result(
    arm: str,
    *,
    sr125: float,
    sr2: float,
    log_loss: float,
) -> tuple[dict[str, object], dict[str, object]]:
    rows = [
        {
            "sample_id": sample_id,
            "cluster_id": f"c{index % builder.QSPATIAL_CLUSTERS}",
            "prediction_text": "1 meter",
            "target": "100 centimeter",
            "sr_at_1_25": sr125,
            "sr_at_2": sr2,
            "capped_absolute_log_ratio": log_loss,
            "relative_error": 0.1,
        }
        for index, sample_id in enumerate(builder.EXPECTED_IDS)
    ]
    cluster_values = {
        "sr_at_1_25": {f"c{index}": sr125 for index in range(builder.QSPATIAL_CLUSTERS)},
        "sr_at_2": {f"c{index}": sr2 for index in range(builder.QSPATIAL_CLUSTERS)},
        "capped_absolute_log_ratio": {
            f"c{index}": log_loss for index in range(builder.QSPATIAL_CLUSTERS)
        },
    }
    row_metrics = {
        "sr_at_1_25": sr125,
        "sr_at_2": sr2,
        "mean_capped_absolute_log_ratio": log_loss,
        "mre": 0.1,
        "valid_prediction_mre": 0.1,
        "median_absolute_log_ratio": log_loss,
        "valid_prediction_median_absolute_log_ratio": log_loss,
        "valid_predictions": builder.QSPATIAL_ROWS,
        "grounding_failures": 0,
        "grounding_failure_rate": 0.0,
        "failure_reasons": {},
        "ground_truth_unit_counts": {"centimeter": builder.QSPATIAL_ROWS},
        "prediction_unit_counts": {"meter": builder.QSPATIAL_ROWS},
    }
    cluster_metrics = {
        "cluster_count": builder.QSPATIAL_CLUSTERS,
        "sr_at_1_25": sr125,
        "sr_at_2": sr2,
        "mean_capped_absolute_log_ratio": log_loss,
    }
    audit = {
        "total_samples": builder.QSPATIAL_ROWS,
        "correct_samples": int(sr125 * builder.QSPATIAL_ROWS),
        "overall_accuracy": sr125,
        "valid_predictions": builder.QSPATIAL_ROWS,
        "cluster_count": builder.QSPATIAL_CLUSTERS,
        "failure_reasons": {},
        "ground_truth_unit_counts": {"centimeter": builder.QSPATIAL_ROWS},
        "prediction_unit_counts": {"meter": builder.QSPATIAL_ROWS},
        "cluster_bootstrap_sr_at_1_25": None,
        "cluster_bootstrap_sr_at_2": None,
    }
    project_metrics = {
        "total_samples": builder.QSPATIAL_ROWS,
        "overall_accuracy": sr125,
        "sr_at_1_25": sr125,
        "sr_at_2": sr2,
        "cluster_balanced_sr_at_1_25": sr125,
        "cluster_balanced_sr_at_2": sr2,
        "valid_predictions": builder.QSPATIAL_ROWS,
        "cluster_count": builder.QSPATIAL_CLUSTERS,
        "grounding_failures": 0,
        "grounding_failure_rate": 0.0,
        "mre": 0.1,
        "valid_prediction_mre": 0.1,
        "median_absolute_log_ratio": log_loss,
        "valid_prediction_median_absolute_log_ratio": log_loss,
        "failure_reasons": {},
        "ground_truth_unit_counts": {"centimeter": builder.QSPATIAL_ROWS},
        "prediction_unit_counts": {"meter": builder.QSPATIAL_ROWS},
    }
    result = {
        "rows": rows,
        "row_metrics": row_metrics,
        "cluster_metrics": cluster_metrics,
        "project_scorer_audit": audit,
        "cluster_values": cluster_values,
        "preregistered_project_metrics": project_metrics,
        "preregistered_project_scorer_crosscheck": {
            "status": "pass",
            "fields": {name: True for name in builder.QSPATIAL_CROSSCHECK_FIELDS},
        },
    }
    return result, project_metrics


def _qspatial_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, supported: bool = False
) -> Path:
    root = tmp_path / "private" / "scratch.global" / "arunshar"
    root.mkdir(parents=True, exist_ok=True)
    expected_ids = root / "expected_ids.json"
    expected_ids.write_bytes(builder._canonical_id_bytes(builder.EXPECTED_IDS))
    assert hashlib.sha256(expected_ids.read_bytes()).hexdigest() == builder.EXPECTED_IDS_SHA256
    expected_ids_record = _artifact(expected_ids)
    readiness = _write_json(
        root / "readiness.json",
        {"schema_version": 1, "kind": "qspatial_operational_pilot_readiness", "status": "pass"},
    )
    readiness_record = _artifact(readiness)
    provenance = {
        name: _artifact(_dummy(root, f"provenance_{name}"))
        for name in builder.EXECUTION_PROVENANCE_NAMES
    }
    runtime = {
        name: _artifact(_dummy(root, f"runtime_{name}")) for name in builder.SCORING_RUNTIME_NAMES
    }
    monkeypatch.setattr(
        builder,
        "QSPATIAL_RUNTIME_SHA256",
        {name: str(record["sha256"]) for name, record in runtime.items()},
    )
    shared = {
        "protocol_manifest": _artifact(_dummy(root, "protocol_manifest")),
        "atlas_policy": _artifact(_dummy(root, "atlas_policy")),
        "image_manifest": _artifact(_dummy(root, "image_manifest")),
        "protocol_records": _artifact(_dummy(root, "protocol_records")),
        "inference_manifest": _artifact(_dummy(root, "inference_manifest")),
    }
    gate_records: dict[str, object] = {}
    journal_records: dict[str, object] = {}
    journal_seal_records: dict[str, object] = {}
    operational = {arm: _operational(arm) for arm in builder.ARM_ORDER}
    previous_gate: object | None = None
    for arm in builder.ARM_ORDER:
        journal = _write_jsonl(root / f"{arm}_journal.jsonl", _journal_rows(arm, "1 meter"))
        journal_record = _artifact(journal)
        journal_records[arm] = journal_record
        journal_seal = _write_json(
            root / f"{arm}_seal.json",
            {
                "schema_version": 1,
                "kind": "qspatial_label_free_prescore_journal_seal",
                "status": "pass",
                "expected_count": builder.QSPATIAL_ROWS,
                "expected_ids": list(builder.EXPECTED_IDS),
                "expected_ids_sha256": builder.EXPECTED_IDS_SHA256,
                "journal_artifact": journal_record,
                "expected_ids_artifact": expected_ids_record,
            },
        )
        journal_seal_record = _artifact(journal_seal)
        journal_seal_records[arm] = journal_seal_record
        gate_artifacts: dict[str, object] = {
            "journal": journal_record,
            "raw_journal": _artifact(_dummy(root, f"{arm}_raw_journal")),
            "seal": journal_seal_record,
            "expected_ids": expected_ids_record,
            "full_accounting": _artifact(_dummy(root, f"{arm}_full_accounting")),
            "endpoint_manifest": _artifact(_dummy(root, f"{arm}_endpoint")),
            **runtime,
            **shared,
            "pilot_readiness": readiness_record,
            "run_manifest": _artifact(_dummy(root, f"{arm}_run_manifest")),
        }
        if previous_gate is not None:
            gate_artifacts["prior_gate"] = previous_gate
        if arm == "metric_shuffled":
            gate_artifacts["control_mapping"] = _artifact(_dummy(root, "control_mapping"))
        if builder.ARM_SPECS[arm][2]:
            gate_artifacts["gpu_liveness_final"] = _artifact(_dummy(root, f"{arm}_gpu"))
        gate = {name: None for name in builder.QSPATIAL_ARM_GATE_FIELDS}
        engine, image_mode, _requires_gpu = builder.ARM_SPECS[arm]
        gate.update(
            {
                "schema_version": 1,
                "kind": "qspatial_full_arm_gate",
                "status": "pass",
                "arm": arm,
                "engine": engine,
                "image_mode": image_mode,
                "expected_count": builder.QSPATIAL_ROWS,
                "slice_sha256": builder.SLICE_SHA256,
                "parser_sha256": builder.PARSER_SHA256,
                "geometry_sha256": builder.GEOMETRY_SHA256,
                "scoring_sha256": builder.SCORING_SHA256,
                "concurrency": 4,
                "control_mapping_sha256": builder.CONTROL_MAPPING_SHA256
                if arm == "metric_shuffled"
                else None,
                "hidden_fallback_count": 0,
                "operational": operational[arm],
                "execution_provenance": provenance,
                "artifacts": gate_artifacts,
            }
        )
        gate_path = _write_json(root / f"{arm}_gate.json", gate)
        previous_gate = _artifact(gate_path)
        gate_records[arm] = previous_gate
    seal = _write_json(
        root / "all_arms_sealed.json",
        {
            "schema_version": 1,
            "kind": "qspatial_full_all_arms_sealed",
            "status": "pass",
            "scoring_authorized": True,
            "labels_loaded": False,
            "arm_order": builder.ARM_ORDER,
            "expected_count_per_arm": builder.QSPATIAL_ROWS,
            "total_sealed_rows": builder.QSPATIAL_ROWS * len(builder.ARM_ORDER),
            "slice_sha256": builder.SLICE_SHA256,
            "parser_sha256": builder.PARSER_SHA256,
            "geometry_sha256": builder.GEOMETRY_SHA256,
            "scoring_sha256": builder.SCORING_SHA256,
            "arm_final_vllm_job_ids": {
                arm: str(index + 100) for index, arm in enumerate(builder.ARM_ORDER)
            },
            "all_attempt_vllm_job_ids": [
                str(index + 100) for index in range(len(builder.ARM_ORDER))
            ],
            "operational_by_arm": operational,
            "execution_provenance": provenance,
            "artifacts": {
                "pilot_readiness": readiness_record,
                "expected_ids": expected_ids_record,
                "arm_gates": gate_records,
                "journal_seals": journal_seal_records,
                "journals": journal_records,
                "scoring_runtime": runtime,
            },
        },
    )
    arm_values = {
        "scenegraph": (0.0, 0.0, 0.5),
        "question_only": (0.0, 0.0, 0.5),
        "metric_correct": (1.0, 1.0, 0.2) if supported else (0.0, 0.0, 0.5),
        "metric_shuffled": (0.0, 0.0, 0.5),
        "native": (0.0, 0.0, 0.5),
    }
    arm_results: dict[str, object] = {}
    scorer_artifacts: dict[str, object] = {}
    for arm, (sr125, sr2, log_loss) in arm_values.items():
        arm_result, project_result = _arm_result(arm, sr125=sr125, sr2=sr2, log_loss=log_loss)
        arm_results[arm] = arm_result
        scorer_artifacts[arm] = _artifact(
            _write_json(root / f"{arm}_project_score.json", project_result)
        )
    differences = {
        "metric_correct_minus_question_only": "question_only",
        "metric_correct_minus_metric_shuffled": "metric_shuffled",
        "metric_correct_minus_scenegraph": "scenegraph",
        "metric_correct_minus_native": "native",
    }
    comparisons: dict[str, object] = {}
    left = arm_values["metric_correct"]
    for name, right_name in differences.items():
        right = arm_values[right_name]
        comparisons[name] = {
            "sr_at_1_25": _comparison(left[0] - right[0]),
            "sr_at_2": _comparison(left[1] - right[1]),
            "capped_absolute_log_ratio": _comparison(left[2] - right[2]),
        }
    score = {
        "schema_version": 1,
        "kind": "qspatial_full_offline_scoring",
        "status": "pass",
        "all_journals_sealed_before_labels_loaded": True,
        "all_arms_seal": _artifact(seal),
        "arm_order": builder.ARM_ORDER,
        "expected_count_per_arm": builder.QSPATIAL_ROWS,
        "cluster_count": builder.QSPATIAL_CLUSTERS,
        "slice_sha256": builder.SLICE_SHA256,
        "scoring_sha256": builder.SCORING_SHA256,
        "execution_provenance": provenance,
        "bootstrap_contract": {
            "unit": "question_cluster",
            "paired": True,
            "cluster_order": "sorted_full_cluster_hash",
            "rng": "numpy.random.default_rng",
            "seed": builder.BOOTSTRAP_SEED,
            "iterations": builder.BOOTSTRAP_ITERATIONS,
            "confidence": builder.BOOTSTRAP_CONFIDENCE,
            "quantile_method": "linear",
        },
        "invalid_prediction_contract": {
            "sr_at_1_25": 0.0,
            "sr_at_2": 0.0,
            "capped_absolute_log_ratio": math.log(10.0),
        },
        "metric_provenance": {
            "project_scoring_contract_sha256": builder.SCORING_SHA256,
            "sealed_scoring_runtime_artifacts": runtime,
            "whole_scorer_is_not_claimed_as_upstream_official": True,
        },
        "arm_results": arm_results,
        "hidden_fallback_counts": {arm: 0 for arm in builder.ARM_ORDER},
        "operational_by_arm": operational,
        "paired_cluster_comparisons": comparisons,
        "claim_gates": {
            "visual_claim_supported": supported,
            "visual_rule": builder.VISUAL_RULE,
            "decisive_win_supported": supported,
            "decisive_rule": builder.DECISIVE_RULE,
            "negative_result_required_if_false": True,
        },
        "preregistered_project_scorer_artifacts": scorer_artifacts,
    }
    return _write_json(root / "qspatial_score.json", score)


def _vsi_gate(tmp_path: Path, qspatial: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "private" / "scratch.global" / "arunshar" / "vsi"
    ids = [f"vsi-{index:04d}" for index in range(builder.VSI_ROWS)]
    ids_hash = hashlib.sha256(builder._canonical_id_bytes(ids)).hexdigest()
    monkeypatch.setattr(builder, "VSI_IDS_SHA256", ids_hash)
    p4_smoke_gate = _artifact(_dummy(root, "p4_smoke_gate"))
    protocol = _write_json(
        root / "protocol.json",
        {
            "schema_version": 1,
            "kind": "p4_vsibench_unbiased_full_protocol",
            "benchmark": "vsibench_unbiased",
            "expected_count": builder.VSI_ROWS,
            "expected_unique_ids": builder.VSI_ROWS,
            "ids_sha256": ids_hash,
            "source_parquet_sha256": builder.VSI_PARQUET_SHA256,
            "concurrency": 1,
            "maximum_chunk_size": 50,
            "video_max_fps": 1.0,
            "question_type_counts": builder.VSI_QUESTION_COUNTS,
            "resume_policy": "accounted_interruption_only",
            "final_scoring": "official_spatialclaw_vsibench_evaluator",
        },
    )
    ids_path = _write_json(
        root / "ids.json",
        {
            "schema_version": 1,
            "kind": "p4_vsibench_unbiased_ids",
            "benchmark": "vsibench_unbiased",
            "count": builder.VSI_ROWS,
            "unique_count": builder.VSI_ROWS,
            "ids_sha256": ids_hash,
            "source_parquet_sha256": builder.VSI_PARQUET_SHA256,
            "ids": ids,
        },
    )
    predictions = _write_jsonl(
        root / "predictions.jsonl",
        [
            {
                "sample_id": sample_id,
                "content": "A",
                "error": None,
                "usage": {"num_calls": 2},
            }
            for sample_id in ids
        ],
    )
    prediction_record = _artifact(predictions)
    chunks = [ids[index : index + 50] for index in range(0, len(ids), 50)]
    statuses = ["accounted_interrupted"] * (len(chunks) - 1) + ["pass"]
    terminal_totals = {
        "stop": builder.VSI_ROWS * 2,
        "length": 0,
        "abort": 0,
        "error": 0,
        "repetition": 0,
    }
    accounting = _write_json(
        root / "accounting.json",
        {
            "schema_version": 1,
            "kind": "p3_full_vllm_accounting",
            "status": "pass",
            "benchmark": "vsibench_unbiased",
            "row_format": "native",
            "expected_count": builder.VSI_ROWS,
            "concurrency": 1,
            "requires_gpu": True,
            "completed_ids": ids,
            "predictions": prediction_record,
            "journaled_call_total": builder.VSI_ROWS * 2,
            "attempt_statuses": statuses,
            "terminal_counter_totals": terminal_totals,
        },
    )
    attempt_records: list[dict[str, object]] = []
    for index, (selected, status) in enumerate(zip(chunks, statuses, strict=True), 1):
        service = {
            "endpoint_model": "qwen3.6-27b",
            "endpoint_reasoning_parser": "qwen3",
            "vllm_job_id": str(10_000 + index),
            "gpu_tool_job_id": str(20_000 + index),
            "endpoint_url": f"http://127.0.0.1:{30_000 + index}",
            "metrics_url": f"http://127.0.0.1:{30_000 + index}/metrics",
            "endpoint_manifest": _artifact(_dummy(root, f"attempt_{index:02d}_endpoint_manifest")),
        }
        selection = _write_json(
            root / f"attempt_{index:02d}_selection.json",
            {
                "schema_version": 1,
                "kind": "p4_vsibench_unbiased_attempt_selection",
                "selected_count": len(selected),
                "selected_ids": selected,
                "final_attempt": index == len(chunks),
                "expected_total": builder.VSI_ROWS,
                "ids_sha256": ids_hash,
                "service": service,
            },
        )
        nested = {
            "selection": _artifact(selection),
            "p4_smoke_gate": p4_smoke_gate,
            **{
                name: _artifact(_dummy(root, f"attempt_{index:02d}_{name}"))
                for name in builder.VSI_ATTEMPT_ARTIFACT_NAMES - {"selection", "p4_smoke_gate"}
            },
        }
        attempt = _write_json(
            root / f"attempt_{index:02d}.json",
            {
                "schema_version": 1,
                "kind": "p4_vsibench_unbiased_attempt_gate",
                "status": status,
                "final_attempt": index == len(chunks),
                "selected_ids": selected,
                "newly_completed_count": len(selected),
                "terminal_counter_deltas": {
                    "stop": len(selected) * 2,
                    "length": 0,
                    "abort": 0,
                    "error": 0,
                    "repetition": 0,
                },
                "journaled_call_delta": len(selected) * 2,
                "service": service,
                "artifacts": nested,
            },
        )
        attempt_records.append(
            {
                **_artifact(attempt),
                "nested_artifacts": nested,
                "immutable_journal_after": nested["immutable_journal_after"],
            }
        )
    detailed: list[dict[str, object]] = []
    offset = 0
    for question_type, count in builder.VSI_QUESTION_COUNTS.items():
        for sample_id in ids[offset : offset + count]:
            detailed.append(
                {
                    "id": sample_id,
                    "question_type": question_type,
                    "ground_truth": "A",
                    "prediction": "B",
                    "extracted": "B",
                    "score": 0.5,
                }
            )
        offset += count
    per_task = {
        name: {"score": 50.0, "count": count}
        for name, count in builder.VSI_QUESTION_COUNTS.items()
        if not name.startswith("object_rel_direction_")
    }
    per_task["object_rel_direction"] = {
        "score": 50.0,
        "count": sum(
            builder.VSI_QUESTION_COUNTS[name]
            for name in (
                "object_rel_direction_easy",
                "object_rel_direction_medium",
                "object_rel_direction_hard",
            )
        ),
    }
    summary = {
        "total_samples": builder.VSI_ROWS,
        "correct_samples": 0,
        "overall_accuracy": 0.5,
        "overall_accuracy_pct": 50.0,
        "per_task_scores": per_task,
    }
    official = _write_json(root / "official.json", {**summary, "detailed_results": detailed})
    qspatial_claims = json.loads(qspatial.read_text(encoding="utf-8"))["claim_gates"]
    return _write_json(
        root / "vsi_gate.json",
        {
            "schema_version": 1,
            "kind": "p4_vsibench_unbiased_full_gate",
            "status": "pass",
            "validated_at_utc": "2026-07-12T00:00:00+00:00",
            "benchmark": "vsibench_unbiased",
            "expected_count": builder.VSI_ROWS,
            "successful_unique_predictions": builder.VSI_ROWS,
            "attempt_count": len(attempt_records),
            "interrupted_attempt_count": statuses.count("accounted_interrupted"),
            "terminal_counter_totals": terminal_totals,
            "labels_loaded_only_for_final_official_scoring": True,
            "official_summary": summary,
            "p3_claim_gates": qspatial_claims,
            "artifacts": {
                "p3_offline_score": _artifact(qspatial),
                "p4_smoke_gate": p4_smoke_gate,
                "protocol": _artifact(protocol),
                "frozen_ids": _artifact(ids_path),
                "full_accounting": _artifact(accounting),
                "predictions": prediction_record,
                "official_results": _artifact(official),
                "attempts": attempt_records,
            },
        },
    )


def _evidence_397b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "private" / "scratch.global" / "arunshar" / "397b"
    paths = {
        "job_script": _write_json(root / "job.json", {"job": builder.EVIDENCE_397B_JOB_ID}),
        "log": _write_json(root / "log.json", {"state": "COMPLETED"}),
        "config": _write_json(root / "config.json", {"model": builder.EVIDENCE_397B_BACKBONE}),
        "predictions": _write_jsonl(
            root / "predictions.jsonl",
            [{"sample_id": f"ERQA_{index}", "prediction": "answer"} for index in range(5)],
        ),
        "result": _write_json(
            root / "result.json",
            {"total_samples": 5, "correct_samples": 3, "overall_accuracy": 0.6},
        ),
    }
    monkeypatch.setattr(
        builder,
        "EVIDENCE_397B_ARTIFACT_SHA256",
        {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    )
    return _write_json(
        root / "397b_evidence.json",
        {
            "schema_version": 1,
            "kind": "p4_397b_evidence",
            "status": "pass",
            "job_id": builder.EVIDENCE_397B_JOB_ID,
            "slurm_state": "COMPLETED",
            "benchmark": "erqa",
            "scope": "subsample5_diagnostic_transfer",
            "backbone": builder.EVIDENCE_397B_BACKBONE,
            "expected_count": 5,
            "successful_count": 5,
            "correct_count": 3,
            "score": 0.6,
            "claim_bearing": False,
            "artifacts": {name: _artifact(path) for name, path in paths.items()},
        },
    )


def _rewrite_artifact(parent: dict[str, object], key: str, path: Path) -> None:
    parent[key] = _artifact(path)


def _update_vsi_attempt(
    vsi_path: Path,
    index: int,
    updates: dict[str, object],
    *,
    sync_nested: bool = False,
) -> None:
    gate = json.loads(vsi_path.read_text(encoding="utf-8"))
    record = gate["artifacts"]["attempts"][index]
    attempt_path = Path(record["path"])
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.update(updates)
    _write_json(attempt_path, attempt)
    nested = attempt["artifacts"] if sync_nested else record["nested_artifacts"]
    gate["artifacts"]["attempts"][index] = {
        **_artifact(attempt_path),
        "nested_artifacts": nested,
        "immutable_journal_after": record["immutable_journal_after"],
    }
    _write_json(vsi_path, gate)


def _update_vsi_accounting(
    vsi_path: Path,
    updates: dict[str, object],
    *,
    mirror_terminal_totals: bool = False,
) -> None:
    gate = json.loads(vsi_path.read_text(encoding="utf-8"))
    accounting_path = Path(gate["artifacts"]["full_accounting"]["path"])
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting.update(updates)
    _write_json(accounting_path, accounting)
    gate["artifacts"]["full_accounting"] = _artifact(accounting_path)
    if mirror_terminal_totals:
        gate["terminal_counter_totals"] = accounting["terminal_counter_totals"]
    _write_json(vsi_path, gate)


def _vsi_attempt(vsi_path: Path, index: int) -> dict[str, object]:
    gate = json.loads(vsi_path.read_text(encoding="utf-8"))
    record = gate["artifacts"]["attempts"][index]
    return json.loads(Path(record["path"]).read_text(encoding="utf-8"))


def _update_vsi_official(vsi_path: Path, update: Callable[[dict[str, object]], None]) -> None:
    gate = json.loads(vsi_path.read_text(encoding="utf-8"))
    official_path = Path(gate["artifacts"]["official_results"]["path"])
    official = json.loads(official_path.read_text(encoding="utf-8"))
    update(official)
    _write_json(official_path, official)
    gate["artifacts"]["official_results"] = _artifact(official_path)
    _write_json(vsi_path, gate)


def test_low_level_fail_closed_helpers(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(builder.EvidenceError, match="cannot load missing JSON"):
        builder._load_json(missing, "missing JSON")

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]\n", encoding="utf-8")
    with pytest.raises(builder.EvidenceError, match="must be a JSON object"):
        builder._load_json(scalar, "scalar JSON")

    with pytest.raises(builder.EvidenceError, match="artifact is missing"):
        builder._artifact(missing)
    with pytest.raises(builder.EvidenceError, match="artifact is missing"):
        builder._public_artifact(missing, "missing-public")
    with pytest.raises(builder.EvidenceError, match="artifact is missing"):
        builder._verify_artifact(None, "missing")
    with pytest.raises(builder.EvidenceError, match="path must be absolute"):
        builder._verify_artifact(
            {"path": "relative", "sha256": "0" * 64, "size_bytes": 0}, "relative"
        )
    with pytest.raises(builder.EvidenceError, match="metadata is invalid"):
        builder._verify_artifact(
            {"path": str(missing.resolve()), "sha256": "invalid", "size_bytes": -1},
            "invalid",
        )
    with pytest.raises(builder.EvidenceError, match="artifact is missing"):
        builder._verify_artifact(
            {"path": str(missing.resolve()), "sha256": "0" * 64, "size_bytes": 0},
            "absent",
        )
    changed = _write_json(tmp_path / "changed.json", {"before": True})
    record = _artifact(changed)
    _write_json(changed, {"after": True})
    with pytest.raises(builder.EvidenceError, match="artifact changed"):
        builder._verify_artifact(record, "changed")

    with pytest.raises(builder.EvidenceError, match="cannot load missing JSONL"):
        builder._load_jsonl(tmp_path / "missing.jsonl", "missing JSONL")
    for name, payload in (("empty", b""), ("unterminated", b"{}")):
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(payload)
        with pytest.raises(builder.EvidenceError, match="nonempty and newline terminated"):
            builder._load_jsonl(path, name)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_bytes(b"\xff\n")
    with pytest.raises(builder.EvidenceError, match="row 1 is invalid"):
        builder._load_jsonl(invalid, "invalid")
    nonobject = tmp_path / "nonobject.jsonl"
    nonobject.write_text("[]\n", encoding="utf-8")
    with pytest.raises(builder.EvidenceError, match="row 1 is not an object"):
        builder._load_jsonl(nonobject, "nonobject")

    with pytest.raises(builder.EvidenceError, match="integer at least 1"):
        builder._integer(True, "integer", minimum=1)
    with pytest.raises(builder.EvidenceError, match="exceeds 1"):
        builder._probability(1.1, "probability")
    with pytest.raises(builder.EvidenceError, match="must be numeric"):
        builder._finite("1", "finite")
    with pytest.raises(builder.EvidenceError, match="outside its valid range"):
        builder._finite(float("inf"), "finite")
    assert builder._optional_finite(None, "optional") is None
    assert builder._optional_finite(1.0, "optional") == 1.0
    assert builder._same_optional_number(None, None)
    assert not builder._same_optional_number(None, 1.0)
    assert not builder._same_number(True, 1)
    assert not builder._same_number("1", 1)

    valid = _write_json(tmp_path / "valid.json", {"ok": True})
    assert builder._validate_artifact_group({"one": _artifact(valid)}, {"one"}, "group") == {
        "one": valid.resolve()
    }
    assert builder._verify_transitive_artifacts([{"one": _artifact(valid)}, "plain"], "tree") == 1


def test_operational_and_bootstrap_reject_inconsistent_inputs() -> None:
    with pytest.raises(builder.EvidenceError, match="usage is missing"):
        builder._operational_from_journal([{"latency_seconds": 1.0}], "scenegraph")
    native_usage = {
        "num_calls": 1,
        "total_prompt_tokens": 1,
        "total_completion_tokens": 1,
        "total_reasoning_tokens": 0,
        "max_prompt_tokens": 2,
        "max_completion_tokens": 1,
    }
    with pytest.raises(builder.EvidenceError, match="native usage is inconsistent"):
        builder._operational_from_journal(
            [{"latency_seconds": 1.0, "usage": native_usage}], "native"
        )
    atlas_usage = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 3,
        "num_calls": 1,
        "estimated_cost_usd": 0.0,
    }
    with pytest.raises(builder.EvidenceError, match="Atlas usage is inconsistent"):
        builder._operational_from_journal(
            [{"latency_seconds": 1.0, "usage": atlas_usage}], "scenegraph"
        )
    with pytest.raises(builder.EvidenceError, match="cluster membership drifted"):
        builder._paired_bootstrap({"left": 1.0}, {"right": 1.0})


def test_current_reviewed_role_config_hash_is_frozen() -> None:
    expected = "0ae73a840abe5a45462e23a0f436d499eaf8860e31767fd60e116467ce62d452"
    assert builder.QSPATIAL_RUNTIME_SHA256["role_config"] == expected


def test_rejects_tampered_role_config_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    score_value = json.loads(score.read_text(encoding="utf-8"))
    seal_path = Path(score_value["all_arms_seal"]["path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    role_path = Path(seal["artifacts"]["scoring_runtime"]["role_config"]["path"])
    _write_json(role_path, {"tampered": True})
    with pytest.raises(builder.EvidenceError, match="role_config artifact changed"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=tmp_path / "unreached-vsi.json")


def test_api_requires_vsi_full_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="vsi_path"):
        builder.build_assets(score, tmp_path / "missing-keyword")
    with pytest.raises(builder.EvidenceError, match="P4 VSI full gate is required"):
        builder.build_assets(score, tmp_path / "explicit-none", vsi_path=None)  # type: ignore[arg-type]


def test_cli_requires_vsi_full_gate() -> None:
    with pytest.raises(SystemExit) as exc_info:
        builder.parse_args(
            [
                "--qspatial-offline-score",
                "qspatial.json",
                "--output-dir",
                "assets",
            ]
        )
    assert exc_info.value.code == 2


def test_builds_location_free_deterministic_negative_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    first = builder.build_assets(score, tmp_path / "assets-a", vsi_path=vsi)
    second = builder.build_assets(score, tmp_path / "assets-b", vsi_path=vsi)
    first_manifest = first["p5_results_manifest.json"].read_bytes()
    second_manifest = second["p5_results_manifest.json"].read_bytes()
    assert first_manifest == second_manifest
    assert b"/scratch.global/" not in first_manifest
    assert b"/Users/" not in first_manifest
    manifest = json.loads(first_manifest)
    digest = manifest.pop("manifest_payload_sha256")
    assert digest == hashlib.sha256(builder._canonical_json_bytes(manifest)).hexdigest()
    assert all("path" not in record for record in manifest["outputs"].values())
    poster = first["p5_poster_results.md"].read_text(encoding="utf-8")
    assert "NOT SUPPORTED: negative result" in poster
    assert "VSI-Bench unbiased full run: VERIFIED, n=2362" in poster
    assert "397B ERQA diagnostic: PENDING OR BLOCKED" in poster


def test_builds_with_required_vsi_and_optional_397b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch, supported=True)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    evidence = _evidence_397b(tmp_path, monkeypatch)
    outputs = builder.build_assets(
        score, tmp_path / "assets", vsi_path=vsi, evidence_397b_path=evidence
    )
    poster = outputs["p5_poster_results.md"].read_text(encoding="utf-8")
    assert "SUPPORTED: the decisive-win claim" in poster
    assert "VSI-Bench unbiased full run: VERIFIED, n=2362" in poster
    assert "397B ERQA diagnostic: VERIFIED, job 12590146, 3/5" in poster


def test_rejects_forged_paired_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    value = json.loads(score.read_text(encoding="utf-8"))
    value["paired_cluster_comparisons"]["metric_correct_minus_native"]["sr_at_1_25"]["estimate"] = (
        1.0
    )
    _write_json(score, value)
    with pytest.raises(builder.EvidenceError, match="independent recomputation"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=tmp_path / "unreached-vsi.json")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("top_schema", "top-level schema drifted"),
        ("claim_boolean", "claim verdicts must be Boolean"),
        ("provenance", "execution provenance differs"),
        ("metric_provenance", "metric provenance is incomplete"),
        ("result_schema", "result does not contain the exact named contract"),
        ("row_count", "exactly 97 scored rows"),
        ("row_order", "not exact ordered Gap-97"),
        ("row_schema", "scored-row schema drifted"),
        ("row_text", "incomplete text or cluster data"),
        ("success_indicator", "invalid success indicators"),
        ("loss_cap", "exceeds capped log-ratio loss"),
        ("relative_error", "outside its valid range"),
        ("grounding_counts", "grounding counts are inconsistent"),
        ("failure_reasons", "failure reasons do not reconcile"),
        ("unit_counts", "ground_truth_unit_counts do not reconcile"),
        ("row_aggregate", "row aggregate sr_at_1_25 drifted"),
        ("cluster_count", "cluster count drifted"),
        ("cluster_aggregate", "cluster aggregate sr_at_1_25 drifted"),
        ("cluster_values", "cluster values sr_at_1_25 drifted"),
        ("mre", "MRE fields drifted"),
        ("median_nullability", "median log-ratio nullability drifted"),
        ("audit", "project scorer audit drifted"),
        ("crosscheck", "project scorer crosscheck is incomplete"),
        ("project_artifact", "project scorer artifact does not bind"),
        ("operational", "operational summary differs"),
        ("fallback", "fallback counts differ"),
        ("claim_recompute", "claim verdict differs"),
    ],
)
def test_rejects_qspatial_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    value = json.loads(score.read_text(encoding="utf-8"))
    result = value["arm_results"]["scenegraph"]
    row = result["rows"][0]
    row_metrics = result["row_metrics"]
    cluster_metrics = result["cluster_metrics"]
    if case == "top_schema":
        value["unexpected"] = None
    elif case == "claim_boolean":
        value["claim_gates"]["visual_claim_supported"] = 0
    elif case == "provenance":
        value["execution_provenance"] = {}
    elif case == "metric_provenance":
        value["metric_provenance"] = None
    elif case == "result_schema":
        del result["cluster_values"]
    elif case == "row_count":
        result["rows"].pop()
    elif case == "row_order":
        result["rows"][0], result["rows"][1] = result["rows"][1], result["rows"][0]
    elif case == "row_schema":
        row["unexpected"] = None
    elif case == "row_text":
        row["prediction_text"] = ""
    elif case == "success_indicator":
        row["sr_at_1_25"] = 0.5
    elif case == "loss_cap":
        row["capped_absolute_log_ratio"] = math.log(10.0) + 1.0
    elif case == "relative_error":
        row["relative_error"] = -1.0
    elif case == "grounding_counts":
        row_metrics["valid_predictions"] = builder.QSPATIAL_ROWS - 1
    elif case == "failure_reasons":
        row_metrics["failure_reasons"] = {"forged": 1}
    elif case == "unit_counts":
        row_metrics["ground_truth_unit_counts"] = {"centimeter": builder.QSPATIAL_ROWS - 1}
    elif case == "row_aggregate":
        row_metrics["sr_at_1_25"] = 0.5
    elif case == "cluster_count":
        cluster_metrics["cluster_count"] = builder.QSPATIAL_CLUSTERS - 1
    elif case == "cluster_aggregate":
        cluster_metrics["sr_at_1_25"] = 0.5
    elif case == "cluster_values":
        result["cluster_values"]["sr_at_1_25"]["c0"] = 0.5
    elif case == "mre":
        row_metrics["valid_prediction_mre"] = 0.2
    elif case == "median_nullability":
        row_metrics["median_absolute_log_ratio"] = None
    elif case == "audit":
        result["project_scorer_audit"]["total_samples"] = builder.QSPATIAL_ROWS - 1
    elif case == "crosscheck":
        result["preregistered_project_scorer_crosscheck"]["status"] = "fail"
    elif case == "project_artifact":
        result["preregistered_project_metrics"]["overall_accuracy"] = 0.5
    elif case == "operational":
        value["operational_by_arm"]["scenegraph"]["total_calls"] += 1
    elif case == "fallback":
        value["hidden_fallback_counts"]["scenegraph"] = 1
    elif case == "claim_recompute":
        value["claim_gates"]["visual_claim_supported"] = True
    else:
        raise AssertionError(f"unknown test case: {case}")
    _write_json(score, value)
    with pytest.raises(builder.EvidenceError, match=message):
        builder.build_assets(score, tmp_path / "assets", vsi_path=tmp_path / "unreached-vsi.json")


def test_rejects_cross_arm_cluster_map_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    value = json.loads(score.read_text(encoding="utf-8"))
    value["arm_results"]["native"]["rows"][0]["cluster_id"] = "c1"
    _write_json(score, value)
    with pytest.raises(builder.EvidenceError, match="sealed journal"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=tmp_path / "unreached-vsi.json")


def test_rejects_missing_named_qspatial_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    score_value = json.loads(score.read_text(encoding="utf-8"))
    seal_path = Path(score_value["all_arms_seal"]["path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    del seal["artifacts"]["arm_gates"]["native"]
    _write_json(seal_path, seal)
    score_value["all_arms_seal"] = _artifact(seal_path)
    _write_json(score, score_value)
    with pytest.raises(builder.EvidenceError, match="exact named contract"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=tmp_path / "unreached-vsi.json")


def test_rejects_nonfinite_rendered_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    value = json.loads(score.read_text(encoding="utf-8"))
    value["arm_results"]["native"]["row_metrics"]["valid_prediction_mre"] = float("nan")
    _write_json(score, value)
    with pytest.raises(builder.EvidenceError, match="MRE"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=tmp_path / "unreached-vsi.json")


def test_rejects_vsi_recomputed_score_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    official_path = Path(gate["artifacts"]["official_results"]["path"])
    official = json.loads(official_path.read_text(encoding="utf-8"))
    official["overall_accuracy"] = 0.9
    official["overall_accuracy_pct"] = 90.0
    gate["official_summary"] = {
        key: value for key, value in official.items() if key != "detailed_results"
    }
    _write_json(official_path, official)
    gate["artifacts"]["official_results"] = _artifact(official_path)
    _write_json(vsi, gate)
    with pytest.raises(builder.EvidenceError, match="independent score recomputation"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_missing_named_vsi_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    del gate["artifacts"]["full_accounting"]
    _write_json(vsi, gate)
    with pytest.raises(builder.EvidenceError, match="exact named contract"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("top_schema", "top-level schema drifted"),
        ("bound_p3", "bound to a different QSpatial"),
        ("prediction_content", "exact successful frozen IDs"),
        ("prediction_usage", "usage is missing"),
        ("statuses", "attempt statuses are invalid"),
        ("aggregate_abnormal", "accounting has abnormal terminal counters"),
        ("gate_terminal", "gate terminal counters differ"),
        ("no_attempts", "no named attempt chain"),
        ("attempt_length", "attempt chain differs"),
        ("attempt_status", "status or kind drifted"),
        ("nested_binding", "nested artifact binding drifted"),
        ("smoke_binding", "binds a different P4 smoke gate"),
        ("journal_binding", "journal snapshot binding drifted"),
        ("selected_count", "selected-ID count drifted"),
        ("service_identity", "service identity drifted"),
        ("selected_chain", "does not cover the exact frozen IDs"),
        ("official_count", "do not contain all 2362 rows"),
        ("official_order", "IDs differ from the frozen order"),
        ("official_schema", "detail row 1 schema drifted"),
        ("official_unknown", "unknown question type"),
        ("official_counts", "question-type counts differ"),
        ("p3_claims", "does not preserve the exact P3 claim verdicts"),
        ("gate_counts", "attempt counts differ from accounting"),
    ],
)
def test_rejects_vsi_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    if case == "top_schema":
        gate["unexpected"] = None
        _write_json(vsi, gate)
    elif case == "bound_p3":
        gate["artifacts"]["p3_offline_score"] = gate["artifacts"]["protocol"]
        _write_json(vsi, gate)
    elif case in {"prediction_content", "prediction_usage"}:
        predictions_path = Path(gate["artifacts"]["predictions"]["path"])
        rows = [
            json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
        ]
        if case == "prediction_content":
            rows[0]["content"] = ""
        else:
            del rows[0]["usage"]
        _write_jsonl(predictions_path, rows)
        prediction_record = _artifact(predictions_path)
        accounting_path = Path(gate["artifacts"]["full_accounting"]["path"])
        accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
        accounting["predictions"] = prediction_record
        _write_json(accounting_path, accounting)
        gate["artifacts"]["predictions"] = prediction_record
        gate["artifacts"]["full_accounting"] = _artifact(accounting_path)
        _write_json(vsi, gate)
    elif case == "statuses":
        _update_vsi_accounting(vsi, {"attempt_statuses": ["pass"] * 48})
    elif case == "aggregate_abnormal":
        terminal = dict(gate["terminal_counter_totals"])
        terminal["error"] = 1
        _update_vsi_accounting(vsi, {"terminal_counter_totals": terminal})
    elif case == "gate_terminal":
        gate["terminal_counter_totals"]["stop"] += 1
        _write_json(vsi, gate)
    elif case == "no_attempts":
        gate["artifacts"]["attempts"] = []
        _write_json(vsi, gate)
    elif case == "attempt_length":
        gate["artifacts"]["attempts"].pop()
        _write_json(vsi, gate)
    elif case == "attempt_status":
        _update_vsi_attempt(vsi, 0, {"status": "pass"})
    elif case == "nested_binding":
        attempt = _vsi_attempt(vsi, 0)
        nested = dict(attempt["artifacts"])
        nested["selection"] = nested["p4_smoke_gate"]
        _update_vsi_attempt(vsi, 0, {"artifacts": nested})
    elif case == "smoke_binding":
        attempt = _vsi_attempt(vsi, 0)
        nested = dict(attempt["artifacts"])
        nested["p4_smoke_gate"] = nested["selection"]
        _update_vsi_attempt(vsi, 0, {"artifacts": nested}, sync_nested=True)
    elif case == "journal_binding":
        record = gate["artifacts"]["attempts"][0]
        record["immutable_journal_after"] = record["nested_artifacts"]["selection"]
        _write_json(vsi, gate)
    elif case == "selected_count":
        _update_vsi_attempt(vsi, 0, {"newly_completed_count": 49})
    elif case == "service_identity":
        attempt = _vsi_attempt(vsi, 0)
        service = dict(attempt["service"])
        service["endpoint_model"] = "wrong-model"
        _update_vsi_attempt(vsi, 0, {"service": service})
    elif case == "selected_chain":
        attempt = _vsi_attempt(vsi, 0)
        selected = list(attempt["selected_ids"])
        selected[0] = "forged-id"
        _update_vsi_attempt(vsi, 0, {"selected_ids": selected})
    elif case == "official_count":
        _update_vsi_official(vsi, lambda official: official["detailed_results"].pop())
    elif case == "official_order":

        def swap_rows(official: dict[str, object]) -> None:
            rows = official["detailed_results"]
            rows[0], rows[1] = rows[1], rows[0]

        _update_vsi_official(vsi, swap_rows)
    elif case == "official_schema":

        def add_field(official: dict[str, object]) -> None:
            official["detailed_results"][0]["unexpected"] = None

        _update_vsi_official(vsi, add_field)
    elif case == "official_unknown":

        def unknown_type(official: dict[str, object]) -> None:
            official["detailed_results"][0]["question_type"] = "unknown"

        _update_vsi_official(vsi, unknown_type)
    elif case == "official_counts":

        def shift_type(official: dict[str, object]) -> None:
            official["detailed_results"][0]["question_type"] = "object_size_estimation"

        _update_vsi_official(vsi, shift_type)
    elif case == "p3_claims":
        gate["p3_claim_gates"] = {}
        _write_json(vsi, gate)
    elif case == "gate_counts":
        gate["attempt_count"] += 1
        _write_json(vsi, gate)
    else:
        raise AssertionError(f"unknown test case: {case}")
    with pytest.raises(builder.EvidenceError, match=message):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_vsi_aggregate_stop_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    terminal = {
        "stop": builder.VSI_ROWS * 2 + 1,
        "length": 0,
        "abort": 0,
        "error": 0,
        "repetition": 0,
    }
    _update_vsi_accounting(
        vsi,
        {"terminal_counter_totals": terminal},
        mirror_terminal_totals=True,
    )
    with pytest.raises(builder.EvidenceError, match="aggregate stop counter"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_vsi_per_attempt_abnormal_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    attempt = _vsi_attempt(vsi, 0)
    deltas = dict(attempt["terminal_counter_deltas"])
    deltas["length"] = 1
    _update_vsi_attempt(vsi, 0, {"terminal_counter_deltas": deltas})
    with pytest.raises(builder.EvidenceError, match="abnormal terminal counters"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_vsi_per_attempt_journal_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    attempt = _vsi_attempt(vsi, 0)
    _update_vsi_attempt(
        vsi,
        0,
        {"journaled_call_delta": int(attempt["journaled_call_delta"]) - 1},
    )
    with pytest.raises(builder.EvidenceError, match="stop counter differs from journal calls"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_vsi_per_attempt_counter_sum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    attempt = _vsi_attempt(vsi, 0)
    deltas = dict(attempt["terminal_counter_deltas"])
    deltas["stop"] = int(deltas["stop"]) + 1
    _update_vsi_attempt(
        vsi,
        0,
        {
            "terminal_counter_deltas": deltas,
            "journaled_call_delta": int(attempt["journaled_call_delta"]) + 1,
        },
    )
    with pytest.raises(builder.EvidenceError, match="do not sum to aggregate totals"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


@pytest.mark.parametrize("scope", ["aggregate", "attempt"])
def test_rejects_vsi_terminal_counter_schema_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    if scope == "aggregate":
        gate = json.loads(vsi.read_text(encoding="utf-8"))
        accounting_path = Path(gate["artifacts"]["full_accounting"]["path"])
        accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
        counters = dict(accounting["terminal_counter_totals"])
        counters["extra"] = 0
        _update_vsi_accounting(vsi, {"terminal_counter_totals": counters})
    else:
        attempt = _vsi_attempt(vsi, 0)
        counters = dict(attempt["terminal_counter_deltas"])
        del counters["repetition"]
        _update_vsi_attempt(vsi, 0, {"terminal_counter_deltas": counters})
    with pytest.raises(builder.EvidenceError, match="exact named contract"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_oversized_vsi_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    ids_path = Path(gate["artifacts"]["frozen_ids"]["path"])
    ids = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]
    _update_vsi_attempt(
        vsi,
        0,
        {"selected_ids": ids[:51], "newly_completed_count": 51},
    )
    with pytest.raises(builder.EvidenceError, match="selected IDs are invalid"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_reused_vsi_service_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    first_record = gate["artifacts"]["attempts"][0]
    first_attempt = json.loads(Path(first_record["path"]).read_text(encoding="utf-8"))
    _update_vsi_attempt(vsi, 1, {"service": first_attempt["service"]})
    with pytest.raises(builder.EvidenceError, match="distinct fresh service pairs"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_same_pair_vsi_service_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    attempt = _vsi_attempt(vsi, 0)
    service = dict(attempt["service"])
    service["gpu_tool_job_id"] = service["vllm_job_id"]
    _update_vsi_attempt(vsi, 0, {"service": service})
    with pytest.raises(builder.EvidenceError, match="one job for both services"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_cross_role_vsi_service_job_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    first = _vsi_attempt(vsi, 0)
    second = _vsi_attempt(vsi, 1)
    service = dict(second["service"])
    service["vllm_job_id"] = first["service"]["gpu_tool_job_id"]
    _update_vsi_attempt(vsi, 1, {"service": service})
    with pytest.raises(builder.EvidenceError, match="distinct fresh service pairs"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_vsi_service_schema_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    attempt = _vsi_attempt(vsi, 0)
    service = dict(attempt["service"])
    service["unexpected"] = "field"
    _update_vsi_attempt(vsi, 0, {"service": service})
    with pytest.raises(builder.EvidenceError, match="exact named contract"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_fewer_than_48_vsi_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    accounting_path = Path(gate["artifacts"]["full_accounting"]["path"])
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting["attempt_statuses"] = ["pass"]
    _write_json(accounting_path, accounting)
    gate["artifacts"]["full_accounting"] = _artifact(accounting_path)
    gate["artifacts"]["attempts"] = [gate["artifacts"]["attempts"][-1]]
    gate["attempt_count"] = 1
    gate["interrupted_attempt_count"] = 0
    _write_json(vsi, gate)
    with pytest.raises(builder.EvidenceError, match="at least 48 sequential attempts"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


@pytest.mark.parametrize("job_field", ["vllm_job_id", "gpu_tool_job_id"])
def test_rejects_nonnumeric_vsi_service_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_field: str
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    gate = json.loads(vsi.read_text(encoding="utf-8"))
    first_record = gate["artifacts"]["attempts"][0]
    first_attempt = json.loads(Path(first_record["path"]).read_text(encoding="utf-8"))
    service = dict(first_attempt["service"])
    service[job_field] = "not-a-job"
    _update_vsi_attempt(vsi, 0, {"service": service})
    with pytest.raises(builder.EvidenceError, match="service job IDs are invalid"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_unicode_decimal_vsi_service_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    attempt = _vsi_attempt(vsi, 0)
    service = dict(attempt["service"])
    service["vllm_job_id"] = "\u0661\u0662\u0663"
    _update_vsi_attempt(vsi, 0, {"service": service})
    with pytest.raises(builder.EvidenceError, match="service job IDs are invalid"):
        builder.build_assets(score, tmp_path / "assets", vsi_path=vsi)


def test_rejects_wrong_historical_397b_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    evidence = _evidence_397b(tmp_path, monkeypatch)
    value = json.loads(evidence.read_text(encoding="utf-8"))
    value["job_id"] = "99999999"
    _write_json(evidence, value)
    with pytest.raises(builder.EvidenceError, match="identity drifted"):
        builder.build_assets(
            score,
            tmp_path / "assets",
            vsi_path=vsi,
            evidence_397b_path=evidence,
        )


def test_refuses_overwrite_and_active_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score = _qspatial_score(tmp_path, monkeypatch)
    vsi = _vsi_gate(tmp_path, score, monkeypatch)
    output = tmp_path / "assets"
    builder.build_assets(score, output, vsi_path=vsi)
    with pytest.raises(builder.EvidenceError, match="refusing to overwrite"):
        builder.build_assets(score, output, vsi_path=vsi)
    other = tmp_path / "locked"
    other.mkdir()
    (other / ".p5-assets.lock").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(builder.EvidenceError, match="publication is active"):
        builder.build_assets(score, other, vsi_path=vsi)
