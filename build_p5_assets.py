"""Build publication assets only from sealed, hash-verified benchmark evidence.

The QSpatial++ Gap-97 offline score and VSI-Bench full gate are required.
Historical 397B evidence is optional: an omitted input is rendered as pending,
while supplied evidence must pass its complete validation contract. Existing
outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ARM_ORDER = ["scenegraph", "question_only", "metric_correct", "metric_shuffled", "native"]
ARM_LABELS = {
    "scenegraph": "Spatial Atlas scenegraph",
    "question_only": "Question only",
    "metric_correct": "Spatial Atlas metric, correct image",
    "metric_shuffled": "Spatial Atlas metric, shuffled image",
    "native": "SpatialClaw native",
}
COMPARISON_ORDER = [
    "metric_correct_minus_question_only",
    "metric_correct_minus_metric_shuffled",
    "metric_correct_minus_scenegraph",
    "metric_correct_minus_native",
]
SLICE_SHA256 = "7180f4879b01b14d78d0500c2d5c44c55498708cfc1bc9102cf19d57ec8cd159"
SCORING_SHA256 = "876c5972862ff56332ca8b98fead30f2c910483e57832cb583a9aebd72cc1cd3"
QSPATIAL_ROWS = 97
QSPATIAL_CLUSTERS = 54
VSI_ROWS = 2362
EXPECTED_IDS = tuple(str(row_id) for row_id in range(101) if row_id not in {77, 78, 91, 98})
EXPECTED_IDS_SHA256 = "2fabce17cc09263f43c2324ed4f5672c5152ed50a0acd1e596f6ccb55e9a72f1"
PARSER_SHA256 = "1993ce6234148b5efee98acb50a9ff0acb207d387e0fc0d40fe67963a371ace8"
GEOMETRY_SHA256 = "7f576fbcf0740d9bc9e2511f273b7e0be1b2d20795f8ee4ea2a6fc92b4ef16a7"
BOOTSTRAP_SEED = 20_260_711
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
VISUAL_RULE = (
    "metric_correct beats question_only and metric_shuffled on cluster-balanced "
    "SR@1.25 with paired-bootstrap lower bounds above zero"
)
DECISIVE_RULE = (
    "visual gate passes; metric_correct beats scenegraph and native on SR@1.25 "
    "with lower bounds above zero, improves SR@2, and reduces capped absolute "
    "log-ratio error"
)
EXECUTION_PROVENANCE_NAMES = {"source", "model", "environment"}
SCORING_RUNTIME_NAMES = {
    "role_config",
    "dataset_config",
    "qspatial_scorer_source",
    "benchmark_factory_source",
}
QSPATIAL_RUNTIME_SHA256 = {
    "role_config": "0ae73a840abe5a45462e23a0f436d499eaf8860e31767fd60e116467ce62d452",
    "dataset_config": "7d1616e4b87e361ba93a0d0f38932a4b4a164c572bc8ec5ff88ca37ba7c4f362",
    "qspatial_scorer_source": "97d5c192258e592c42256ddcb7d4973714ca12b773f08e129df12d7dd45244a9",
    "benchmark_factory_source": "53029dd1618b94042cf2d2b129ed32b4dd782f6e8425704bd08d30339d23f328",
}
CONTROL_MAPPING_SHA256 = "95dc901e09fd05d5b742c93c390d51812682d2fe7e19abb396f27da793a99c1c"
QSPATIAL_SEAL_ARTIFACT_NAMES = {
    "pilot_readiness",
    "expected_ids",
    "arm_gates",
    "journal_seals",
    "journals",
    "scoring_runtime",
}
QSPATIAL_SEAL_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "scoring_authorized",
    "labels_loaded",
    "arm_order",
    "expected_count_per_arm",
    "total_sealed_rows",
    "slice_sha256",
    "parser_sha256",
    "geometry_sha256",
    "scoring_sha256",
    "arm_final_vllm_job_ids",
    "all_attempt_vllm_job_ids",
    "operational_by_arm",
    "execution_provenance",
    "artifacts",
}
QSPATIAL_ARM_GATE_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "arm",
    "engine",
    "image_mode",
    "expected_count",
    "slice_sha256",
    "parser_sha256",
    "geometry_sha256",
    "scoring_sha256",
    "concurrency",
    "image_root",
    "service",
    "service_chain",
    "role_policy",
    "atlas_policy",
    "label_free_protocol",
    "lock_protocol",
    "terminal_counter_deltas",
    "usage_totals",
    "attempt_count",
    "interrupted_attempt_count",
    "arm_attempt_vllm_job_ids",
    "arm_attempt_gpu_tool_job_ids",
    "gpu_liveness_pairs",
    "attempt_preflight_artifacts",
    "geometry_valid_count",
    "unavailable_count",
    "hidden_fallback_count",
    "operational",
    "control_mapping_sha256",
    "execution_provenance",
    "artifacts",
}
ARM_SPECS = {
    "scenegraph": ("scenegraph", "correct", False),
    "question_only": ("scenegraph", "question-only", False),
    "metric_correct": ("metric", "correct", True),
    "metric_shuffled": ("metric", "shuffled", True),
    "native": ("native", "correct", True),
}
QSPATIAL_BASE_GATE_ARTIFACT_NAMES = {
    "journal",
    "raw_journal",
    "seal",
    "expected_ids",
    "full_accounting",
    "endpoint_manifest",
    "role_config",
    "dataset_config",
    "qspatial_scorer_source",
    "benchmark_factory_source",
    "protocol_manifest",
    "atlas_policy",
    "image_manifest",
    "protocol_records",
    "inference_manifest",
    "pilot_readiness",
    "run_manifest",
}
QSPATIAL_CROSSCHECK_FIELDS = {
    "total_samples",
    "correct_samples",
    "overall_accuracy",
    "sr_at_1_25",
    "sr_at_2",
    "cluster_balanced_sr_at_1_25",
    "cluster_balanced_sr_at_2",
    "valid_predictions",
    "cluster_count",
    "grounding_failures",
    "grounding_failure_rate",
    "median_absolute_log_ratio",
    "valid_prediction_median_absolute_log_ratio",
    "mre",
    "valid_prediction_mre",
    "failure_reasons",
    "ground_truth_unit_counts",
    "prediction_unit_counts",
    "cluster_bootstrap_sr_at_1_25",
    "cluster_bootstrap_sr_at_2",
}
VSI_IDS_SHA256 = "e45c57fe8158d46a623f4ac06527f71cc9addfd5ab8bcb1783b15ec244e13690"
VSI_PARQUET_SHA256 = "dd4fcb087f724652f9f178af4ff6ebf2be1eb11bf7ffba51ce2b8396329ee9e1"
VSI_QUESTION_COUNTS = {
    "obj_appearance_order": 318,
    "object_abs_distance": 434,
    "object_counting": 251,
    "object_rel_direction_easy": 212,
    "object_rel_direction_hard": 116,
    "object_rel_direction_medium": 54,
    "object_rel_distance": 310,
    "object_size_estimation": 353,
    "room_size_estimation": 200,
    "route_planning": 114,
}
VSI_GATE_ARTIFACT_NAMES = {
    "p3_offline_score",
    "p4_smoke_gate",
    "protocol",
    "frozen_ids",
    "full_accounting",
    "predictions",
    "official_results",
    "attempts",
}
VSI_ATTEMPT_ARTIFACT_NAMES = {
    "selection",
    "accounting_gate",
    "immutable_journal_after",
    "immutable_run_config",
    "gpu_liveness_before",
    "gpu_liveness_after",
    "p4_smoke_gate",
}
VSI_TERMINAL_REASONS = {"stop", "length", "abort", "error", "repetition"}
VSI_ABNORMAL_REASONS = VSI_TERMINAL_REASONS - {"stop"}
VSI_SERVICE_FIELDS = {
    "endpoint_model",
    "endpoint_reasoning_parser",
    "vllm_job_id",
    "gpu_tool_job_id",
    "endpoint_url",
    "metrics_url",
    "endpoint_manifest",
}
EVIDENCE_397B_JOB_ID = "12590146"
EVIDENCE_397B_BACKBONE = "Qwen3.5-397B-A17B"
EVIDENCE_397B_ARTIFACT_SHA256 = {
    "job_script": "416605476e7820782f536037ed09f5270f0bea6b39dbc1701e3a2fb21c094734",
    "log": "8909b51848eb775f74e886ccd2ef5a22dada4b36ed2c0e55e8fc5d90708008e5",
    "config": "3cf7954e67df75a7abaacfc427e60ff10cc12b866f1385b84d8b87752f67b748",
    "predictions": "fe3fd907e8554700c6ae3c70dea9774faa3627b3c3b44f9359a36f0b7ad80b3c",
    "result": "bcb16bd0c69e0854bb9fcd7e9b403669d213f27eaece66512937ad8aecd53043",
}
OUTPUT_NAMES = (
    "p5_qspatial_arms.csv",
    "p5_qspatial_comparisons.csv",
    "p5_poster_results.md",
    "p5_paper_results.md",
    "p5_results_manifest.json",
)


class EvidenceError(ValueError):
    """An evidence artifact failed a fail-closed validation."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot load {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be a JSON object")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvidenceError(f"artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_artifact(record: object, description: str) -> Path:
    if not isinstance(record, Mapping):
        raise EvidenceError(f"{description} artifact is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise EvidenceError(f"{description} artifact path must be absolute")
    sha256 = record.get("sha256")
    size = record.get("size_bytes")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise EvidenceError(f"{description} artifact metadata is invalid")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise EvidenceError(f"{description} artifact is missing: {path}")
    if record.get("sha256") != _sha256(path) or record.get("size_bytes") != path.stat().st_size:
        raise EvidenceError(f"{description} artifact changed")
    return path


def _public_artifact(path: Path, logical_name: str) -> dict[str, Any]:
    """Return location-free artifact metadata safe for a public manifest."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvidenceError(f"artifact is missing: {logical_name}")
    return {
        "logical_name": logical_name,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _exact_mapping(value: object, names: set[str], description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise EvidenceError(f"{description} does not contain the exact named contract")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot load {description}: {exc}") from exc
    if not payload or not payload.endswith(b"\n"):
        raise EvidenceError(f"{description} must be nonempty and newline terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"{description} row {line_number} is invalid: {exc}") from exc
        if not isinstance(row, dict):
            raise EvidenceError(f"{description} row {line_number} is not an object")
        rows.append(row)
    return rows


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _canonical_id_bytes(ids: Sequence[str]) -> bytes:
    return (json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _integer(value: object, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{description} must be an integer at least {minimum}")
    return value


def _terminal_counters(value: object, description: str) -> dict[str, int]:
    counters = _exact_mapping(value, VSI_TERMINAL_REASONS, description)
    return {
        reason: _integer(counters[reason], f"{description}.{reason}")
        for reason in sorted(VSI_TERMINAL_REASONS)
    }


def _ascii_decimal_job_id(value: object) -> bool:
    return isinstance(value, str) and value.isascii() and value.isdecimal()


def _probability(value: object, description: str) -> float:
    result = _finite(value, description, minimum=0.0)
    if result > 1.0:
        raise EvidenceError(f"{description} exceeds 1")
    return result


def _same_optional_number(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _same_number(left, right)


def _validate_artifact_group(value: object, names: set[str], description: str) -> dict[str, Path]:
    group = _exact_mapping(value, names, description)
    return {name: _verify_artifact(group[name], f"{description}.{name}") for name in sorted(names)}


def _verify_transitive_artifacts(value: object, description: str) -> int:
    """Verify every embedded artifact record and return the number checked."""
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            _verify_artifact(value, description)
            return 1
        return sum(
            _verify_transitive_artifacts(child, f"{description}.{key}")
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(
            _verify_transitive_artifacts(child, f"{description}[{index}]")
            for index, child in enumerate(value)
        )
    return 0


def _finite(value: object, description: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise EvidenceError(f"{description} is outside its valid range")
    return result


def _optional_finite(value: object, description: str) -> float | None:
    return None if value is None else _finite(value, description, minimum=0.0)


def _same_number(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _required_identity(
    value: Mapping[str, Any], expected: Mapping[str, Any], description: str
) -> None:
    drift = {key: value.get(key) for key, wanted in expected.items() if value.get(key) != wanted}
    if drift:
        raise EvidenceError(f"{description} identity drifted: {drift}")


def _expected_gate_artifacts(arm: str) -> set[str]:
    names = set(QSPATIAL_BASE_GATE_ARTIFACT_NAMES)
    if arm != ARM_ORDER[0]:
        names.add("prior_gate")
    if arm == "metric_shuffled":
        names.add("control_mapping")
    if ARM_SPECS[arm][2]:
        names.add("gpu_liveness_final")
    return names


def _validate_qspatial_seal(
    seal_path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int], int]:
    seal = _load_json(seal_path, "QSpatial all-arms seal")
    if set(seal) != QSPATIAL_SEAL_FIELDS:
        raise EvidenceError("QSpatial all-arms seal top-level schema drifted")
    _required_identity(
        seal,
        {
            "schema_version": 1,
            "kind": "qspatial_full_all_arms_sealed",
            "status": "pass",
            "scoring_authorized": True,
            "labels_loaded": False,
            "arm_order": ARM_ORDER,
            "expected_count_per_arm": QSPATIAL_ROWS,
            "total_sealed_rows": QSPATIAL_ROWS * len(ARM_ORDER),
            "slice_sha256": SLICE_SHA256,
            "parser_sha256": PARSER_SHA256,
            "geometry_sha256": GEOMETRY_SHA256,
            "scoring_sha256": SCORING_SHA256,
        },
        "QSpatial all-arms seal",
    )
    provenance = _exact_mapping(
        seal.get("execution_provenance"),
        EXECUTION_PROVENANCE_NAMES,
        "QSpatial execution provenance",
    )
    for name, record in provenance.items():
        _verify_artifact(record, f"QSpatial {name} provenance")
    artifacts = _exact_mapping(
        seal.get("artifacts"), QSPATIAL_SEAL_ARTIFACT_NAMES, "QSpatial seal artifacts"
    )
    readiness_path = _verify_artifact(artifacts["pilot_readiness"], "QSpatial pilot readiness")
    readiness = _load_json(readiness_path, "QSpatial pilot readiness")
    _required_identity(
        readiness,
        {"kind": "qspatial_operational_pilot_readiness", "status": "pass"},
        "QSpatial pilot readiness",
    )
    expected_ids_path = _verify_artifact(artifacts["expected_ids"], "QSpatial expected IDs")
    if expected_ids_path.read_bytes() != _canonical_id_bytes(EXPECTED_IDS):
        raise EvidenceError("QSpatial expected-ID artifact is not canonical Gap-97")
    if _sha256(expected_ids_path) != EXPECTED_IDS_SHA256:
        raise EvidenceError("QSpatial expected-ID hash drifted")
    runtime = _exact_mapping(
        artifacts["scoring_runtime"], SCORING_RUNTIME_NAMES, "QSpatial scoring runtime"
    )
    for name, record in runtime.items():
        runtime_path = _verify_artifact(record, f"QSpatial scoring runtime {name}")
        if _sha256(runtime_path) != QSPATIAL_RUNTIME_SHA256[name]:
            raise EvidenceError(f"QSpatial scoring runtime {name} differs from the frozen source")
    gates = _exact_mapping(artifacts["arm_gates"], set(ARM_ORDER), "QSpatial arm gates")
    seals = _exact_mapping(artifacts["journal_seals"], set(ARM_ORDER), "QSpatial journal seals")
    journals = _exact_mapping(artifacts["journals"], set(ARM_ORDER), "QSpatial journals")
    operational = _exact_mapping(
        seal.get("operational_by_arm"), set(ARM_ORDER), "QSpatial sealed operations"
    )
    journal_rows: dict[str, list[dict[str, Any]]] = {}
    hidden_fallback_counts: dict[str, int] = {}
    previous_gate: object | None = None
    verified = len(provenance) + len(runtime) + 2
    for arm in ARM_ORDER:
        gate_path = _verify_artifact(gates[arm], f"QSpatial {arm} arm gate")
        journal_path = _verify_artifact(journals[arm], f"QSpatial {arm} journal")
        journal_seal_path = _verify_artifact(seals[arm], f"QSpatial {arm} journal seal")
        verified += 3
        gate = _load_json(gate_path, f"QSpatial {arm} arm gate")
        if set(gate) != QSPATIAL_ARM_GATE_FIELDS:
            raise EvidenceError(f"QSpatial {arm} arm-gate schema drifted")
        engine, image_mode, _requires_gpu = ARM_SPECS[arm]
        _required_identity(
            gate,
            {
                "schema_version": 1,
                "kind": "qspatial_full_arm_gate",
                "status": "pass",
                "arm": arm,
                "engine": engine,
                "image_mode": image_mode,
                "expected_count": QSPATIAL_ROWS,
                "slice_sha256": SLICE_SHA256,
                "parser_sha256": PARSER_SHA256,
                "geometry_sha256": GEOMETRY_SHA256,
                "scoring_sha256": SCORING_SHA256,
                "concurrency": 4,
                "control_mapping_sha256": CONTROL_MAPPING_SHA256
                if arm == "metric_shuffled"
                else None,
            },
            f"QSpatial {arm} arm gate",
        )
        if (
            gate.get("execution_provenance") != provenance
            or gate.get("operational") != operational[arm]
        ):
            raise EvidenceError(
                f"QSpatial {arm} gate is not bound to seal provenance and operations"
            )
        hidden_fallback_counts[arm] = _integer(
            gate.get("hidden_fallback_count"), f"QSpatial {arm} hidden fallback count"
        )
        gate_artifacts = _exact_mapping(
            gate.get("artifacts"), _expected_gate_artifacts(arm), f"QSpatial {arm} gate artifacts"
        )
        for name, record in gate_artifacts.items():
            _verify_artifact(record, f"QSpatial {arm} gate artifact {name}")
        if (
            gate_artifacts["journal"] != journals[arm]
            or gate_artifacts["seal"] != seals[arm]
            or gate_artifacts["expected_ids"] != artifacts["expected_ids"]
            or gate_artifacts["pilot_readiness"] != artifacts["pilot_readiness"]
            or any(gate_artifacts[name] != runtime[name] for name in SCORING_RUNTIME_NAMES)
        ):
            raise EvidenceError(f"QSpatial {arm} gate artifact bindings drifted")
        if arm == ARM_ORDER[0]:
            if "prior_gate" in gate_artifacts:
                raise EvidenceError("QSpatial first arm unexpectedly has a predecessor")
        elif gate_artifacts.get("prior_gate") != previous_gate:
            raise EvidenceError(f"QSpatial {arm} predecessor binding drifted")
        previous_gate = gates[arm]
        rows = _load_jsonl(journal_path, f"QSpatial {arm} journal")
        if [str(row.get("sample_id")) for row in rows] != list(EXPECTED_IDS):
            raise EvidenceError(f"QSpatial {arm} journal is not exact ordered Gap-97")
        journal_rows[arm] = rows
        journal_seal = _load_json(journal_seal_path, f"QSpatial {arm} journal seal")
        _required_identity(
            journal_seal,
            {
                "schema_version": 1,
                "kind": "qspatial_label_free_prescore_journal_seal",
                "status": "pass",
                "expected_count": QSPATIAL_ROWS,
                "expected_ids": list(EXPECTED_IDS),
                "expected_ids_sha256": EXPECTED_IDS_SHA256,
                "journal_artifact": journals[arm],
                "expected_ids_artifact": artifacts["expected_ids"],
            },
            f"QSpatial {arm} journal seal",
        )
    return seal, journal_rows, hidden_fallback_counts, verified


def _operational_from_journal(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows, 1):
        latency = _finite(
            source.get("latency_seconds"), f"QSpatial {arm} row {index} latency", minimum=0.0
        )
        usage = source.get("usage")
        if not isinstance(usage, Mapping):
            raise EvidenceError(f"QSpatial {arm} row {index} usage is missing")
        if arm == "native":
            fields = {
                "num_calls",
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_reasoning_tokens",
                "max_prompt_tokens",
                "max_completion_tokens",
            }
            _exact_mapping(usage, fields, f"QSpatial {arm} row {index} usage")
            values = {
                name: _integer(usage[name], f"QSpatial {arm} row {index} {name}") for name in fields
            }
            if (
                values["max_prompt_tokens"] > values["total_prompt_tokens"]
                or values["max_completion_tokens"] > values["total_completion_tokens"]
                or values["total_reasoning_tokens"] > values["total_completion_tokens"]
            ):
                raise EvidenceError(f"QSpatial {arm} row {index} native usage is inconsistent")
            normalized.append(
                {
                    "latency": latency,
                    "calls": values["num_calls"],
                    "prompt": values["total_prompt_tokens"],
                    "completion": values["total_completion_tokens"],
                    "reasoning": values["total_reasoning_tokens"],
                    "cost": None,
                }
            )
        else:
            fields = {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "num_calls",
                "estimated_cost_usd",
            }
            _exact_mapping(usage, fields, f"QSpatial {arm} row {index} usage")
            prompt = _integer(usage["prompt_tokens"], f"QSpatial {arm} row {index} prompt tokens")
            completion = _integer(
                usage["completion_tokens"], f"QSpatial {arm} row {index} completion tokens"
            )
            total = _integer(usage["total_tokens"], f"QSpatial {arm} row {index} total tokens")
            calls = _integer(usage["num_calls"], f"QSpatial {arm} row {index} calls")
            cost = _finite(
                usage["estimated_cost_usd"], f"QSpatial {arm} row {index} cost", minimum=0.0
            )
            if total != prompt + completion or (calls > 0) != (total > 0):
                raise EvidenceError(f"QSpatial {arm} row {index} Atlas usage is inconsistent")
            normalized.append(
                {
                    "latency": latency,
                    "calls": calls,
                    "prompt": prompt,
                    "completion": completion,
                    "reasoning": None,
                    "cost": cost,
                }
            )
    latencies = [item["latency"] for item in normalized]
    ordered = sorted(latencies)
    rank = 0.95 * (len(ordered) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    p95 = ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])
    costs = [item["cost"] for item in normalized]
    priced = [float(value) for value in costs if value is not None]
    cost_available = len(priced) == len(costs) and any(value > 0.0 for value in priced)
    return {
        "row_count": len(normalized),
        "total_calls": sum(item["calls"] for item in normalized),
        "total_prompt_tokens": sum(item["prompt"] for item in normalized),
        "total_completion_tokens": sum(item["completion"] for item in normalized),
        "total_tokens": sum(item["prompt"] + item["completion"] for item in normalized),
        "total_reasoning_tokens": sum(item["reasoning"] for item in normalized)
        if all(item["reasoning"] is not None for item in normalized)
        else None,
        "estimated_cost_usd": math.fsum(priced) if cost_available else None,
        "estimated_cost_available": cost_available,
        "estimated_cost_unavailable_reason": None
        if cost_available
        else "native usage does not expose monetary cost"
        if arm == "native"
        else "local vLLM usage has no verified nonzero pricing",
        "latency_seconds": {
            "total": math.fsum(latencies),
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95_linear": p95,
            "maximum": max(latencies),
        },
    }


def _cluster_means(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        values.setdefault(str(row["cluster_id"]), []).append(float(row[field]))
    return {name: math.fsum(items) / len(items) for name, items in values.items()}


def _paired_bootstrap(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, Any]:
    clusters = sorted(left)
    if clusters != sorted(right) or len(clusters) != QSPATIAL_CLUSTERS:
        raise EvidenceError("QSpatial paired-bootstrap cluster membership drifted")
    differences = np.asarray([left[name] - right[name] for name in clusters], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(clusters), size=(BOOTSTRAP_ITERATIONS, len(clusters)))
    samples = differences[indices].mean(axis=1)
    quantiles = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "estimate": float(differences.mean()),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "quantile_method": "linear",
        "ci": {"lower": float(quantiles[0]), "upper": float(quantiles[1])},
    }


def _numbers_equal_structure(actual: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_numbers_equal_structure(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _same_number(actual, expected)
    return actual == expected


def _validate_qspatial(path: Path) -> tuple[dict[str, Any], int]:
    score = _load_json(path, "QSpatial offline score")
    expected_score_fields = {
        "schema_version",
        "kind",
        "status",
        "all_journals_sealed_before_labels_loaded",
        "all_arms_seal",
        "arm_order",
        "expected_count_per_arm",
        "cluster_count",
        "slice_sha256",
        "scoring_sha256",
        "execution_provenance",
        "bootstrap_contract",
        "invalid_prediction_contract",
        "metric_provenance",
        "arm_results",
        "hidden_fallback_counts",
        "operational_by_arm",
        "paired_cluster_comparisons",
        "claim_gates",
        "preregistered_project_scorer_artifacts",
    }
    if set(score) != expected_score_fields:
        raise EvidenceError("QSpatial offline score top-level schema drifted")
    _required_identity(
        score,
        {
            "schema_version": 1,
            "kind": "qspatial_full_offline_scoring",
            "status": "pass",
            "all_journals_sealed_before_labels_loaded": True,
            "arm_order": ARM_ORDER,
            "expected_count_per_arm": QSPATIAL_ROWS,
            "cluster_count": QSPATIAL_CLUSTERS,
            "slice_sha256": SLICE_SHA256,
            "scoring_sha256": SCORING_SHA256,
            "bootstrap_contract": {
                "unit": "question_cluster",
                "paired": True,
                "cluster_order": "sorted_full_cluster_hash",
                "rng": "numpy.random.default_rng",
                "seed": BOOTSTRAP_SEED,
                "iterations": BOOTSTRAP_ITERATIONS,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "quantile_method": "linear",
            },
            "invalid_prediction_contract": {
                "sr_at_1_25": 0.0,
                "sr_at_2": 0.0,
                "capped_absolute_log_ratio": math.log(10.0),
            },
        },
        "QSpatial offline score",
    )
    claims = _exact_mapping(
        score.get("claim_gates"),
        {
            "visual_claim_supported",
            "visual_rule",
            "decisive_win_supported",
            "decisive_rule",
            "negative_result_required_if_false",
        },
        "QSpatial claim gates",
    )
    _required_identity(
        claims,
        {
            "visual_rule": VISUAL_RULE,
            "decisive_rule": DECISIVE_RULE,
            "negative_result_required_if_false": True,
        },
        "QSpatial claim gates",
    )
    if not all(
        isinstance(claims[name], bool)
        for name in ("visual_claim_supported", "decisive_win_supported")
    ):
        raise EvidenceError("QSpatial claim verdicts must be Boolean")
    seal_path = _verify_artifact(score["all_arms_seal"], "QSpatial all-arms seal")
    seal, journals, sealed_hidden_fallback_counts, verified = _validate_qspatial_seal(seal_path)
    if score.get("execution_provenance") != seal.get("execution_provenance"):
        raise EvidenceError("QSpatial score execution provenance differs from the seal")
    metric_provenance = score.get("metric_provenance")
    if (
        not isinstance(metric_provenance, Mapping)
        or metric_provenance.get("project_scoring_contract_sha256") != SCORING_SHA256
        or metric_provenance.get("whole_scorer_is_not_claimed_as_upstream_official") is not True
        or metric_provenance.get("sealed_scoring_runtime_artifacts")
        != seal["artifacts"]["scoring_runtime"]
    ):
        raise EvidenceError("QSpatial metric provenance is incomplete or drifted")
    arms = _exact_mapping(score.get("arm_results"), set(ARM_ORDER), "QSpatial arm results")
    operational = _exact_mapping(
        score.get("operational_by_arm"), set(ARM_ORDER), "QSpatial operations"
    )
    scorer_artifacts = _exact_mapping(
        score.get("preregistered_project_scorer_artifacts"),
        set(ARM_ORDER),
        "QSpatial project scorer artifacts",
    )
    cluster_maps: dict[str, dict[str, str]] = {}
    cluster_values: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARM_ORDER:
        result = _exact_mapping(
            arms[arm],
            {
                "rows",
                "row_metrics",
                "cluster_metrics",
                "project_scorer_audit",
                "cluster_values",
                "preregistered_project_metrics",
                "preregistered_project_scorer_crosscheck",
            },
            f"QSpatial {arm} result",
        )
        rows = result["rows"]
        if not isinstance(rows, list) or len(rows) != QSPATIAL_ROWS:
            raise EvidenceError(f"QSpatial {arm} must contain exactly 97 scored rows")
        if [
            str(item.get("sample_id")) if isinstance(item, Mapping) else "" for item in rows
        ] != list(EXPECTED_IDS):
            raise EvidenceError(f"QSpatial {arm} scored rows are not exact ordered Gap-97")
        frozen_rows = journals[arm]
        by_id = {str(item["sample_id"]): item for item in frozen_rows}
        normalized_rows: list[Mapping[str, Any]] = []
        for item in rows:
            if not isinstance(item, Mapping) or set(item) != {
                "sample_id",
                "cluster_id",
                "prediction_text",
                "target",
                "sr_at_1_25",
                "sr_at_2",
                "capped_absolute_log_ratio",
                "relative_error",
            }:
                raise EvidenceError(f"QSpatial {arm} scored-row schema drifted")
            sample_id = str(item["sample_id"])
            cluster_id = item["cluster_id"]
            if (
                not isinstance(cluster_id, str)
                or not cluster_id
                or not isinstance(item["prediction_text"], str)
                or not item["prediction_text"].strip()
                or not isinstance(item["target"], str)
                or not item["target"].strip()
            ):
                raise EvidenceError(
                    f"QSpatial {arm} row {sample_id} has incomplete text or cluster data"
                )
            if (
                str(by_id[sample_id].get("cluster_id")) != cluster_id
                or by_id[sample_id].get("prediction_text") != item["prediction_text"]
            ):
                raise EvidenceError(
                    f"QSpatial {arm} row {sample_id} differs from its sealed journal"
                )
            sr125 = _probability(item["sr_at_1_25"], f"QSpatial {arm} row {sample_id} SR@1.25")
            sr2 = _probability(item["sr_at_2"], f"QSpatial {arm} row {sample_id} SR@2")
            if sr125 not in (0.0, 1.0) or sr2 not in (0.0, 1.0) or sr125 > sr2:
                raise EvidenceError(
                    f"QSpatial {arm} row {sample_id} has invalid success indicators"
                )
            loss = _finite(
                item["capped_absolute_log_ratio"],
                f"QSpatial {arm} row {sample_id} log ratio",
                minimum=0.0,
            )
            if loss > math.log(10.0) + 1e-12:
                raise EvidenceError(f"QSpatial {arm} row {sample_id} exceeds capped log-ratio loss")
            if item["relative_error"] is not None:
                _finite(
                    item["relative_error"],
                    f"QSpatial {arm} row {sample_id} relative error",
                    minimum=0.0,
                )
            normalized_rows.append(item)
        current_map = {str(item["sample_id"]): str(item["cluster_id"]) for item in normalized_rows}
        if len(set(current_map.values())) != QSPATIAL_CLUSTERS:
            raise EvidenceError(f"QSpatial {arm} does not contain 54 clusters")
        cluster_maps[arm] = current_map
        row_metrics = _exact_mapping(
            result["row_metrics"],
            {
                "sr_at_1_25",
                "sr_at_2",
                "mean_capped_absolute_log_ratio",
                "mre",
                "valid_prediction_mre",
                "median_absolute_log_ratio",
                "valid_prediction_median_absolute_log_ratio",
                "valid_predictions",
                "grounding_failures",
                "grounding_failure_rate",
                "failure_reasons",
                "ground_truth_unit_counts",
                "prediction_unit_counts",
            },
            f"QSpatial {arm} row metrics",
        )
        valid_rows = [item for item in normalized_rows if item["relative_error"] is not None]
        valid = len(valid_rows)
        failures = QSPATIAL_ROWS - valid
        if (
            _integer(row_metrics["valid_predictions"], f"QSpatial {arm} valid predictions") != valid
            or _integer(row_metrics["grounding_failures"], f"QSpatial {arm} failures") != failures
            or not _same_number(row_metrics["grounding_failure_rate"], failures / QSPATIAL_ROWS)
        ):
            raise EvidenceError(f"QSpatial {arm} grounding counts are inconsistent")
        failure_reasons = row_metrics["failure_reasons"]
        if (
            not isinstance(failure_reasons, Mapping)
            or any(
                not isinstance(name, str) or _integer(count, f"QSpatial {arm} failure count") < 0
                for name, count in failure_reasons.items()
            )
            or sum(failure_reasons.values()) != failures
        ):
            raise EvidenceError(f"QSpatial {arm} failure reasons do not reconcile")
        for name, expected_total in (
            ("ground_truth_unit_counts", QSPATIAL_ROWS),
            ("prediction_unit_counts", valid),
        ):
            counts = row_metrics[name]
            if (
                not isinstance(counts, Mapping)
                or any(
                    not isinstance(unit, str)
                    or not unit
                    or _integer(count, f"QSpatial {arm} {name}") < 0
                    for unit, count in counts.items()
                )
                or sum(counts.values()) != expected_total
            ):
                raise EvidenceError(f"QSpatial {arm} {name} do not reconcile")
        fields = {
            "sr_at_1_25": "sr_at_1_25",
            "sr_at_2": "sr_at_2",
            "mean_capped_absolute_log_ratio": "capped_absolute_log_ratio",
        }
        calculated_clusters: dict[str, dict[str, float]] = {}
        for aggregate_name, item_name in fields.items():
            row_mean = math.fsum(float(item[item_name]) for item in normalized_rows) / QSPATIAL_ROWS
            if not _same_number(row_metrics[aggregate_name], row_mean):
                raise EvidenceError(f"QSpatial {arm} row aggregate {aggregate_name} drifted")
            calculated_clusters[item_name] = _cluster_means(normalized_rows, item_name)
        cluster_metrics = _exact_mapping(
            result["cluster_metrics"],
            {"cluster_count", "sr_at_1_25", "sr_at_2", "mean_capped_absolute_log_ratio"},
            f"QSpatial {arm} cluster metrics",
        )
        if cluster_metrics["cluster_count"] != QSPATIAL_CLUSTERS:
            raise EvidenceError(f"QSpatial {arm} cluster count drifted")
        for aggregate_name, item_name in fields.items():
            expected_mean = math.fsum(calculated_clusters[item_name].values()) / QSPATIAL_CLUSTERS
            if not _same_number(cluster_metrics[aggregate_name], expected_mean):
                raise EvidenceError(f"QSpatial {arm} cluster aggregate {aggregate_name} drifted")
        claimed_cluster_values = _exact_mapping(
            result["cluster_values"],
            {"sr_at_1_25", "sr_at_2", "capped_absolute_log_ratio"},
            f"QSpatial {arm} cluster values",
        )
        for item_name, expected_values in calculated_clusters.items():
            if not _numbers_equal_structure(claimed_cluster_values[item_name], expected_values):
                raise EvidenceError(f"QSpatial {arm} cluster values {item_name} drifted")
        cluster_values[arm] = calculated_clusters
        valid_errors = [float(item["relative_error"]) for item in valid_rows]
        expected_valid_mre = statistics.fmean(valid_errors) if valid_errors else None
        if not _same_optional_number(
            row_metrics["valid_prediction_mre"], expected_valid_mre
        ) or not _same_optional_number(
            row_metrics["mre"], expected_valid_mre if valid == QSPATIAL_ROWS else None
        ):
            raise EvidenceError(f"QSpatial {arm} MRE fields drifted")
        for name in ("median_absolute_log_ratio", "valid_prediction_median_absolute_log_ratio"):
            value = row_metrics[name]
            if value is not None:
                _finite(value, f"QSpatial {arm} {name}", minimum=0.0)
        if (valid != QSPATIAL_ROWS) != (row_metrics["median_absolute_log_ratio"] is None) or (
            valid == 0
        ) != (row_metrics["valid_prediction_median_absolute_log_ratio"] is None):
            raise EvidenceError(f"QSpatial {arm} median log-ratio nullability drifted")
        audit = result["project_scorer_audit"]
        if (
            not isinstance(audit, Mapping)
            or audit.get("total_samples") != QSPATIAL_ROWS
            or audit.get("cluster_count") != QSPATIAL_CLUSTERS
            or audit.get("valid_predictions") != valid
        ):
            raise EvidenceError(f"QSpatial {arm} project scorer audit drifted")
        crosscheck = _exact_mapping(
            result["preregistered_project_scorer_crosscheck"],
            {"status", "fields"},
            f"QSpatial {arm} scorer crosscheck",
        )
        if (
            crosscheck["status"] != "pass"
            or not isinstance(crosscheck["fields"], Mapping)
            or set(crosscheck["fields"]) != QSPATIAL_CROSSCHECK_FIELDS
            or not all(value is True for value in crosscheck["fields"].values())
        ):
            raise EvidenceError(f"QSpatial {arm} project scorer crosscheck is incomplete")
        scorer_path = _verify_artifact(
            scorer_artifacts[arm], f"QSpatial {arm} project scorer result"
        )
        project_result = _load_json(scorer_path, f"QSpatial {arm} project scorer result")
        if result["preregistered_project_metrics"] != {
            key: project_result.get(key) for key in result["preregistered_project_metrics"]
        }:
            raise EvidenceError(f"QSpatial {arm} project scorer artifact does not bind its metrics")
        recalculated_op = _operational_from_journal(frozen_rows, arm)
        if not _numbers_equal_structure(
            operational[arm], recalculated_op
        ) or not _numbers_equal_structure(seal["operational_by_arm"][arm], recalculated_op):
            raise EvidenceError(f"QSpatial {arm} operational summary differs from its journal")
        verified += 1
    if any(cluster_maps[arm] != cluster_maps[ARM_ORDER[0]] for arm in ARM_ORDER[1:]):
        raise EvidenceError("QSpatial arms do not share the exact sample-to-cluster map")
    expected_comparisons = {
        "metric_correct_minus_question_only": ("metric_correct", "question_only"),
        "metric_correct_minus_metric_shuffled": ("metric_correct", "metric_shuffled"),
        "metric_correct_minus_scenegraph": ("metric_correct", "scenegraph"),
        "metric_correct_minus_native": ("metric_correct", "native"),
    }
    recomputed: dict[str, Any] = {}
    for name, (left, right) in expected_comparisons.items():
        recomputed[name] = {
            metric: _paired_bootstrap(cluster_values[left][metric], cluster_values[right][metric])
            for metric in ("sr_at_1_25", "sr_at_2", "capped_absolute_log_ratio")
        }
    if not _numbers_equal_structure(score.get("paired_cluster_comparisons"), recomputed):
        raise EvidenceError(
            "QSpatial paired cluster comparisons differ from independent recomputation"
        )
    hidden = _exact_mapping(
        score.get("hidden_fallback_counts"), set(ARM_ORDER), "QSpatial fallback counts"
    )
    if any(_integer(hidden[arm], f"QSpatial {arm} fallback count") < 0 for arm in ARM_ORDER):
        raise EvidenceError("QSpatial fallback counts are invalid")
    if dict(hidden) != sealed_hidden_fallback_counts:
        raise EvidenceError("QSpatial fallback counts differ from the sealed arm gates")
    visual = (
        recomputed["metric_correct_minus_question_only"]["sr_at_1_25"]["ci"]["lower"] > 0.0
        and recomputed["metric_correct_minus_metric_shuffled"]["sr_at_1_25"]["ci"]["lower"] > 0.0
        and hidden["metric_correct"] == 0
    )
    decisive = visual and all(
        recomputed[name]["sr_at_1_25"]["ci"]["lower"] > 0.0
        and recomputed[name]["sr_at_2"]["estimate"] > 0.0
        and recomputed[name]["capped_absolute_log_ratio"]["estimate"] < 0.0
        for name in ("metric_correct_minus_scenegraph", "metric_correct_minus_native")
    )
    if (
        claims["visual_claim_supported"] is not visual
        or claims["decisive_win_supported"] is not decisive
    ):
        raise EvidenceError(
            "QSpatial claim verdict differs from independent preregistered recomputation"
        )
    return score, verified


def _validate_vsi(path: Path, qspatial_path: Path) -> tuple[dict[str, Any], int]:
    gate = _load_json(path, "VSI full gate")
    expected_gate_fields = {
        "schema_version",
        "kind",
        "status",
        "validated_at_utc",
        "benchmark",
        "expected_count",
        "successful_unique_predictions",
        "attempt_count",
        "interrupted_attempt_count",
        "terminal_counter_totals",
        "labels_loaded_only_for_final_official_scoring",
        "official_summary",
        "p3_claim_gates",
        "artifacts",
    }
    if set(gate) != expected_gate_fields:
        raise EvidenceError("VSI full gate top-level schema drifted")
    _required_identity(
        gate,
        {
            "schema_version": 1,
            "kind": "p4_vsibench_unbiased_full_gate",
            "status": "pass",
            "benchmark": "vsibench_unbiased",
            "expected_count": VSI_ROWS,
            "successful_unique_predictions": VSI_ROWS,
            "labels_loaded_only_for_final_official_scoring": True,
        },
        "VSI full gate",
    )
    artifacts = _exact_mapping(
        gate.get("artifacts"), VSI_GATE_ARTIFACT_NAMES, "VSI full gate artifacts"
    )
    bound_p3 = _verify_artifact(artifacts.get("p3_offline_score"), "VSI predecessor QSpatial")
    if bound_p3 != qspatial_path.expanduser().resolve():
        raise EvidenceError("VSI gate is bound to a different QSpatial offline score")
    protocol_path = _verify_artifact(artifacts["protocol"], "VSI frozen protocol")
    protocol = _load_json(protocol_path, "VSI frozen protocol")
    _required_identity(
        protocol,
        {
            "schema_version": 1,
            "kind": "p4_vsibench_unbiased_full_protocol",
            "benchmark": "vsibench_unbiased",
            "expected_count": VSI_ROWS,
            "expected_unique_ids": VSI_ROWS,
            "ids_sha256": VSI_IDS_SHA256,
            "source_parquet_sha256": VSI_PARQUET_SHA256,
            "concurrency": 1,
            "maximum_chunk_size": 50,
            "video_max_fps": 1.0,
            "question_type_counts": VSI_QUESTION_COUNTS,
            "resume_policy": "accounted_interruption_only",
            "final_scoring": "official_spatialclaw_vsibench_evaluator",
        },
        "VSI frozen protocol",
    )
    ids_path = _verify_artifact(artifacts["frozen_ids"], "VSI frozen IDs")
    ids_value = _load_json(ids_path, "VSI frozen IDs")
    ids = ids_value.get("ids")
    _required_identity(
        ids_value,
        {
            "schema_version": 1,
            "kind": "p4_vsibench_unbiased_ids",
            "benchmark": "vsibench_unbiased",
            "count": VSI_ROWS,
            "unique_count": VSI_ROWS,
            "ids_sha256": VSI_IDS_SHA256,
            "source_parquet_sha256": VSI_PARQUET_SHA256,
        },
        "VSI frozen IDs",
    )
    if (
        not isinstance(ids, list)
        or len(ids) != VSI_ROWS
        or len(set(ids)) != VSI_ROWS
        or any(not isinstance(item, str) or not item for item in ids)
        or _sha256_bytes(_canonical_id_bytes(ids)) != VSI_IDS_SHA256
    ):
        raise EvidenceError("VSI frozen IDs are not the exact canonical sequence")
    predictions_path = _verify_artifact(artifacts["predictions"], "VSI predictions")
    prediction_rows = _load_jsonl(predictions_path, "VSI predictions")
    if [str(row.get("sample_id")) for row in prediction_rows] != ids or any(
        not isinstance(row.get("content"), str)
        or not row["content"].strip()
        or row.get("error") not in (None, "")
        for row in prediction_rows
    ):
        raise EvidenceError("VSI predictions do not contain exact successful frozen IDs")
    journal_call_total = 0
    for index, row in enumerate(prediction_rows, 1):
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            raise EvidenceError(f"VSI prediction row {index} usage is missing")
        journal_call_total += _integer(
            usage.get("num_calls"), f"VSI prediction row {index} usage.num_calls", minimum=1
        )
    accounting_path = _verify_artifact(artifacts["full_accounting"], "VSI full accounting")
    accounting = _load_json(accounting_path, "VSI full accounting")
    _required_identity(
        accounting,
        {
            "schema_version": 1,
            "kind": "p3_full_vllm_accounting",
            "status": "pass",
            "benchmark": "vsibench_unbiased",
            "row_format": "native",
            "expected_count": VSI_ROWS,
            "concurrency": 1,
            "requires_gpu": True,
            "completed_ids": ids,
            "predictions": artifacts["predictions"],
            "journaled_call_total": journal_call_total,
        },
        "VSI full accounting",
    )
    statuses = accounting.get("attempt_statuses")
    if (
        not isinstance(statuses, list)
        or not statuses
        or statuses[-1] != "pass"
        or any(status != "accounted_interrupted" for status in statuses[:-1])
    ):
        raise EvidenceError("VSI accounting attempt statuses are invalid")
    minimum_attempt_count = math.ceil(VSI_ROWS / protocol["maximum_chunk_size"])
    if len(statuses) < minimum_attempt_count:
        raise EvidenceError(
            f"VSI accounting must contain at least {minimum_attempt_count} sequential attempts"
        )
    terminal = _terminal_counters(
        accounting.get("terminal_counter_totals"), "VSI aggregate terminal counters"
    )
    if any(terminal[reason] != 0 for reason in VSI_ABNORMAL_REASONS):
        raise EvidenceError("VSI accounting has abnormal terminal counters")
    if terminal["stop"] != journal_call_total:
        raise EvidenceError("VSI aggregate stop counter differs from summed journal calls")
    if gate.get("terminal_counter_totals") != terminal:
        raise EvidenceError("VSI gate terminal counters differ from full accounting")
    attempt_records = artifacts["attempts"]
    if not isinstance(attempt_records, list) or not attempt_records:
        raise EvidenceError("VSI gate has no named attempt chain")
    if len(attempt_records) != len(statuses):
        raise EvidenceError("VSI attempt chain differs from accounting statuses")
    selected_ids: list[str] = []
    seen_service_jobs: set[str] = set()
    summed_deltas = {reason: 0 for reason in VSI_TERMINAL_REASONS}
    for index, record in enumerate(attempt_records):
        attempt_path = _verify_artifact(record, f"VSI attempt {index + 1}")
        attempt = _load_json(attempt_path, f"VSI attempt {index + 1}")
        expected_status = "pass" if index == len(attempt_records) - 1 else "accounted_interrupted"
        expected_final = index == len(attempt_records) - 1
        if (
            attempt.get("kind") != "p4_vsibench_unbiased_attempt_gate"
            or attempt.get("status") != expected_status
            or statuses[index] != expected_status
            or attempt.get("final_attempt") is not expected_final
        ):
            raise EvidenceError(f"VSI attempt {index + 1} status or kind drifted")
        nested = _exact_mapping(
            attempt.get("artifacts"),
            VSI_ATTEMPT_ARTIFACT_NAMES,
            f"VSI attempt {index + 1} artifacts",
        )
        recorded_nested = _exact_mapping(
            record.get("nested_artifacts"),
            VSI_ATTEMPT_ARTIFACT_NAMES,
            f"VSI attempt {index + 1} recorded artifacts",
        )
        for name in VSI_ATTEMPT_ARTIFACT_NAMES:
            _verify_artifact(nested[name], f"VSI attempt {index + 1} {name}")
            if nested[name] != recorded_nested[name]:
                raise EvidenceError(f"VSI attempt {index + 1} nested artifact binding drifted")
        if nested["p4_smoke_gate"] != artifacts["p4_smoke_gate"]:
            raise EvidenceError(f"VSI attempt {index + 1} binds a different P4 smoke gate")
        if record.get("immutable_journal_after") != nested["immutable_journal_after"]:
            raise EvidenceError(f"VSI attempt {index + 1} journal snapshot binding drifted")
        selected = attempt.get("selected_ids")
        if (
            not isinstance(selected, list)
            or not 1 <= len(selected) <= protocol["maximum_chunk_size"]
            or any(not isinstance(item, str) or not item for item in selected)
        ):
            raise EvidenceError(f"VSI attempt {index + 1} selected IDs are invalid")
        if attempt.get("newly_completed_count") != len(selected):
            raise EvidenceError(f"VSI attempt {index + 1} selected-ID count drifted")
        deltas = _terminal_counters(
            attempt.get("terminal_counter_deltas"),
            f"VSI attempt {index + 1} terminal counters",
        )
        journaled_calls = _integer(
            attempt.get("journaled_call_delta"),
            f"VSI attempt {index + 1} journaled call delta",
        )
        if any(deltas[reason] != 0 for reason in VSI_ABNORMAL_REASONS):
            raise EvidenceError(f"VSI attempt {index + 1} has abnormal terminal counters")
        if deltas["stop"] != journaled_calls:
            raise EvidenceError(f"VSI attempt {index + 1} stop counter differs from journal calls")
        for reason in VSI_TERMINAL_REASONS:
            summed_deltas[reason] += deltas[reason]
        service = _exact_mapping(
            attempt.get("service"),
            VSI_SERVICE_FIELDS,
            f"VSI attempt {index + 1} service identity",
        )
        _verify_artifact(service["endpoint_manifest"], f"VSI attempt {index + 1} endpoint manifest")
        if (
            service["endpoint_model"] != "qwen3.6-27b"
            or service["endpoint_reasoning_parser"] != "qwen3"
            or any(
                not isinstance(service[field], str) or not service[field].strip()
                for field in ("endpoint_url", "metrics_url")
            )
        ):
            raise EvidenceError(f"VSI attempt {index + 1} service identity drifted")
        vllm_job = service["vllm_job_id"]
        gpu_job = service["gpu_tool_job_id"]
        if not _ascii_decimal_job_id(vllm_job) or not _ascii_decimal_job_id(gpu_job):
            raise EvidenceError(f"VSI attempt {index + 1} service job IDs are invalid")
        if vllm_job == gpu_job:
            raise EvidenceError(f"VSI attempt {index + 1} reuses one job for both services")
        if vllm_job in seen_service_jobs or gpu_job in seen_service_jobs:
            raise EvidenceError("VSI attempts do not use distinct fresh service pairs")
        seen_service_jobs.update((vllm_job, gpu_job))
        selected_ids.extend(selected)
    if selected_ids != ids:
        raise EvidenceError("VSI attempt chain does not cover the exact frozen IDs")
    if summed_deltas != terminal:
        raise EvidenceError("VSI per-attempt terminal counters do not sum to aggregate totals")
    official_path = _verify_artifact(artifacts.get("official_results"), "VSI official results")
    official = _load_json(official_path, "VSI official results")
    detailed = official.get("detailed_results")
    if not isinstance(detailed, list) or len(detailed) != VSI_ROWS:
        raise EvidenceError("VSI official results do not contain all 2362 rows")
    if [str(row.get("id")) if isinstance(row, Mapping) else "" for row in detailed] != ids:
        raise EvidenceError("VSI official result IDs differ from the frozen order")
    scores: dict[str, list[float]] = {name: [] for name in VSI_QUESTION_COUNTS}
    for index, row in enumerate(detailed):
        if not isinstance(row, Mapping) or set(row) != {
            "id",
            "question_type",
            "ground_truth",
            "prediction",
            "extracted",
            "score",
        }:
            raise EvidenceError(f"VSI official detail row {index + 1} schema drifted")
        question_type = row["question_type"]
        if question_type not in VSI_QUESTION_COUNTS:
            raise EvidenceError(f"VSI official detail row {index + 1} has an unknown question type")
        scores[question_type].append(
            _probability(row["score"], f"VSI detail row {index + 1} score")
        )
    if {name: len(values) for name, values in scores.items()} != VSI_QUESTION_COUNTS:
        raise EvidenceError("VSI detailed question-type counts differ from the frozen protocol")
    task_scores = {name: statistics.fmean(values) for name, values in scores.items()}
    directions = [
        task_scores.pop(name)
        for name in (
            "object_rel_direction_easy",
            "object_rel_direction_medium",
            "object_rel_direction_hard",
        )
    ]
    task_scores["object_rel_direction"] = statistics.fmean(directions)
    expected_per_task = {
        name: {
            "score": value * 100.0,
            "count": sum(
                VSI_QUESTION_COUNTS[item]
                for item in (
                    "object_rel_direction_easy",
                    "object_rel_direction_medium",
                    "object_rel_direction_hard",
                )
            )
            if name == "object_rel_direction"
            else VSI_QUESTION_COUNTS[name],
        }
        for name, value in task_scores.items()
    }
    expected_overall = statistics.fmean(task_scores.values())
    expected_summary = {
        "total_samples": VSI_ROWS,
        "correct_samples": sum(1 for row in detailed if float(row["score"]) >= 1.0),
        "overall_accuracy": expected_overall,
        "overall_accuracy_pct": expected_overall * 100.0,
        "per_task_scores": expected_per_task,
    }
    summary = {key: value for key, value in official.items() if key != "detailed_results"}
    if not _numbers_equal_structure(summary, expected_summary) or not _numbers_equal_structure(
        gate.get("official_summary"), expected_summary
    ):
        raise EvidenceError("VSI official summary differs from independent score recomputation")
    qspatial_claims = _load_json(qspatial_path, "VSI predecessor QSpatial").get("claim_gates")
    if gate.get("p3_claim_gates") != qspatial_claims:
        raise EvidenceError("VSI gate does not preserve the exact P3 claim verdicts")
    if gate.get("attempt_count") != len(attempt_records) or gate.get(
        "interrupted_attempt_count"
    ) != statuses.count("accounted_interrupted"):
        raise EvidenceError("VSI gate attempt counts differ from accounting")
    transitive = _verify_transitive_artifacts(gate, "VSI gate")
    return gate, transitive


def _validate_397b(path: Path) -> tuple[dict[str, Any], int]:
    value = _load_json(path, "397B evidence")
    _required_identity(
        value,
        {
            "schema_version": 1,
            "kind": "p4_397b_evidence",
            "status": "pass",
            "slurm_state": "COMPLETED",
            "benchmark": "erqa",
            "scope": "subsample5_diagnostic_transfer",
            "job_id": EVIDENCE_397B_JOB_ID,
            "backbone": EVIDENCE_397B_BACKBONE,
            "expected_count": 5,
            "successful_count": 5,
            "correct_count": 3,
            "score": 0.6,
            "claim_bearing": False,
        },
        "397B evidence",
    )
    artifacts = _exact_mapping(
        value.get("artifacts"), set(EVIDENCE_397B_ARTIFACT_SHA256), "397B evidence artifacts"
    )
    for name, expected_sha256 in EVIDENCE_397B_ARTIFACT_SHA256.items():
        artifact_path = _verify_artifact(artifacts[name], f"397B {name}")
        if _sha256(artifact_path) != expected_sha256:
            raise EvidenceError(f"397B {name} differs from the frozen historical artifact")
    result = _load_json(_verify_artifact(artifacts["result"], "397B result"), "397B result")
    if (
        result.get("total_samples") != 5
        or result.get("correct_samples") != 3
        or not _same_number(result.get("overall_accuracy"), 0.6)
    ):
        raise EvidenceError("397B result artifact does not report frozen 3/5 evidence")
    predictions = _load_jsonl(
        _verify_artifact(artifacts["predictions"], "397B predictions"), "397B predictions"
    )
    if len(predictions) != 5 or len({str(row.get("sample_id")) for row in predictions}) != 5:
        raise EvidenceError("397B predictions artifact does not contain five unique rows")
    transitive = len(artifacts)
    return value, transitive


def _format_optional(value: object, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _arms_csv(score: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "arm",
        "label",
        "rows",
        "valid_predictions",
        "grounding_failures",
        "row_sr_at_1_25",
        "row_sr_at_2",
        "cluster_sr_at_1_25",
        "cluster_sr_at_2",
        "cluster_mean_capped_absolute_log_ratio",
        "valid_prediction_mre",
        "mean_latency_seconds",
        "p95_latency_seconds",
        "total_calls",
        "total_tokens",
        "estimated_cost_usd",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for arm in ARM_ORDER:
        result = score["arm_results"][arm]
        row = result["row_metrics"]
        cluster = result["cluster_metrics"]
        op = score["operational_by_arm"][arm]
        writer.writerow(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "rows": QSPATIAL_ROWS,
                "valid_predictions": row["valid_predictions"],
                "grounding_failures": row["grounding_failures"],
                "row_sr_at_1_25": f"{row['sr_at_1_25']:.10f}",
                "row_sr_at_2": f"{row['sr_at_2']:.10f}",
                "cluster_sr_at_1_25": f"{cluster['sr_at_1_25']:.10f}",
                "cluster_sr_at_2": f"{cluster['sr_at_2']:.10f}",
                "cluster_mean_capped_absolute_log_ratio": (
                    f"{cluster['mean_capped_absolute_log_ratio']:.10f}"
                ),
                "valid_prediction_mre": _format_optional(row["valid_prediction_mre"], 10),
                "mean_latency_seconds": f"{op['latency_seconds']['mean']:.6f}",
                "p95_latency_seconds": f"{op['latency_seconds']['p95_linear']:.6f}",
                "total_calls": op["total_calls"],
                "total_tokens": op["total_tokens"],
                "estimated_cost_usd": _format_optional(op["estimated_cost_usd"], 6),
            }
        )
    return stream.getvalue()


def _comparisons_csv(score: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = ["comparison", "metric", "estimate", "ci_lower", "ci_upper", "clusters", "iterations"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for comparison in COMPARISON_ORDER:
        for metric in ("sr_at_1_25", "sr_at_2", "capped_absolute_log_ratio"):
            value = score["paired_cluster_comparisons"][comparison][metric]
            writer.writerow(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "estimate": f"{value['estimate']:.10f}",
                    "ci_lower": f"{value['ci']['lower']:.10f}",
                    "ci_upper": f"{value['ci']['upper']:.10f}",
                    "clusters": QSPATIAL_CLUSTERS,
                    "iterations": value["iterations"],
                }
            )
    return stream.getvalue()


def _verdict(supported: bool, description: str) -> str:
    return (
        f"SUPPORTED: {description}"
        if supported
        else f"NOT SUPPORTED: negative result for {description}"
    )


def _poster_markdown(
    score: Mapping[str, Any], vsi: Mapping[str, Any], evidence_397b: Mapping[str, Any] | None
) -> str:
    claims = score["claim_gates"]
    lines = [
        "# Spatial Atlas benchmark results",
        "",
        "## QSpatial++ Gap-97",
        "",
        "| System | n | Valid | Failures | Cluster SR@1.25 | Cluster SR@2 | Log-ratio error | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        result = score["arm_results"][arm]
        row = result["row_metrics"]
        cluster = result["cluster_metrics"]
        latency = score["operational_by_arm"][arm]["latency_seconds"]["mean"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {QSPATIAL_ROWS} | {row['valid_predictions']} | "
            f"{row['grounding_failures']} | {cluster['sr_at_1_25']:.3f} | "
            f"{cluster['sr_at_2']:.3f} | {cluster['mean_capped_absolute_log_ratio']:.3f} | "
            f"{latency:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered claim verdicts",
            "",
            _verdict(claims["visual_claim_supported"], "the visual-grounding claim"),
            "",
            _verdict(claims["decisive_win_supported"], "the decisive-win claim"),
            "",
            "False gates are negative results and must remain visible in every derived writeup.",
            "",
            "## Transfer evidence",
            "",
        ]
    )
    summary = vsi["official_summary"]
    lines.append(
        f"VSI-Bench unbiased full run: VERIFIED, n={VSI_ROWS}, official mean task score="
        f"{summary['overall_accuracy']:.4f}."
    )
    lines.append("")
    if evidence_397b is None:
        lines.append(
            "397B ERQA diagnostic: PENDING OR BLOCKED. No verified evidence metadata was supplied."
        )
    else:
        lines.append(
            f"397B ERQA diagnostic: VERIFIED, job {evidence_397b['job_id']}, "
            f"{evidence_397b['correct_count']}/{evidence_397b['expected_count']} "
            f"(score {evidence_397b['score']:.4f}). This diagnostic is non-claim-bearing."
        )
    lines.extend(
        [
            "",
            "Metric provenance: QSpatial++ uses the sealed project scoring contract. "
            "VSI-Bench, when present, uses the SpatialClaw official evaluator.",
            "",
        ]
    )
    return "\n".join(lines)


def _paper_markdown(
    score: Mapping[str, Any], vsi: Mapping[str, Any], evidence_397b: Mapping[str, Any] | None
) -> str:
    metric = score["arm_results"]["metric_correct"]
    row = metric["row_metrics"]
    cluster = metric["cluster_metrics"]
    claims = score["claim_gates"]
    lines = [
        "## Machine-derived results snippet",
        "",
        "We evaluated five preregistered arms on the sealed QSpatial++ Gap-97 slice "
        f"({QSPATIAL_ROWS} questions, {QSPATIAL_CLUSTERS} question clusters). The correct-image "
        f"metric arm achieved cluster-balanced SR@1.25={cluster['sr_at_1_25']:.4f} and "
        f"SR@2={cluster['sr_at_2']:.4f}, with {row['grounding_failures']} grounding failures.",
        "",
        _verdict(claims["visual_claim_supported"], "the preregistered visual-grounding claim")
        + ".",
        "",
        _verdict(claims["decisive_win_supported"], "the preregistered decisive-win claim") + ".",
        "",
    ]
    lines.append(
        f"On VSI-Bench unbiased, the verified full run scored {vsi['official_summary']['overall_accuracy']:.4f} "
        f"across {VSI_ROWS} samples using the official SpatialClaw evaluator."
    )
    lines.append("")
    if evidence_397b is None:
        lines.append("The 397B ERQA diagnostic is pending or blocked; no score is reported.")
    else:
        lines.append(
            f"A non-claim-bearing 397B ERQA subsample diagnostic completed "
            f"{evidence_397b['correct_count']}/{evidence_397b['expected_count']} items "
            f"(score {evidence_397b['score']:.4f}, Slurm job {evidence_397b['job_id']})."
        )
    lines.extend(
        [
            "",
            "All values in this snippet were rendered from hash-verified artifacts. "
            "A passing execution gate is not treated as evidence for a positive scientific claim.",
            "",
        ]
    )
    return "\n".join(lines)


def build_assets(
    qspatial_path: Path,
    output_dir: Path,
    *,
    vsi_path: Path,
    evidence_397b_path: Path | None = None,
) -> dict[str, Path]:
    if vsi_path is None:
        raise EvidenceError("P4 VSI full gate is required")
    qspatial_resolved = qspatial_path.expanduser().resolve()
    score, q_count = _validate_qspatial(qspatial_resolved)
    vsi, vsi_count = _validate_vsi(vsi_path, qspatial_resolved)
    evidence_397b = None
    evidence_count = 0
    if evidence_397b_path is not None:
        evidence_397b, evidence_count = _validate_397b(evidence_397b_path)

    destination = output_dir.expanduser().resolve()
    output_paths = {name: destination / name for name in OUTPUT_NAMES}
    rendered: dict[str, bytes] = {
        "p5_qspatial_arms.csv": _arms_csv(score).encode(),
        "p5_qspatial_comparisons.csv": _comparisons_csv(score).encode(),
        "p5_poster_results.md": _poster_markdown(score, vsi, evidence_397b).encode(),
        "p5_paper_results.md": _paper_markdown(score, vsi, evidence_397b).encode(),
    }
    inputs: dict[str, Any] = {
        "qspatial_offline_score": _public_artifact(
            qspatial_resolved, "qspatial_full_offline_score.json"
        )
    }
    inputs["p4_vsi_full_gate"] = _public_artifact(
        vsi_path.expanduser().resolve(), "p4_vsibench_unbiased_full_gate.json"
    )
    inputs["p4_397b_evidence"] = (
        _public_artifact(evidence_397b_path.expanduser().resolve(), "p4_397b_evidence.json")
        if evidence_397b_path is not None
        else {"status": "pending_or_blocked"}
    )
    manifest = {
        "schema_version": 1,
        "kind": "spatial_atlas_p5_results_assets",
        "status": "pass",
        "scientific_claim_verdicts": dict(score["claim_gates"]),
        "optional_evidence_status": {
            "p4_vsi_full_gate": "verified",
            "p4_397b_evidence": "verified" if evidence_397b is not None else "pending_or_blocked",
        },
        "verified_transitive_artifact_counts": {
            "qspatial": q_count,
            "vsi": vsi_count,
            "397b": evidence_count,
        },
        "inputs": inputs,
        "outputs": {
            name: {
                "logical_name": name,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
            for name, payload in rendered.items()
        },
    }
    manifest["manifest_payload_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    rendered["p5_results_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()

    destination_existed = destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise EvidenceError("P5 output destination is not a directory")
    lock = destination / ".p5-assets.lock"
    lock_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock, lock_flags, 0o600)
    except FileExistsError as exc:
        raise EvidenceError("another P5 asset publication is active") from exc
    os.close(lock_fd)
    written: list[Path] = []
    temporaries: list[Path] = []
    try:
        existing = [path.name for path in output_paths.values() if path.exists()]
        if existing:
            raise EvidenceError(f"refusing to overwrite existing P5 assets: {existing}")
        for name in OUTPUT_NAMES:
            path = output_paths[name]
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=destination, prefix=f".{path.name}.tmp-"
            )
            temporary = Path(temporary_name)
            temporaries.append(temporary)
            try:
                with os.fdopen(file_descriptor, "wb") as stream:
                    stream.write(rendered[name])
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            except FileExistsError as exc:
                raise EvidenceError(f"refusing to overwrite existing P5 asset: {name}") from exc
            finally:
                temporary.unlink(missing_ok=True)
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
        if not destination_existed:
            try:
                destination.rmdir()
            except OSError:
                pass
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qspatial-offline-score", type=Path, required=True)
    parser.add_argument("--p4-vsi-full-gate", type=Path, required=True)
    parser.add_argument("--p4-397b-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = build_assets(
            args.qspatial_offline_score,
            args.output_dir,
            vsi_path=args.p4_vsi_full_gate,
            evidence_397b_path=args.p4_397b_evidence,
        )
    except EvidenceError as exc:
        print(f"[fatal] {exc}")
        return 1
    for name in OUTPUT_NAMES:
        print(f"[ok] {outputs[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
