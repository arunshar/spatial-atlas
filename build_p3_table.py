"""Build a verified P3 comparison table from three benchmark run directories.

The builder treats ``predictions.jsonl`` as an append-only journal. It keeps the
latest row for each sample, rejects unresolved errors and empty answers, and
recomputes the benchmark score with SpatialClaw's evaluator. The stored
``results_summary.json`` must match that recomputation before any table is
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


PREDICTIONS_FILENAME = "predictions.jsonl"
SUMMARY_FILENAME = "results_summary.json"
MANIFEST_FILENAME = "run_manifest.json"
CONFIG_FILENAME = "config.json"
NATIVE_MANIFEST_FILENAME = "native_run_manifest.json"
MARKDOWN_FILENAME = "p3_results.md"
CSV_FILENAME = "p3_results.csv"
GATE_FILENAME = "p3_results_gate.json"
GATE_SCHEMA_VERSION = 1
DEFAULT_SELECTION_SEED = 42
ACCOUNTED_ATTEMPT_STATUSES = {"pass", "accounted_interrupted"}


@dataclass(frozen=True)
class SystemSpec:
    """Input metadata for one comparison system."""

    key: str
    label: str
    row_format: str
    run_dir: Path
    grounding: str
    backbone: str = "Qwen3.6-27B"
    question_slice: str = "Warehouse validation distance"
    engine: str | None = None
    endpoint_manifest: Path | None = None
    selection_seed: int = DEFAULT_SELECTION_SEED
    concurrency: int = 1
    accounting_path: Path | None = None


@dataclass(frozen=True)
class SystemResult:
    """Verified score and operational means for one system."""

    spec: SystemSpec
    samples: int
    official_score: float
    official_distance_mean_relative_error: float
    distance_relative_error_count: int
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_total_tokens: float
    mean_calls: float
    mean_latency_seconds: float
    backbone: str
    question_slice: str
    predictions_path: Path
    summary_path: Path
    accounting_path: Path
    accounted_attempts: int
    interrupted_attempts: int
    vllm_stop_calls: int
    vllm_abnormal_calls: int
    unattributed_stop_calls: int
    terminal_counter_totals: dict[str, int]
    journaled_call_total: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _question_types(values: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        normalized.extend(part.strip() for part in value.split(",") if part.strip())
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-count", required=True, type=_positive_int)
    parser.add_argument(
        "--atlas-concurrency",
        required=True,
        type=_positive_int,
        help="Required concurrency recorded by both Spatial Atlas run manifests.",
    )
    parser.add_argument(
        "--native-concurrency",
        required=True,
        type=_positive_int,
        help="Required concurrency recorded by the SpatialClaw native run manifest.",
    )
    parser.add_argument(
        "--question-type",
        dest="question_types",
        action="append",
        default=[],
        help="Repeat or pass a comma-separated list.",
    )
    parser.add_argument("--scenegraph-dir", required=True)
    parser.add_argument("--metric-dir", required=True)
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--scenegraph-accounting", required=True)
    parser.add_argument("--metric-accounting", required=True)
    parser.add_argument("--native-accounting", required=True)
    parser.add_argument("--scenegraph-grounding", required=True)
    parser.add_argument("--metric-grounding", required=True)
    parser.add_argument("--native-grounding", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--question-slice", required=True)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
        help=f"Deterministic subsample seed (default: {DEFAULT_SELECTION_SEED}).",
    )
    parser.add_argument(
        "--native-endpoint-manifest",
        required=True,
        help="vLLM endpoint control JSON used by the native run.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--spatialclaw-root",
        default=os.environ.get("SPATIALCLAW_ROOT"),
        help="SpatialClaw checkout root, or set SPATIALCLAW_ROOT.",
    )
    args = parser.parse_args(argv)
    if not args.spatialclaw_root:
        parser.error("--spatialclaw-root or SPATIALCLAW_ROOT is required")
    args.question_types = _question_types(args.question_types)
    return args


def _load_benchmark(
    spatialclaw_root: str | os.PathLike[str],
    benchmark_name: str,
    data_root: str | os.PathLike[str],
    question_types: Sequence[str] | None,
) -> Any:
    checkout = Path(spatialclaw_root).expanduser().resolve()
    factory_path = checkout / "spatial_agent" / "evals" / "factory.py"
    if not factory_path.is_file():
        raise FileNotFoundError(f"SpatialClaw factory not found: {factory_path}")

    checkout_string = str(checkout)
    while checkout_string in sys.path:
        sys.path.remove(checkout_string)
    sys.path.insert(0, checkout_string)
    factory_module = importlib.import_module("spatial_agent.evals.factory")
    benchmark = factory_module.BenchmarkFactory.create_benchmark(
        benchmark_name,
        data_root=str(Path(data_root).expanduser().resolve()),
        question_type=list(question_types) or None,
    )
    if benchmark is None:
        raise ValueError(f"benchmark {benchmark_name!r} resolved to None")
    return benchmark


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON at {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object: {path}")
    return value


def _model_identity(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    leaf = value.strip().rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", leaf.lower())


def _load_latest_rows(run_dir: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    predictions_path = run_dir.expanduser().resolve() / PREDICTIONS_FILENAME
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions journal not found: {predictions_path}")

    latest: dict[str, dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError(f"{predictions_path}:{line_number} is blank")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{predictions_path}:{line_number} has invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{predictions_path}:{line_number} is not a JSON object")
            sample_id = row.get("sample_id")
            if sample_id is None or not str(sample_id).strip():
                raise ValueError(f"{predictions_path}:{line_number} has no sample_id")
            latest[str(sample_id)] = row
    return predictions_path, latest


def _answer(row: Mapping[str, Any], row_format: str) -> str:
    if row_format == "atlas":
        value = row.get("answer")
    elif row_format == "native":
        value = row.get("content")
    else:
        raise ValueError(f"unsupported row format: {row_format}")
    return "" if value is None else str(value).strip()


def _validated_rows(
    spec: SystemSpec, expected_count: int
) -> tuple[Path, dict[str, dict[str, Any]]]:
    predictions_path, rows = _load_latest_rows(spec.run_dir)
    if len(rows) != expected_count:
        raise ValueError(
            f"{spec.label} has {len(rows)} latest rows, expected {expected_count}: "
            f"{predictions_path}"
        )
    for sample_id, row in rows.items():
        if row.get("error") not in (None, ""):
            raise ValueError(f"{spec.label} sample {sample_id} has error: {row['error']}")
        if not _answer(row, spec.row_format):
            raise ValueError(f"{spec.label} sample {sample_id} has a blank answer")
        if spec.row_format == "atlas":
            if spec.engine not in {"scenegraph", "metric"}:
                raise ValueError(f"{spec.label} requires an explicit Atlas engine")
            if row.get("requested_engine") != spec.engine:
                raise ValueError(
                    f"{spec.label} sample {sample_id} requested_engine does not match {spec.engine}"
                )
            if row.get("used_engine") != spec.engine:
                raise ValueError(
                    f"{spec.label} sample {sample_id} used_engine does not match {spec.engine}"
                )
            if spec.engine == "metric" and row.get("grounding_source") != spec.grounding:
                raise ValueError(
                    f"{spec.label} sample {sample_id} grounding_source does not match "
                    "the declared grounding"
                )
    return predictions_path, rows


def _finite_nonnegative(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{description} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{description} must be finite and nonnegative")
    return converted


def _row_metrics(row: Mapping[str, Any], row_format: str, description: str) -> tuple[float, ...]:
    usage = row.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError(f"{description} usage must be a JSON object")

    if row_format == "atlas":
        prompt = _finite_nonnegative(usage.get("prompt_tokens"), f"{description} prompt_tokens")
        completion = _finite_nonnegative(
            usage.get("completion_tokens"), f"{description} completion_tokens"
        )
        total = _finite_nonnegative(usage.get("total_tokens"), f"{description} total_tokens")
        if not math.isclose(total, prompt + completion, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{description} total_tokens must equal prompt_tokens plus completion_tokens"
            )
        latency_key = "latency_seconds"
    elif row_format == "native":
        prompt = _finite_nonnegative(
            usage.get("total_prompt_tokens"), f"{description} total_prompt_tokens"
        )
        completion = _finite_nonnegative(
            usage.get("total_completion_tokens"), f"{description} total_completion_tokens"
        )
        total = prompt + completion
        latency_key = "latency_sec"
    else:
        raise ValueError(f"unsupported row format: {row_format}")

    calls = _finite_nonnegative(usage.get("num_calls"), f"{description} num_calls")
    latency = _finite_nonnegative(row.get(latency_key), f"{description} {latency_key}")
    return prompt, completion, total, calls, latency


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _operational_means(
    rows: Mapping[str, Mapping[str, Any]], row_format: str, system_label: str
) -> tuple[float, ...]:
    metrics = [
        _row_metrics(row, row_format, f"{system_label} sample {sample_id}")
        for sample_id, row in rows.items()
    ]
    columns = list(zip(*metrics, strict=True))
    return tuple(_mean(column) for column in columns)


def _official_score(results: Mapping[str, Any]) -> float:
    value = results.get("overall_accuracy")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("official evaluator did not return numeric overall_accuracy")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"official overall_accuracy is outside [0, 1]: {score}")
    return score


def _recomputed_distance_mean_relative_error(
    results: Mapping[str, Any], expected_ids: set[str]
) -> tuple[float, int]:
    per_category = results.get("per_category")
    if not isinstance(per_category, Mapping):
        raise ValueError("official evaluator did not return per_category results")
    distance = per_category.get("distance")
    if not isinstance(distance, Mapping):
        raise ValueError("official evaluator did not return per_category.distance")
    stored_value = distance.get("mean_error_rate")
    if isinstance(stored_value, bool) or not isinstance(stored_value, Real):
        raise ValueError("official distance mean_error_rate is not numeric")
    official_value = float(stored_value)
    if not math.isfinite(official_value) or official_value < 0:
        raise ValueError("official distance mean_error_rate is invalid")

    detailed = results.get("detailed_results")
    if not isinstance(detailed, list):
        raise ValueError("official evaluator did not return detailed prediction/GT rows")
    seen_ids: set[str] = set()
    errors: list[float] = []
    for index, row in enumerate(detailed):
        if not isinstance(row, Mapping):
            raise ValueError(f"official detailed result {index} is not an object")
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen_ids:
            raise ValueError("official detailed prediction/GT row IDs are invalid")
        seen_ids.add(sample_id)
        if row.get("category") != "distance":
            raise ValueError("official detailed result is outside the distance slice")
        try:
            prediction = float(row.get("normalized_prediction"))
            target = float(row.get("ground_truth"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(prediction) or not math.isfinite(target):
            continue
        errors.append(abs(prediction - target) / (target + 1e-8))
    if seen_ids != expected_ids:
        raise ValueError("official detailed prediction/GT rows do not cover the exact sample IDs")
    recomputed = _mean(errors) if errors else 0.0
    if not math.isclose(recomputed, official_value, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "official distance mean_error_rate does not match recomputed prediction/GT rows"
        )
    return recomputed, len(errors)


def _load_summary(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = run_dir.expanduser().resolve() / SUMMARY_FILENAME
    summary = _load_json_object(summary_path, "results summary")
    return summary_path, summary


def _assert_summary_matches(expected: Any, stored: Any, location: str = "results") -> None:
    """Require every official result field to match its stored summary value."""
    if isinstance(expected, Mapping):
        if not isinstance(stored, Mapping):
            raise ValueError(f"summary mismatch at {location}: expected an object")
        for key, value in expected.items():
            if key == "detailed_results":
                continue
            if key not in stored:
                raise ValueError(f"summary mismatch at {location}.{key}: missing key")
            _assert_summary_matches(value, stored[key], f"{location}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(stored, list) or len(expected) != len(stored):
            raise ValueError(f"summary mismatch at {location}: list shape differs")
        for index, (expected_item, stored_item) in enumerate(zip(expected, stored, strict=True)):
            _assert_summary_matches(expected_item, stored_item, f"{location}[{index}]")
        return
    if isinstance(expected, Real) and not isinstance(expected, bool):
        if isinstance(stored, bool) or not isinstance(stored, Real):
            raise ValueError(f"summary mismatch at {location}: expected numeric value")
        if not math.isclose(float(expected), float(stored), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"summary mismatch at {location}: recomputed={expected!r}, stored={stored!r}"
            )
        return
    if expected != stored:
        raise ValueError(
            f"summary mismatch at {location}: recomputed={expected!r}, stored={stored!r}"
        )


def _derive_selected_ids(
    samples: Sequence[Any],
    expected_count: int,
    selection_seed: int,
    explicit_ids: Sequence[Any] | None = None,
) -> list[str]:
    all_ids = [str(sample.sample_id) for sample in samples]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("benchmark has duplicate sample IDs")
    if explicit_ids:
        requested = [str(sample_id) for sample_id in explicit_ids]
        if len(set(requested)) != len(requested):
            raise ValueError("native config sample_ids contains duplicates")
        missing = sorted(set(requested) - set(all_ids))
        if missing:
            raise ValueError("native config sample_ids are absent from the benchmark")
        selected = [sample_id for sample_id in all_ids if sample_id in set(requested)]
    elif expected_count == len(all_ids):
        selected = all_ids
    else:
        selected = list(all_ids)
        random.Random(selection_seed).shuffle(selected)
        selected = selected[:expected_count]
    if len(selected) != expected_count:
        raise ValueError(
            f"derived selection has {len(selected)} samples, expected {expected_count}"
        )
    return selected


def _validate_atlas_manifest(
    spec: SystemSpec,
    *,
    benchmark_name: str,
    question_types: Sequence[str],
    expected_ids: Sequence[str],
    expected_concurrency: int = 1,
) -> None:
    manifest_path = spec.run_dir.expanduser().resolve() / MANIFEST_FILENAME
    manifest = _load_json_object(manifest_path, "Atlas run manifest")
    required = {
        "benchmark": benchmark_name,
        "question_filters": list(question_types),
        "engine": spec.engine,
        "llm_enable_thinking": False,
        "concurrency": expected_concurrency,
        "selected_ids": list(expected_ids),
        "seed": spec.selection_seed,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{spec.label} run manifest {key} does not match the table request")
    model_tiers = manifest.get("model_tiers")
    if not isinstance(model_tiers, Mapping) or not model_tiers:
        raise ValueError(f"{spec.label} run manifest has no model_tiers")
    expected_model = _model_identity(spec.backbone)
    if not expected_model or any(
        _model_identity(model_name) != expected_model for model_name in model_tiers.values()
    ):
        raise ValueError(f"{spec.label} run manifest model_tiers do not match the backbone")


def _validate_native_config(
    spec: SystemSpec,
    rows: Mapping[str, Mapping[str, Any]],
    all_samples: Sequence[Any],
    *,
    benchmark_name: str,
    question_types: Sequence[str],
    expected_count: int,
) -> list[str]:
    config_path = spec.run_dir.expanduser().resolve() / CONFIG_FILENAME
    config = _load_json_object(config_path, "native run config")
    if config.get("benchmark") != benchmark_name:
        raise ValueError(f"{spec.label} config benchmark does not match the table request")
    if config.get("question_type") != list(question_types):
        raise ValueError(f"{spec.label} config question_type does not match the table request")
    if _model_identity(config.get("llm_model")) != _model_identity(spec.backbone):
        raise ValueError(f"{spec.label} config llm_model does not match the backbone")

    if spec.endpoint_manifest is None:
        raise ValueError(f"{spec.label} requires a native endpoint manifest")
    endpoint = _load_json_object(
        spec.endpoint_manifest.expanduser().resolve(), "native endpoint manifest"
    )
    if _model_identity(endpoint.get("model")) != _model_identity(spec.backbone):
        raise ValueError("native endpoint manifest model does not match the backbone")
    configured_url = config.get("llm_base_url")
    endpoint_url = endpoint.get("url")
    if (
        not isinstance(configured_url, str)
        or not configured_url.strip()
        or not isinstance(endpoint_url, str)
        or configured_url.rstrip("/") != endpoint_url.rstrip("/")
    ):
        raise ValueError("native config llm_base_url does not match the endpoint manifest")

    limit = config.get("limit")
    if limit is not None and limit != expected_count:
        raise ValueError(f"{spec.label} config limit does not match expected_count")
    expected_ids = _derive_selected_ids(
        all_samples,
        expected_count,
        spec.selection_seed,
        config.get("sample_ids"),
    )
    if set(rows) != set(expected_ids):
        raise ValueError(f"{spec.label} prediction IDs do not match the derived selection")

    native_manifest_path = spec.run_dir.expanduser().resolve() / NATIVE_MANIFEST_FILENAME
    native_manifest = _load_json_object(native_manifest_path, "native run manifest")
    required_manifest = {
        "benchmark": benchmark_name,
        "question_filters": list(question_types),
        "concurrency": spec.concurrency,
        "endpoint_model": endpoint.get("model"),
    }
    for key, expected in required_manifest.items():
        if native_manifest.get(key) != expected:
            raise ValueError(f"{spec.label} native run manifest {key} does not match")
    selection = native_manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{spec.label} native run manifest has no selection object")
    required_selection = {
        "seed": spec.selection_seed,
        "expected_count": expected_count,
        "selected_ids": expected_ids,
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"{spec.label} native run manifest selection.{key} does not match")
    return expected_ids


def _verified_accounting_artifact(value: Any, description: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} accounting artifact is missing")
    raw_path = value.get("path")
    expected_hash = value.get("sha256")
    expected_size = value.get("size_bytes")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{description} accounting artifact metadata is invalid")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} accounting artifact not found: {path}")
    if _sha256(path) != expected_hash or path.stat().st_size != expected_size:
        raise ValueError(f"{description} accounting artifact hash or size changed")
    return path


def _validate_full_accounting(
    spec: SystemSpec,
    *,
    expected_count: int,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    if spec.accounting_path is None:
        raise ValueError(f"{spec.label} requires a full vLLM accounting artifact")
    accounting_path = spec.accounting_path.expanduser().resolve()
    accounting = _load_json_object(accounting_path, "full vLLM accounting")
    predictions_path = spec.run_dir.expanduser().resolve() / PREDICTIONS_FILENAME
    required = {
        "schema_version": 1,
        "status": "pass",
        "kind": "p3_full_vllm_accounting",
        "system_key": spec.key,
        "row_format": spec.row_format,
        "expected_count": expected_count,
        "concurrency": spec.concurrency,
        "output_dir": str(spec.run_dir.expanduser().resolve()),
    }
    mismatches = {
        key: accounting.get(key)
        for key, expected in required.items()
        if accounting.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{spec.label} full accounting mismatch: {mismatches}")
    predictions_artifact = accounting.get("predictions")
    verified_predictions = _verified_accounting_artifact(
        predictions_artifact, f"{spec.label} predictions"
    )
    if verified_predictions != predictions_path:
        raise ValueError(f"{spec.label} full accounting references different predictions")
    completed_ids = accounting.get("completed_ids")
    if (
        not isinstance(completed_ids, list)
        or len(completed_ids) != expected_count
        or set(map(str, completed_ids)) != set(expected_ids)
    ):
        raise ValueError(f"{spec.label} full accounting does not cover the exact sample IDs")
    attempt_count = accounting.get("attempt_count")
    interrupted_attempt_count = accounting.get("interrupted_attempt_count")
    attempt_statuses = accounting.get("attempt_statuses")
    attempts = accounting.get("attempts")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count <= 0
        or isinstance(interrupted_attempt_count, bool)
        or not isinstance(interrupted_attempt_count, int)
        or interrupted_attempt_count < 0
        or interrupted_attempt_count > attempt_count
        or not isinstance(attempt_statuses, list)
        or len(attempt_statuses) != attempt_count
        or any(status not in ACCOUNTED_ATTEMPT_STATUSES for status in attempt_statuses)
        or attempt_statuses[-1] != "pass"
        or attempt_statuses.count("accounted_interrupted") != interrupted_attempt_count
        or not isinstance(attempts, list)
        or len(attempts) != attempt_count
    ):
        raise ValueError(f"{spec.label} full accounting attempt list is invalid")
    attempt_paths = [
        _verified_accounting_artifact(artifact, f"{spec.label} attempt {index}")
        for index, artifact in enumerate(attempts)
    ]
    totals = accounting.get("terminal_counter_totals")
    expected_reasons = {"stop", "length", "abort", "error", "repetition"}
    if not isinstance(totals, Mapping) or set(totals) != expected_reasons:
        raise ValueError(f"{spec.label} full accounting five-reason schema is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in totals.values()
    ):
        raise ValueError(f"{spec.label} full accounting counters are invalid")
    abnormal = sum(int(totals[reason]) for reason in expected_reasons - {"stop"})
    journaled_calls = accounting.get("journaled_call_total")
    unattributed = accounting.get("unattributed_stop_calls")
    if (
        abnormal != 0
        or isinstance(journaled_calls, bool)
        or not isinstance(journaled_calls, int)
        or journaled_calls <= 0
        or isinstance(unattributed, bool)
        or not isinstance(unattributed, int)
        or unattributed != 0
        or totals["stop"] != journaled_calls
    ):
        raise ValueError(f"{spec.label} full accounting has unreconciled vLLM calls")
    vllm_jobs = accounting.get("vllm_job_ids")
    gpu_jobs = accounting.get("gpu_tool_job_ids")
    endpoint_urls = accounting.get("endpoint_urls")
    metrics_urls = accounting.get("metrics_urls")
    endpoint_manifests = accounting.get("endpoint_manifests")
    if (
        not isinstance(vllm_jobs, list)
        or len(vllm_jobs) != attempt_count
        or len(set(map(str, vllm_jobs))) != attempt_count
        or any(not str(value).isdigit() or int(str(value)) <= 0 for value in vllm_jobs)
    ):
        raise ValueError(f"{spec.label} full accounting vLLM service IDs are invalid")
    if spec.key in {"metric", "native"}:
        if (
            not isinstance(gpu_jobs, list)
            or len(gpu_jobs) != attempt_count
            or len(set(map(str, gpu_jobs))) != attempt_count
            or any(not str(value).isdigit() or int(str(value)) <= 0 for value in gpu_jobs)
        ):
            raise ValueError(f"{spec.label} full accounting GPU service IDs are invalid")
    elif gpu_jobs != []:
        raise ValueError(f"{spec.label} scenegraph accounting must not contain GPU jobs")
    if (
        not isinstance(endpoint_urls, list)
        or len(endpoint_urls) != attempt_count
        or not isinstance(metrics_urls, list)
        or len(metrics_urls) != attempt_count
        or not isinstance(endpoint_manifests, list)
        or len(endpoint_manifests) != attempt_count
    ):
        raise ValueError(f"{spec.label} full accounting endpoint provenance is invalid")

    accounted_ids: list[str] = []
    attempt_vllm_jobs: list[str] = []
    attempt_gpu_jobs: list[str] = []
    attempt_endpoint_urls: list[str] = []
    attempt_metrics_urls: list[str] = []
    attempt_endpoint_manifests: list[dict[str, Any]] = []
    attempt_totals = {reason: 0 for reason in expected_reasons}
    attempt_journaled_calls = 0
    for index, attempt_path in enumerate(attempt_paths):
        attempt = _load_json_object(attempt_path, "full vLLM attempt gate")
        attempt_required = {
            "schema_version": 1,
            "kind": "p3_vllm_accounting_attempt_gate",
            "system_key": spec.key,
            "row_format": spec.row_format,
            "expected_count": expected_count,
            "concurrency": spec.concurrency,
            "output_dir": str(spec.run_dir.expanduser().resolve()),
            "predictions": str(predictions_path),
        }
        attempt_mismatches = {
            key: attempt.get(key)
            for key, expected in attempt_required.items()
            if attempt.get(key) != expected
        }
        if attempt_mismatches:
            raise ValueError(
                f"{spec.label} attempt {index} metadata mismatch: {attempt_mismatches}"
            )
        status = attempt.get("status")
        runner_exit = attempt.get("runner_exit")
        termination = attempt.get("termination")
        if (
            status != attempt_statuses[index]
            or status != attempts[index].get("status")
            or status not in ACCOUNTED_ATTEMPT_STATUSES
        ):
            raise ValueError(f"{spec.label} attempt {index} status is invalid")
        if (
            isinstance(runner_exit, bool)
            or not isinstance(runner_exit, int)
            or runner_exit < 0
            or (status == "pass" and (runner_exit != 0 or termination != "normal"))
            or (status == "accounted_interrupted" and runner_exit == 0 and termination == "normal")
        ):
            raise ValueError(f"{spec.label} attempt {index} runner status is invalid")
        nested = attempt.get("artifacts")
        full_nested = attempts[index].get("nested_artifacts")
        required_nested = {"start", "counter_before", "counter_after", "endpoint_manifest"}
        if (
            not isinstance(nested, Mapping)
            or set(nested) != required_nested
            or full_nested != nested
        ):
            raise ValueError(f"{spec.label} attempt {index} nested artifacts are invalid")
        verified_nested: dict[str, Path] = {}
        for name in sorted(required_nested):
            nested_path = _verified_accounting_artifact(
                nested[name], f"{spec.label} attempt {index} {name}"
            )
            if nested_path.parent != attempt_path.parent:
                raise ValueError(f"{spec.label} attempt {index} {name} artifact path is invalid")
            verified_nested[name] = nested_path
        new_ids = attempt.get("newly_completed_ids")
        if (
            not isinstance(new_ids, list)
            or len(new_ids) != attempt.get("newly_completed_count")
            or len(set(map(str, new_ids))) != len(new_ids)
        ):
            raise ValueError(f"{spec.label} attempt {index} new-ID accounting is invalid")
        overlap = set(accounted_ids) & set(map(str, new_ids))
        if overlap:
            raise ValueError(f"{spec.label} attempt gates overlap on IDs: {sorted(overlap)}")
        accounted_ids.extend(map(str, new_ids))
        attempt_deltas = attempt.get("terminal_counter_deltas")
        if not isinstance(attempt_deltas, Mapping) or set(attempt_deltas) != expected_reasons:
            raise ValueError(f"{spec.label} attempt {index} five-reason schema is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in attempt_deltas.values()
        ):
            raise ValueError(f"{spec.label} attempt {index} counters are invalid")
        attempt_calls = attempt.get("journaled_call_delta")
        attempt_unattributed = attempt.get("unattributed_stop_calls")
        if (
            isinstance(attempt_calls, bool)
            or not isinstance(attempt_calls, int)
            or attempt_calls < 0
            or isinstance(attempt_unattributed, bool)
            or not isinstance(attempt_unattributed, int)
            or attempt_unattributed != 0
            or attempt_deltas["stop"] != attempt_calls
            or any(attempt_deltas[reason] != 0 for reason in expected_reasons - {"stop"})
        ):
            raise ValueError(f"{spec.label} attempt {index} calls are unreconciled")
        service = attempt.get("service")
        if not isinstance(service, Mapping):
            raise ValueError(f"{spec.label} attempt {index} has no service identity")
        if (
            service.get("endpoint_model") != "qwen3.6-27b"
            or service.get("endpoint_reasoning_parser") != "qwen3"
        ):
            raise ValueError(f"{spec.label} attempt {index} endpoint identity is invalid")
        endpoint_url = service.get("endpoint_url")
        metrics_url = service.get("metrics_url")
        endpoint_manifest = service.get("endpoint_manifest")
        if (
            not isinstance(endpoint_url, str)
            or not endpoint_url.startswith(("http://", "https://"))
            or not isinstance(metrics_url, str)
            or not metrics_url.startswith(("http://", "https://"))
            or endpoint_manifest != nested.get("endpoint_manifest")
        ):
            raise ValueError(f"{spec.label} attempt {index} endpoint provenance is invalid")
        manifest_record = _load_json_object(
            verified_nested["endpoint_manifest"], "attempt endpoint manifest"
        )
        if (
            manifest_record.get("model") != "qwen3.6-27b"
            or manifest_record.get("reasoning_parser") != "qwen3"
            or str(manifest_record.get("job_id", "")) != str(service.get("vllm_job_id", ""))
            or manifest_record.get("url") != endpoint_url
        ):
            raise ValueError(f"{spec.label} attempt {index} endpoint manifest identity is invalid")
        attempt_endpoint_urls.append(endpoint_url)
        attempt_metrics_urls.append(metrics_url)
        attempt_endpoint_manifests.append(dict(endpoint_manifest))
        attempt_vllm_jobs.append(str(service.get("vllm_job_id", "")))
        gpu_job = service.get("gpu_tool_job_id")
        if gpu_job is not None:
            attempt_gpu_jobs.append(str(gpu_job))
        for reason in expected_reasons:
            attempt_totals[reason] += int(attempt_deltas[reason])
        attempt_journaled_calls += attempt_calls
    if set(accounted_ids) != set(expected_ids) or len(accounted_ids) != expected_count:
        raise ValueError(f"{spec.label} attempt gates do not form the exact disjoint ID union")
    if (
        attempt_totals != dict(totals)
        or attempt_journaled_calls != journaled_calls
        or attempt_vllm_jobs != [str(value) for value in vllm_jobs]
        or attempt_gpu_jobs != [str(value) for value in gpu_jobs]
        or attempt_endpoint_urls != endpoint_urls
        or attempt_metrics_urls != metrics_urls
        or attempt_endpoint_manifests != endpoint_manifests
    ):
        raise ValueError(f"{spec.label} full accounting does not match its attempt gates")
    return {
        "path": accounting_path,
        "attempt_count": attempt_count,
        "interrupted_attempt_count": interrupted_attempt_count,
        "stop_calls": int(totals["stop"]),
        "abnormal_calls": abnormal,
        "unattributed_calls": unattributed,
        "terminal_totals": {str(key): int(value) for key, value in totals.items()},
        "journaled_calls": journaled_calls,
        "vllm_job_ids": [str(value) for value in vllm_jobs],
        "gpu_job_ids": [str(value) for value in gpu_jobs],
    }


def _select_benchmark_samples(benchmark: Any, sample_ids: set[str], expected_count: int) -> None:
    samples_by_id: dict[str, Any] = {}
    for sample in benchmark.data:
        sample_id = str(sample.sample_id)
        if sample_id in samples_by_id:
            raise ValueError(f"benchmark has duplicate sample_id {sample_id!r}")
        samples_by_id[sample_id] = sample
    missing = sorted(sample_ids - samples_by_id.keys())
    if missing:
        raise ValueError("prediction sample IDs are absent from benchmark: " + ", ".join(missing))
    benchmark.data = [sample for sample in benchmark.data if str(sample.sample_id) in sample_ids]
    if len(benchmark.data) != expected_count:
        raise ValueError(
            f"selected benchmark has {len(benchmark.data)} samples, expected {expected_count}"
        )


def _evaluate_system(
    benchmark: Any,
    spec: SystemSpec,
    rows: Mapping[str, Mapping[str, Any]],
    expected_count: int,
    predictions_path: Path,
    accounting: Mapping[str, Any],
) -> SystemResult:
    predictions = {sample_id: _answer(row, spec.row_format) for sample_id, row in rows.items()}
    official_results = benchmark.evaluate(predictions, output_dir=None)
    if not isinstance(official_results, Mapping):
        raise ValueError(f"{spec.label} official evaluator returned a non-object")
    score = _official_score(official_results)
    distance_error, distance_error_count = _recomputed_distance_mean_relative_error(
        official_results, set(rows)
    )
    summary_path, stored_summary = _load_summary(spec.run_dir)
    if stored_summary.get("total_samples") != expected_count:
        raise ValueError(
            f"{spec.label} summary total_samples={stored_summary.get('total_samples')!r}, "
            f"expected {expected_count}"
        )
    _assert_summary_matches(official_results, stored_summary)
    prompt, completion, total, calls, latency = _operational_means(
        rows, spec.row_format, spec.label
    )
    return SystemResult(
        spec=spec,
        samples=expected_count,
        official_score=score,
        official_distance_mean_relative_error=distance_error,
        distance_relative_error_count=distance_error_count,
        mean_prompt_tokens=prompt,
        mean_completion_tokens=completion,
        mean_total_tokens=total,
        mean_calls=calls,
        mean_latency_seconds=latency,
        backbone=spec.backbone,
        question_slice=spec.question_slice,
        predictions_path=predictions_path,
        summary_path=summary_path,
        accounting_path=accounting["path"],
        accounted_attempts=accounting["attempt_count"],
        interrupted_attempts=accounting["interrupted_attempt_count"],
        vllm_stop_calls=accounting["stop_calls"],
        vllm_abnormal_calls=accounting["abnormal_calls"],
        unattributed_stop_calls=accounting["unattributed_calls"],
        terminal_counter_totals=accounting["terminal_totals"],
        journaled_call_total=accounting["journaled_calls"],
    )


def _csv_text(results: Sequence[SystemResult]) -> str:
    stream = io.StringIO(newline="")
    fieldnames = [
        "system",
        "grounding",
        "backbone",
        "question_slice",
        "samples",
        "official_score",
        "official_distance_mean_relative_error",
        "distance_relative_error_count",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_total_tokens",
        "mean_calls",
        "concurrency",
        "accounted_attempts",
        "interrupted_attempts",
        "vllm_stop_calls",
        "vllm_length_calls",
        "vllm_abort_calls",
        "vllm_error_calls",
        "vllm_repetition_calls",
        "journaled_call_total",
        "vllm_abnormal_calls",
        "unattributed_stop_calls",
        "mean_per_row_latency_seconds_at_declared_concurrency",
        "run_dir",
        "predictions_path",
        "summary_path",
        "accounting_path",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "system": result.spec.label,
                "grounding": result.spec.grounding,
                "backbone": result.backbone,
                "question_slice": result.question_slice,
                "samples": result.samples,
                "official_score": f"{result.official_score:.10f}",
                "official_distance_mean_relative_error": (
                    f"{result.official_distance_mean_relative_error:.10f}"
                ),
                "distance_relative_error_count": result.distance_relative_error_count,
                "mean_prompt_tokens": f"{result.mean_prompt_tokens:.6f}",
                "mean_completion_tokens": f"{result.mean_completion_tokens:.6f}",
                "mean_total_tokens": f"{result.mean_total_tokens:.6f}",
                "mean_calls": f"{result.mean_calls:.6f}",
                "concurrency": result.spec.concurrency,
                "accounted_attempts": result.accounted_attempts,
                "interrupted_attempts": result.interrupted_attempts,
                "vllm_stop_calls": result.vllm_stop_calls,
                "vllm_length_calls": result.terminal_counter_totals["length"],
                "vllm_abort_calls": result.terminal_counter_totals["abort"],
                "vllm_error_calls": result.terminal_counter_totals["error"],
                "vllm_repetition_calls": result.terminal_counter_totals["repetition"],
                "journaled_call_total": result.journaled_call_total,
                "vllm_abnormal_calls": result.vllm_abnormal_calls,
                "unattributed_stop_calls": result.unattributed_stop_calls,
                "mean_per_row_latency_seconds_at_declared_concurrency": (
                    f"{result.mean_latency_seconds:.6f}"
                ),
                "run_dir": str(result.spec.run_dir.expanduser().resolve()),
                "predictions_path": str(result.predictions_path),
                "summary_path": str(result.summary_path),
                "accounting_path": str(result.accounting_path),
            }
        )
    return stream.getvalue()


def _markdown_text(
    results: Sequence[SystemResult], benchmark_name: str, expected_count: int
) -> str:
    def escaped(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")

    lines = [
        "# P3 benchmark comparison",
        "",
        f"Benchmark: `{benchmark_name}`",
        "",
        f"Backbone: `{escaped(results[0].backbone)}`",
        "",
        f"Question slice: `{escaped(results[0].question_slice)}`",
        "",
        f"Verified sample count per system: {expected_count}",
        "",
        "| System | Grounding | Backbone | Question slice | n | Official accuracy | "
        "Official distance mean relative error | Relative-error rows | "
        "Mean prompt tokens | Mean completion tokens | Mean total tokens | Mean calls | "
        "Concurrency | Attempts | Interrupted attempts | vLLM stop calls | Abnormal calls | "
        "Unattributed calls | "
        "Mean per-row latency at declared concurrency (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {escaped(result.spec.label)} | {escaped(result.spec.grounding)} | "
            f"{escaped(result.backbone)} | {escaped(result.question_slice)} | "
            f"{result.samples} | {result.official_score:.6f} | "
            f"{result.official_distance_mean_relative_error:.6f} | "
            f"{result.distance_relative_error_count} | "
            f"{result.mean_prompt_tokens:.2f} | {result.mean_completion_tokens:.2f} | "
            f"{result.mean_total_tokens:.2f} | {result.mean_calls:.2f} | "
            f"{result.spec.concurrency} | {result.accounted_attempts} | "
            f"{result.interrupted_attempts} | "
            f"{result.vllm_stop_calls} | {result.vllm_abnormal_calls} | "
            f"{result.unattributed_stop_calls} | {result.mean_latency_seconds:.3f} |"
        )
    lines.extend(["", "## Verified artifacts", ""])
    for result in results:
        lines.extend(
            [
                f"### {escaped(result.spec.label)}",
                "",
                f"Grounding: `{escaped(result.spec.grounding)}`",
                "",
                f"Backbone: `{escaped(result.backbone)}`",
                "",
                f"Question slice: `{escaped(result.question_slice)}`",
                "",
                "Official distance mean relative error: "
                f"`{result.official_distance_mean_relative_error:.10f}` "
                f"over `{result.distance_relative_error_count}` numeric prediction/GT rows",
                "",
                f"Concurrency: `{result.spec.concurrency}`",
                "",
                "Latency definition: mean measured per-row latency at the declared concurrency.",
                "",
                f"Accounting attempts: `{result.accounted_attempts}`",
                "",
                f"Interrupted but fully accounted attempts: `{result.interrupted_attempts}`",
                "",
                f"vLLM stop/abnormal/unattributed calls: `{result.vllm_stop_calls}` / "
                f"`{result.vllm_abnormal_calls}` / `{result.unattributed_stop_calls}`",
                "",
                "vLLM terminal totals (stop/length/abort/error/repetition): "
                f"`{result.terminal_counter_totals['stop']}` / "
                f"`{result.terminal_counter_totals['length']}` / "
                f"`{result.terminal_counter_totals['abort']}` / "
                f"`{result.terminal_counter_totals['error']}` / "
                f"`{result.terminal_counter_totals['repetition']}`",
                "",
                f"Journaled model calls: `{result.journaled_call_total}`",
                "",
                f"Run directory: `{result.spec.run_dir.expanduser().resolve()}`",
                "",
                f"Predictions: `{result.predictions_path}`",
                "",
                f"Stored summary: `{result.summary_path}`",
                "",
                f"Full vLLM accounting: `{result.accounting_path}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_payload(
    results: Sequence[SystemResult],
    *,
    benchmark_name: str,
    expected_count: int,
    markdown_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "pass",
        "benchmark": benchmark_name,
        "question_slice": results[0].question_slice,
        "backbone": results[0].backbone,
        "expected_count": expected_count,
        "latency_definition": "mean measured per-row latency at declared concurrency",
        "system_keys": [result.spec.key for result in results],
        "markdown": {"path": str(markdown_path), "sha256": _sha256(markdown_path)},
        "csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
        "systems": [
            {
                "key": result.spec.key,
                "label": result.spec.label,
                "grounding": result.spec.grounding,
                "concurrency": result.spec.concurrency,
                "official_accuracy": result.official_score,
                "official_distance_mean_relative_error": (
                    result.official_distance_mean_relative_error
                ),
                "distance_relative_error_count": result.distance_relative_error_count,
                "accounted_attempts": result.accounted_attempts,
                "interrupted_attempts": result.interrupted_attempts,
                "vllm_stop_calls": result.vllm_stop_calls,
                "terminal_counter_totals": result.terminal_counter_totals,
                "journaled_call_total": result.journaled_call_total,
                "vllm_abnormal_calls": result.vllm_abnormal_calls,
                "unattributed_stop_calls": result.unattributed_stop_calls,
                "mean_per_row_latency_seconds_at_declared_concurrency": (
                    result.mean_latency_seconds
                ),
                "predictions_path": str(result.predictions_path),
                "summary_path": str(result.summary_path),
                "accounting": {
                    "path": str(result.accounting_path),
                    "sha256": _sha256(result.accounting_path),
                },
            }
            for result in results
        ],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_result_tables(
    benchmark: Any,
    systems: Sequence[SystemSpec],
    *,
    benchmark_name: str,
    expected_count: int,
    output_dir: str | os.PathLike[str],
    question_types: Sequence[str] = (),
    atlas_concurrency: int = 1,
) -> tuple[Path, Path, Path, list[SystemResult]]:
    """Validate three runs and write the Markdown and CSV comparison tables."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if atlas_concurrency <= 0:
        raise ValueError("atlas_concurrency must be positive")
    if len(systems) != 3:
        raise ValueError(f"exactly three systems are required, got {len(systems)}")
    if len({spec.key for spec in systems}) != len(systems):
        raise ValueError("system keys must be unique")
    if any(not spec.grounding.strip() for spec in systems):
        raise ValueError("every system requires an explicit grounding label")
    if any(not spec.backbone.strip() for spec in systems):
        raise ValueError("every system requires an explicit backbone label")
    if any(not spec.question_slice.strip() for spec in systems):
        raise ValueError("every system requires an explicit question-slice label")
    if any(spec.concurrency <= 0 for spec in systems):
        raise ValueError("every system requires a positive concurrency")
    if len({spec.backbone for spec in systems}) != 1:
        raise ValueError("all systems must declare the same backbone")
    if len({spec.question_slice for spec in systems}) != 1:
        raise ValueError("all systems must declare the same question slice")
    if len({spec.selection_seed for spec in systems}) != 1:
        raise ValueError("all systems must declare the same selection seed")
    atlas_specs = [spec for spec in systems if spec.row_format == "atlas"]
    if any(spec.concurrency != atlas_concurrency for spec in atlas_specs):
        raise ValueError("Atlas system concurrency does not match atlas_concurrency")

    loaded: list[tuple[SystemSpec, Path, dict[str, dict[str, Any]]]] = []
    reference_ids: set[str] | None = None
    for spec in systems:
        predictions_path, rows = _validated_rows(spec, expected_count)
        sample_ids = set(rows)
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            missing = sorted(reference_ids - sample_ids)
            extra = sorted(sample_ids - reference_ids)
            raise ValueError(
                f"{spec.label} sample ID set differs; missing={missing}, extra={extra}"
            )
        loaded.append((spec, predictions_path, rows))

    assert reference_ids is not None
    all_samples = list(benchmark.data)
    expected_ids = _derive_selected_ids(all_samples, expected_count, systems[0].selection_seed)
    if reference_ids != set(expected_ids):
        raise ValueError("prediction IDs do not match the benchmark-derived selection")
    accounting_by_key: dict[str, dict[str, Any]] = {}
    for spec, _predictions_path, rows in loaded:
        if spec.row_format == "atlas":
            _validate_atlas_manifest(
                spec,
                benchmark_name=benchmark_name,
                question_types=question_types,
                expected_ids=expected_ids,
                expected_concurrency=atlas_concurrency,
            )
        elif spec.row_format == "native":
            _validate_native_config(
                spec,
                rows,
                all_samples,
                benchmark_name=benchmark_name,
                question_types=question_types,
                expected_count=expected_count,
            )
        else:
            raise ValueError(f"unsupported row format: {spec.row_format}")
        accounting_by_key[spec.key] = _validate_full_accounting(
            spec,
            expected_count=expected_count,
            expected_ids=expected_ids,
        )
    all_vllm_jobs = [
        job_id for spec in systems for job_id in accounting_by_key[spec.key]["vllm_job_ids"]
    ]
    if len(all_vllm_jobs) != len(set(all_vllm_jobs)):
        raise ValueError("P3 systems reused a vLLM service across full attempts")
    all_gpu_jobs = [
        job_id for spec in systems for job_id in accounting_by_key[spec.key]["gpu_job_ids"]
    ]
    if len(all_gpu_jobs) != len(set(all_gpu_jobs)):
        raise ValueError("P3 metric/native systems reused a GPU-tool service")
    _select_benchmark_samples(benchmark, reference_ids, expected_count)
    results = [
        _evaluate_system(
            benchmark,
            spec,
            rows,
            expected_count,
            predictions_path,
            accounting_by_key[spec.key],
        )
        for spec, predictions_path, rows in loaded
    ]

    destination = Path(output_dir).expanduser().resolve()
    markdown_path = destination / MARKDOWN_FILENAME
    csv_path = destination / CSV_FILENAME
    gate_path = destination / GATE_FILENAME
    _atomic_write(markdown_path, _markdown_text(results, benchmark_name, expected_count))
    _atomic_write(csv_path, _csv_text(results))
    _atomic_write_json(
        gate_path,
        _gate_payload(
            results,
            benchmark_name=benchmark_name,
            expected_count=expected_count,
            markdown_path=markdown_path,
            csv_path=csv_path,
        ),
    )
    return markdown_path, csv_path, gate_path, results


def run_from_args(args: argparse.Namespace) -> tuple[Path, Path, Path, list[SystemResult]]:
    benchmark = _load_benchmark(
        args.spatialclaw_root,
        args.benchmark,
        args.data_root,
        args.question_types,
    )
    systems = [
        SystemSpec(
            "scenegraph",
            "Spatial Atlas scenegraph",
            "atlas",
            Path(args.scenegraph_dir),
            args.scenegraph_grounding,
            backbone=args.backbone,
            question_slice=args.question_slice,
            engine="scenegraph",
            selection_seed=args.selection_seed,
            concurrency=args.atlas_concurrency,
            accounting_path=Path(args.scenegraph_accounting),
        ),
        SystemSpec(
            "metric",
            "Spatial Atlas metric",
            "atlas",
            Path(args.metric_dir),
            args.metric_grounding,
            backbone=args.backbone,
            question_slice=args.question_slice,
            engine="metric",
            selection_seed=args.selection_seed,
            concurrency=args.atlas_concurrency,
            accounting_path=Path(args.metric_accounting),
        ),
        SystemSpec(
            "native",
            "SpatialClaw native",
            "native",
            Path(args.native_dir),
            args.native_grounding,
            backbone=args.backbone,
            question_slice=args.question_slice,
            endpoint_manifest=Path(args.native_endpoint_manifest),
            selection_seed=args.selection_seed,
            concurrency=args.native_concurrency,
            accounting_path=Path(args.native_accounting),
        ),
    ]
    return build_result_tables(
        benchmark,
        systems,
        benchmark_name=args.benchmark,
        expected_count=args.expected_count,
        output_dir=args.output_dir,
        question_types=args.question_types,
        atlas_concurrency=args.atlas_concurrency,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        markdown_path, csv_path, gate_path, _ = run_from_args(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports validation failures
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[ok] Markdown: {markdown_path}")
    print(f"[ok] CSV: {csv_path}")
    print(f"[ok] P3 gate: {gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
