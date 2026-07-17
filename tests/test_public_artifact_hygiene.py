from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    "README.md",
    "ARCHITECTURE.md",
    "TUTORIAL.md",
    "paper/spatial_atlas.md",
    "paper/spatial_atlas.tex",
)


@pytest.mark.unit
def test_ci_does_not_pass_repository_secrets_to_test_container():
    workflow = (ROOT / ".github/workflows/test-and-publish.yml").read_text()

    assert "toJson(secrets)" not in workflow
    assert "SECRETS_JSON" not in workflow
    assert "--env-file .env" not in workflow


@pytest.mark.unit
def test_generated_junit_report_is_ignored():
    ignored_entries = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "junit.xml" in ignored_entries


@pytest.mark.unit
@pytest.mark.parametrize("paper_path", ["paper/spatial_atlas.md", "paper/spatial_atlas.tex"])
def test_paper_marks_fieldworkarena_as_unevaluated(paper_path):
    paper = (ROOT / paper_path).read_text()

    assert "FieldWorkArena remained gated and inaccessible" in paper
    for unsupported_claim in (
        "Full System (SSG + EG + F2)",
        "45,200",
        "21--24 percentage point improvement",
        "achieving an 82% valid submission rate",
        "achieving an 82\\% valid submission rate",
    ):
        assert unsupported_claim not in paper


@pytest.mark.unit
@pytest.mark.parametrize("doc_path", PUBLIC_DOCS)
def test_public_docs_match_fail_closed_mle_runtime(doc_path):
    document = (ROOT / doc_path).read_text()
    normalized = document.replace("\\_", "_").lower()

    assert "\u2014" not in document
    assert "at most 3 total attempts" in normalized or "3 total attempts" in normalized
    assert "600-second" in normalized or "600s" in normalized
    assert "dummy" in normalized and "disabled" in normalized
    assert "150000" in normalized.replace("_", "").replace(",", "")
    assert "public a2a" in normalized
    assert "concurrency-safe" in normalized
    assert "heuristic" in normalized
    assert "one a2a execution" in normalized or "per-execution" in normalized
    assert "estimate" in normalized and "prompt" in normalized
    assert "maximum completion" in normalized
    assert "not exact tokenizer" in normalized
    assert "frozen benchmark" in normalized
    assert "observational" in normalized
    assert "artifact" in normalized
    assert "oversubscribe" in normalized or "exceed" in normalized

    assert "atlas_enable_mlebench_code_execution" in normalized
    assert "atlas_trusted_isolated_worker" in normalized
    assert "atlas_allow_dummy_submission" in normalized
    assert "atlas_bearer_token" in normalized
    assert "atlas_allow_unauthenticated_public" in normalized
    assert "32" in normalized
    assert "normal non-loopback" in normalized
    assert "loopback" in normalized and "may omit" in normalized
    assert "test-only" in normalized
    assert "64 mib" in normalized
    assert "503" in normalized
    assert "4" in normalized and ("concurrency" in normalized or "active requests" in normalized)

    for stale_claim in (
        "sandboxed subprocess",
        "safe sandbox",
        "default: 300 seconds",
        "300 seconds per attempt",
        "total of 4 attempts",
        "1 initial + 3",
    ):
        assert stale_claim not in normalized
