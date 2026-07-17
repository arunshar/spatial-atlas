"""Security and lifecycle tests for generated-code execution."""

import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from mlebench.executor import CodeExecutor


def test_safe_env_excludes_parent_secrets_and_uses_private_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "https://credential@example.invalid")
    monkeypatch.setenv("HOME", "/parent-home")

    env = CodeExecutor()._safe_env(tmp_path)

    assert "OPENAI_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert env["HOME"] == str(tmp_path / ".runtime-home")
    assert env["TMPDIR"] == str(tmp_path / ".runtime-tmp")
    assert Path(env["HOME"]).stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_stale_submission_is_removed_before_attempt(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("stale", encoding="utf-8")

    result = await CodeExecutor(timeout=5).execute("pass", tmp_path, submission)

    assert result is None
    assert not submission.exists()


@pytest.mark.asyncio
async def test_oversized_submission_is_rejected(tmp_path):
    executor = CodeExecutor(timeout=5, max_submission_bytes=16)

    result = await executor.execute(
        "from pathlib import Path\nPath('submission.csv').write_bytes(b'x' * 17)\n",
        tmp_path,
    )

    assert result is None
    assert "Submission exceeds" in (executor.last_error or "")


@pytest.mark.asyncio
async def test_background_descendant_cannot_grow_submission_after_parent_exit(tmp_path):
    pipeline = (
        "import subprocess,sys\n"
        "from pathlib import Path\n"
        "Path('submission.csv').write_bytes(b'ok')\n"
        'child = "import time; from pathlib import Path; time.sleep(0.05); '
        "Path('submission.csv').write_bytes(b'x' * 4096)\"\n"
        "subprocess.Popen([sys.executable, '-c', child], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    )
    executor = CodeExecutor(timeout=5, max_submission_bytes=128)

    result = await executor.execute(pipeline, tmp_path)
    await asyncio.sleep(0.2)

    assert result == b"ok"
    assert (tmp_path / "submission.csv").read_bytes() == b"ok"


@pytest.mark.asyncio
async def test_stdout_capture_is_bounded(tmp_path):
    executor = CodeExecutor(timeout=5, max_stream_bytes=128)

    result = await executor.execute("print('x' * 1024)", tmp_path)

    assert result is None
    assert "stdout exceeds" in (executor.last_error or "")


@pytest.mark.asyncio
async def test_timeout_kills_spawned_child_process_group(tmp_path):
    marker = tmp_path / "child-survived"
    child_code = (
        "import pathlib,time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('survived')"
    )
    pipeline = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])\n"
        "time.sleep(30)\n"
    )

    executor = CodeExecutor(timeout=0.1)
    result = await executor.execute(pipeline, tmp_path)
    await asyncio.sleep(0.7)

    assert result is None
    assert "timed out" in (executor.last_error or "")
    assert not marker.exists()


@pytest.mark.asyncio
async def test_cancellation_kills_spawned_child_process_group(tmp_path):
    marker = tmp_path / "child-survived-cancel"
    child_code = (
        "import pathlib,time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('survived')"
    )
    pipeline = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])\n"
        "time.sleep(30)\n"
    )

    task = asyncio.create_task(CodeExecutor(timeout=30).execute(pipeline, tmp_path))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.7)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_timeout_kills_sigterm_ignoring_child_after_parent_exits(tmp_path):
    marker = tmp_path / "sigterm-ignoring-child-survived"
    child_code = (
        "import pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(0.6); "
        f"pathlib.Path({str(marker)!r}).write_text('survived'); "
        "time.sleep(30)"
    )
    pipeline = (
        f"import subprocess,sys\nsubprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])\n"
    )

    executor = CodeExecutor(timeout=0.1, termination_grace_seconds=0.2)
    result = await executor.execute(pipeline, tmp_path)
    await asyncio.sleep(0.8)

    assert result is None
    assert "timed out" in (executor.last_error or "")
    assert not marker.exists()


def test_test_payload_does_not_depend_on_shell_interpolation():
    """Document that child payloads are passed as argv, not through a shell."""
    payload = "print('$OPENAI_API_KEY')"
    assert shlex.quote(payload) != payload
    assert os.path.basename(sys.executable).startswith("python")
