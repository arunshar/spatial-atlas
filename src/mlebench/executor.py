"""
Spatial Atlas — Safe Code Executor

Runs generated ML pipeline scripts in a subprocess with timeout.
Captures stdout/stderr for debugging and self-healing.
"""

import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger("spatial-atlas.mlebench.executor")


class CodeExecutor:
    """Execute ML pipeline code safely in a subprocess."""

    def __init__(self, timeout: int = 600):
        self.timeout = timeout
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        self.last_error: str | None = None

    async def execute(
        self,
        code: str,
        working_dir: Path,
        submission_path: Path | None = None,
    ) -> bytes | None:
        """
        Execute ML code in subprocess, return submission.csv bytes.

        Args:
            code: Complete Python script to execute
            working_dir: Directory to run in (contains data/)
            submission_path: Where to find submission.csv (default: working_dir/submission.csv)

        Returns:
            submission.csv bytes if produced, None if execution failed
        """
        script_path = working_dir / "pipeline.py"
        script_path.write_text(code)

        if submission_path is None:
            submission_path = working_dir / "submission.csv"

        self.last_stdout = ""
        self.last_stderr = ""
        self.last_error = None

        logger.info(f"Executing pipeline.py in {working_dir} (timeout={self.timeout}s)")

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._safe_env(),
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            self.last_stdout = stdout.decode("utf-8", errors="replace")
            self.last_stderr = stderr.decode("utf-8", errors="replace")

            # Log output for debugging
            if self.last_stdout:
                logger.info(f"Pipeline stdout (last 500 chars): {self.last_stdout[-500:]}")
            if self.last_stderr:
                logger.warning(f"Pipeline stderr (last 500 chars): {self.last_stderr[-500:]}")

            if proc.returncode != 0:
                self.last_error = (
                    f"Script exited with code {proc.returncode}.\n"
                    f"Stderr:\n{self.last_stderr[-2000:]}"
                )
                logger.error(f"Pipeline failed: exit code {proc.returncode}")
                return None

            # Check for submission file
            if submission_path.exists():
                csv_bytes = submission_path.read_bytes()
                logger.info(f"Submission produced: {len(csv_bytes)} bytes")
                return csv_bytes
            else:
                self.last_error = (
                    f"Script ran successfully but did not produce {submission_path.name}.\n"
                    f"Stdout:\n{self.last_stdout[-1000:]}"
                )
                logger.error("No submission.csv found after execution")
                return None

        except asyncio.TimeoutError:
            self.last_error = f"Code execution timed out after {self.timeout}s"
            logger.error(self.last_error)
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return None
        except Exception as e:
            self.last_error = f"Execution error: {e}"
            logger.error(f"Pipeline execution exception: {e}")
            return None

    def _safe_env(self) -> dict[str, str]:
        """Build a safe environment for subprocess execution."""
        import os
        env = os.environ.copy()
        # Ensure reproducibility
        env["PYTHONHASHSEED"] = "42"
        # Suppress warnings that clutter stderr
        env["PYTHONWARNINGS"] = "ignore"
        return env
