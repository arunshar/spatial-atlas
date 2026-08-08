"""Repo-root guard against running one-use warm-control bundle tests by accident.

`sessions/fable/work` holds one-use, forward-only transaction bundles. Their suites are written
against a bundle's DRAFT (guarded) bytes, so once a bundle is armed, running them can perform a real
invocation and permanently consume a one-shot identity. That is not hypothetical: after the b73e20c5
guard transition, a guarded-only test called `monitor.main([])` expecting the draft guard to stop it;
with the guard removed, that call became a real armed invocation and permanently consumed the
b73e20c5 identity and attempt. The Slurm job it happened to target, 13979984, was not itself
consumed or affected; it is not a one-use resource in this project's identity discipline.

Three layers protect that directory, because no single one covers every invocation:

  1. `testpaths` in pyproject.toml, which pytest honors only when the invocation directory IS the
     rootdir and no path argument was given.
  2. `sessions/fable/work/conftest.py`, whose `pytest_ignore_collect` stops directory recursion
     regardless of the working directory. That file is untracked, because .gitignore excludes
     `sessions/`, so it protects a working copy but never arrives with a fresh clone.
  3. This file, which is tracked, and which catches the case the other two structurally cannot:
     a test file or node id named explicitly on the command line. pytest exempts paths given as
     arguments from `pytest_ignore_collect` (see `_pytest/main.py`, the `isinitpath` check), so an
     IDE gutter "Run test" button, or a node id copied from an earlier failure summary, walks
     straight past layer 2. `pytest_collection_modifyitems` sees every collected item however it
     was named.

This hook is inert when `sessions/fable/work` does not exist, which is the case in CI, since
.gitignore keeps `sessions/` out of the checkout entirely.

Known limit, stated so nobody assumes more: collection has already imported the test modules by the
time this hook runs, so module-level side effects are not prevented. The danger in these suites
lives in test bodies, which this does stop. `--noconftest` disables this file along with every
other conftest, and is a deliberate act.

To run these suites on purpose, against a bundle confirmed to still be in its guarded epoch:

    SPATIAL_ATLAS_ALLOW_WORK_TESTS=1 pytest sessions/fable/work
"""

import os
from pathlib import Path

import pytest

OPT_IN = "SPATIAL_ATLAS_ALLOW_WORK_TESTS"
GUARDED_DIR = Path(__file__).resolve().parent / "sessions" / "fable" / "work"


def _opted_in() -> bool:
    return os.environ.get(OPT_IN) == "1"


def _under_guard(item) -> bool:
    raw = getattr(item, "path", None) or getattr(item, "fspath", None)
    if raw is None:
        return False
    try:
        candidate = Path(str(raw)).resolve()
    except OSError:
        return False
    return candidate == GUARDED_DIR or GUARDED_DIR in candidate.parents


def pytest_collection_modifyitems(config, items):
    """Refuse the whole run if any collected item lives under the guarded directory.

    Refusing the run rather than silently deselecting is deliberate. A silent deselect is
    indistinguishable from "there were no tests there", and exits 0, which automation reads as
    success. `pytest.UsageError` exits 4 with a visible message.
    """

    if _opted_in() or not items:
        return
    caught = [item for item in items if _under_guard(item)]
    if not caught:
        return
    raise pytest.UsageError(
        f"refusing to run {len(caught)} test(s) under {GUARDED_DIR}: these are one-use "
        f"warm-control bundles, and running them against armed bytes can permanently consume a "
        f"one-shot identity. Set {OPT_IN}=1 if you have confirmed the bundle is still in its "
        f"guarded epoch."
    )
