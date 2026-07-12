# Spatial Atlas continuation handoff

This branch contains the reviewed implementation for the SpatialClaw to Spatial Atlas metric-perception
evaluation. It is intended to let another local coding agent continue the project without rediscovering
the code and safety boundaries.

## Read order

1. Read this file.
2. If this checkout has the ignored local `sessions/` directory, read
   `sessions/CODEX_CHECKPOINT_2026-07-12.md`, `sessions/LEDGER.md`, and
   `sessions/CODEX_HANDOFF.md` in full. Those files own the live MSI job state and detailed run history.
3. Inspect `git status --short` before editing. Preserve unrelated changes.

The `sessions/` directory is deliberately not in GitHub. Do not force-add it or recreate its operational
details in a public file. If the local checkpoint is unavailable, do not assume the live run state. Report
that missing context and restrict work to local review and tests until the owner provides it.

## Current project state

- P0 to P2 are complete.
- P3 is in progress and is governed by the private checkpoint and ledger.
- P4 and P5 have not started and must remain after P3.
- The primary experiment is QSpatial Gap-97: 97 rows, 84 images, 54 clusters.
- The frozen five-arm order is: Spatial Atlas scenegraph, question-only, correct-image metric,
  fixed shuffled-image metric, then SpatialClaw native.
- QSpatial journals are label-free. Do not load benchmark targets until all five full journals are sealed.

## Code delivered on this branch

- Metric perception in Spatial Atlas using SAM3 plus DA3 instead of generated spatial coordinates.
- Frozen QSpatial question parsing and horizontal surface-gap geometry.
- `eval_bench.py` and supporting tests for deterministic, resumable evaluation journals.
- Dataset and evaluation compatibility changes needed for Omni3D, Warehouse diagnostics, and QSpatial.
- Tests, fakes, strict geometry validation, and model-thinking policy support.

## Required safety rules

- Never push the adjacent SpatialClaw working copy. Its remote is NVlabs upstream.
- Push only this repository, `arunshar/spatial-atlas`.
- Never commit `sessions/`, raw data, generated results, secrets, tokens, or endpoint registries.
- Never submit a job wider than two GPUs unless the owner explicitly authorizes it.
- Never use Warehouse full-486 as claim-bearing metric evidence.
- Never fabricate, extrapolate, or hand-copy a result. Every reported number needs an executed artifact.
- Do not start P4 until P3 concludes, even if the P3 outcome is negative.

## Verification commands

```bash
cd /Users/arunsharma/code/spatial-atlas
uv run pytest -q -m 'not e2e and not gpu'

cd /Users/arunsharma/code/SpatialClaw
.venv-test/bin/python -m pytest -q -m 'not e2e and not gpu'
```

Run the local tests before any MSI mutation. For MSI actions, use only the private session handoff and
ledger. Validate completed jobs from immutable journals, service-counter snapshots, and sealed artifacts,
not from Slurm completion alone.

## Handoff prompt

```text
Read CLAUDE_HANDOFF.md first. Then, if present, read sessions/CODEX_CHECKPOINT_2026-07-12.md, sessions/LEDGER.md, and sessions/CODEX_HANDOFF.md in full. Continue the SpatialClaw to Spatial Atlas project strictly in phase order. Obey every hard rule in the private handoff. Preserve the current worktree, never push SpatialClaw or sessions, and do not infer MSI job outcomes. Revalidate live jobs and immutable artifacts before launching anything. P3 must pass its smoke and sequential pilot gates before full QSpatial arms. P4 and P5 remain blocked behind P3. Maintain sessions/LEDGER.md with commands, artifacts, results, and blockers.
```
