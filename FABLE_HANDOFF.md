# Spatial Atlas handoff for Fable

This is the public-safe continuation record for the SpatialClaw to Spatial Atlas project as of
2026-07-17. It summarizes the complete recovery arc without publishing private MSI artifacts,
credentials, endpoint registries, raw scheduler records, or raw model logs.

## Read order

1. Read this file and `CLAUDE_HANDOFF.md` completely.
2. Inspect `git status --short --branch` before editing.
3. If the ignored local `sessions/` directory exists, read `sessions/CONTEXT_CHECKPOINT.md`,
   `sessions/LEDGER.md`, and `sessions/CODEX_HANDOFF.md` completely and in that order.
4. Treat `sessions/LEDGER.md` as the authoritative chronological record. Section 9 of the private
   handoff contains hard rules.

The private `sessions/` directory is deliberately ignored. Never force-add or publish it. If it is
not available, do not perform live MSI work from this public summary alone.

## Current state

- Branch: `feat/metric-perception`.
- P0, P1, and P2 are complete.
- P3 is paused inside Task 1, before a qualifying preflight and before any replacement smoke.
- There are no live MSI jobs at this checkpoint.
- No authenticated endpoint exists.
- No service pair or smoke was started after the latest correction.
- Tasks 2 through 13, P4, and P5 remain blocked behind Task 1.
- The `continue-spatial-atlas-benchmark` automation remains paused.
- A new service pair or smoke requires explicit owner authorization.

The immediate blocker is a compute-only CUDA startup guard failure. Fresh preflight job `13434346`
had exact resources, zero restarts, stable accounting, and a valid 50-record journal ending at
`S07_CUDA_GUARD_FAIL`. The corrected environment gate at S05 and exact 244-file source gate at S06
both passed.

## Nonnegotiable boundaries

- Never push the adjacent SpatialClaw checkout. Its remote is upstream NVlabs.
- Push only `arunshar/spatial-atlas`.
- Never commit `sessions/`, raw results, protected journals, credentials, tokens, endpoint
  registries, or raw logs.
- Never resume, reuse, or count any failed or canceled job listed below.
- Never use more than two GPUs per job.
- Never overlap benchmark clients.
- Never load QSpatial labels before all five full-arm journals are sealed.
- Never use Warehouse full-486 as claim-bearing evidence.
- Never weaken a gate or fabricate a result.
- Record every live command, job, state, hash, artifact, result, and blocker in the private ledger.

## Recovery arc

Every job in this table is diagnostic-only and permanently ineligible for reuse or counting.

| Job | Outcome | What was learned or corrected |
| --- | --- | --- |
| `13227016` | Failed before endpoint publication | Initial vLLM service failed. It produced no authenticated endpoint and no smoke was submitted. |
| `13227017` | Safely canceled | The paired GPU-tools allocation was canceled after the vLLM failure. |
| `13293912` | Safely canceled | The inherited service could not satisfy fresh lifetime and provenance gates. Reuse was rejected. |
| `13404101` | Wrapper failure | Slurm-owned `ENVIRONMENT=BATCH` and `SLURMD_NODENAME` were rejected by an overly narrow export allowlist. Exact-value handling and tests were added. |
| `13412087` | Wrapper failure | Slurm also injects mandatory OMPI and PRTE launcher variables, plus conditional Hydra and GRES variables. The allowlist was corrected narrowly with exact values and regressions. |
| `13414262` | Failed before receipt sealing | The run advanced beyond earlier wrapper failures but produced no protected receipt. This motivated a separate fixed-enum, preflight-only journal. |
| `13415703` | Held transaction rolled back | The first held record had not converged on the expected node-count representation. The job never executed. |
| `13416015` | Held transaction rolled back | A bounded poll proved MSI renders the one-node request as exact `NumNodes=1-1`, not `NumNodes=1`. The job never executed. |
| `13416892` | Failed at `S05_ENV_GUARD_FAIL` | The fixed-enum journal localized the failure to the environment guard. |
| `13425224` | Classifier completed | A terminal-validator bug expected an extra delimiter from `sacct -P`; MSI correctly returned seven fields without a trailing delimiter. Validator v2 fixed only that parser. The accepted result proved exact runtime semantics but nonempty successful-run stderr. |
| `13434346` | Failed at `S07_CUDA_GUARD_FAIL` | After the torch warning correction, S05 environment and S06 source gates passed. Accounting was exact and the journal was valid. The CUDA guard is the remaining blocker. |

## Corrections already completed

The private MSI runtime contains the reviewed corrections. Their current exact hashes are included
for local verification, not for publication of their contents.

| Private runtime file | SHA-256 |
| --- | --- |
| `qspatial_environment_guard.py` | `576d38828f887349e01b45c4b0aa392c60a6a347b44a33fcd5829fdb02c50829` |
| `qspatial_source_provenance.json` | `69332de1bb26fc4e9ffc166ed91442efd26a0b083052f5e45723480df57c8cb1` |
| `p3_vllm_startup_diagnostic.py` | `0f970de22665442fa3a7ff653bc0661f75c3f0af22fb30c4f8753a8204944df7` |
| `p3_vllm_preflight_milestone.py` | `6427fa6a0d48cc5a6c53a06289a073659b8b86146cc23bc781fafbcaef92656f` |
| `p3_vllm_preflight_milestone.sbatch` | `55886659de8e203ee7c9b2d9291e7f4113592954a41c0e4f0731f88ec20e1a86` |

The accepted intervention classified successful environment-guard stderr as a `FutureWarning`
originating from the `torch.cuda` module family. The production correction suppresses only that
FutureWarning while importing torch. It does not suppress UserWarning, warnings from other torch
modules, or warnings from unrelated modules. Exact provenance was regenerated through the source
manifest, startup supervisor, preflight runner, and wrapper.

Local verification after this correction included:

- 39 focused environment-guard tests.
- 714 focused guard, provenance, supervisor, runtime, wrapper, transaction, and service tests.
- 1,363 tests across the private MSI suite.
- Exact 244-file source-provenance validation.
- Bash syntax, ShellCheck, read-only Python compilation, and full MSI gate replay.

## Current failure evidence

Job `13434346` produced the following sanitized accepted classification:

```text
state=FAILED
exit=CODE_1
restarts=ZERO
request=EXACT
allocation=EXACT
stable=PASS
closure=EXACT_TWO
journal=PASS
count=50
last=S07_CUDA_GUARD_FAIL
journal_sha256=93bd3f13d4ff0b735591fe57843c28aa0a060379f8088d796b016e460aeaebb9
journal_size=999
```

This does not yet distinguish a torch-import warning from CUDA availability, visible-device count,
kernel execution, numeric validation, stdout mismatch, timeout, capture, or cleanup failure. Prior
evidence makes the same `torch.cuda` FutureWarning a supported candidate, but it is still an
inference for this guard.

## Exact next step

Do not edit the CUDA guard yet. First build and review one fresh compute-only fixed-enum
intervention classifier that:

1. Requests one node and exactly two typed A100 GPUs, starts no vLLM process, endpoint, service,
   smoke, or benchmark client.
2. Runs the unchanged CUDA guard twice to prove stable normal behavior.
3. Repeats it with only a `torch.cuda` FutureWarning filter applied during torch import.
4. Captures all child stdout and stderr in memory with strict byte limits and never prints them.
5. Classifies the guard's already fixed JSON success or failure reason, stderr empty or nonempty,
   timeout, exit, process-group cleanup, and adopted-child cleanup using allowlisted enums only.
6. Uses a fresh held transaction with exact resources, fresh comment, fresh sentinel, fresh receipt,
   exact rollback, and no reuse of any prior job.

If normal execution passes with nonempty stderr and the narrow filtered run passes with empty
stderr, mirror the already reviewed import-scoped filter in `p3_cuda_startup_guard.py`, add
regression coverage, regenerate exact provenance, synchronize only reviewed files, replay every MSI
gate, and submit one completely fresh compute-only preflight. If the result is different, implement
only the correction directly supported by the fixed classification.

Task 2 can begin only after a fresh Task 1 preflight reaches `D02_COMPLETE`. Starting a service pair
or smoke still requires explicit authorization.

## Tracked code on this branch

The branch contains the metric-perception implementation plus the reviewed local hardening and
evaluation surface:

- SAM3 plus Depth Anything 3 metric perception and deterministic QSpatial gap evaluation.
- LLM budget enforcement, per-call telemetry, and task-failure lifecycle handling.
- MLEBench archive limits and executor security checks.
- Request-size middleware and public artifact hygiene.
- Pinned model revisions and deterministic P5 asset generation.
- Updated tests, CI, Docker image, project documentation, tutorial, architecture, and paper text.

Run the complete local suite before changing live MSI state:

```bash
cd /Users/arunsharma/code/spatial-atlas
.venv/bin/python -m pytest -q -p no:cacheprovider -m 'not e2e and not gpu'
```

At this handoff, that command passed 1,778 tests with one E2E/GPU test deselected. Offline Ruff
lint passed, and Ruff formatting passed for all changed and new Python files.

## Resume prompt

```text
Read FABLE_HANDOFF.md and CLAUDE_HANDOFF.md completely. If the ignored local sessions directory is available, then read sessions/CONTEXT_CHECKPOINT.md, sessions/LEDGER.md, and sessions/CODEX_HANDOFF.md completely in that order. Treat the ledger as authoritative and obey section 9 of the private handoff. P0 to P2 are complete. P3 Task 1 is blocked after diagnostic-only job 13434346 ended at S07_CUDA_GUARD_FAIL, while S05 environment and S06 source passed. Do not reuse any historical job. Do not start a service or smoke. Build a fresh compute-only fixed-enum CUDA intervention classifier first, preserve the two-GPU cap, replay all gates, and record every command and result in the ledger. Never push SpatialClaw or sessions.
```
