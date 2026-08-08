# Warm-Launch Wrapper Regeneration

The warm-launch SSH control system establishes a durable SSH control master to `login.msi.umn.edu` through a hardened, hash-pinned wrapper. The wrapper, launcher, outer body, classifier, stop request helper, derivation script, and qualification manifests form a closed hash-pinned verification chain: every file's SHA256 is verified by the launcher at runtime and cross-referenced by the manifests and derivation script.

When an SSH argument in the wrapper changes (for example adding an identity file or adjusting the password prompt count), the wrapper's SHA256 changes, which breaks the chain. This document describes how to regenerate the full chain so the launcher accepts the modified wrapper.

## When regeneration is needed

Regenerate when any of these wrapper components change:

- The `master_argv` SSH argument list in `warm_m5_teardown_r15_control.sh` (the authenticating connection)
- The `check_argv` SSH argument list (the control-socket probe)
- The pinned remote constants (`REMOTE_HOST`, `REMOTE_USER`, `REMOTE_PORT`)
- The auth-timing constants (`AUTH_TIMEOUT_SECONDS`, `CHECK_TIMEOUT_SECONDS`)

Do not regenerate for changes to the state directory path, control directory path, or known hosts path unless the launcher's constant validation gate (lines 1714-1740) is also updated.

## Files in the pin chain

All files live under `sessions/fable/work/`, which is gitignored and stays local (see the project handoff hard rules). The pin chain has these layers:

| File | Role | What pins it |
|------|------|-------------|
| `warm_m5_teardown_r15_control.sh` | Wrapper: the SSH supervisor (contains `master_argv`) | Launcher's `EXPECTED_WRAPPER_SHA256` + manifests + derivation patch |
| `warm_m5_teardown_r15_launch.sh` | Launcher: materializes, verifies, and execs the wrapper | Test's `LAUNCH_SHA256` + manifests + r25 loader |
| `warm_m5_teardown_r15_outer_body.sh` | Outer body: the bash layer between launcher and wrapper | Launcher's `EXPECTED_OUTER_SHA256` + test + manifests |
| `warm_m5_teardown_r15_journal_classifier.py` | Read-only journal classifier | Test's `CLASSIFIER_SHA256` + manifests + derivation payload |
| `warm_m5_teardown_r15_stop_request.py` | Stop request creator | Test's `STOP_HELPER_SHA256` + manifests + derivation payload |
| `derive_warm_m5_teardown_r15.py` | Derivation: reconstructs all artifacts from R14 parents + patches | Test's `DERIVATION_SHA256` + manifests |
| `warm_m5_teardown_r15_cleanup.py` | Stale-state cleanup | Owns `WRAPPER_SHA256` for the READY line |
| `test_warm_m5_teardown_r15_control.py` | Test suite: verifies all hashes and expected SSH args | Manifests pin the test's own hash |
| `warm_m5_teardown_r15_*_manifest_*.json` | Qualification manifests: pin every artifact's hash + size | R25 loader and recovery test |
| `r25_live_fire_loader_v1_7aa0b55d.py` | Live-fire loader: pins hashes, sizes, inodes | Recovery test |

## Regeneration steps

### 1. Edit the wrapper

Modify `master_argv` (or `check_argv`, or the pinned constants) in `warm_m5_teardown_r15_control.sh`. For example, adding an identity file and raising the prompt count:

```python
master_argv = [
    "/usr/bin/ssh",
    "-F", "none",
    "-i", "/Users/arunsharma/.ssh/codex_agate_ed25519",
    "-M",
    "-N",
    ...
    "-o", "NumberOfPasswordPrompts=3",
    ...
]
```

### 2. Compute the new wrapper hash and size

```bash
cd sessions/fable/work
shasum -a 256 warm_m5_teardown_r15_control.sh
wc -c warm_m5_teardown_r15_control.sh
```

Record the new SHA256 and byte size.

### 3. Update the launcher's pinned wrapper hash

In `warm_m5_teardown_r15_launch.sh`, replace `EXPECTED_WRAPPER_SHA256` with the new hash:

```bash
readonly EXPECTED_WRAPPER_SHA256=<new-wrapper-hash>
```

### 4. Update the test's expected master args

In `test_warm_m5_teardown_r15_control.py`:

- Update `WRAPPER_SHA256` to the new hash.
- Update `_expected_master_args()` to mirror the new `master_argv` (add `-i`, change `NumberOfPasswordPrompts`, etc.). This must match the wrapper's `master_argv` exactly, since the test verifies the rendered fake SSH receives the expected arguments.
- After editing the test, compute its new hash and size (needed for the manifests):
  ```bash
  shasum -a 256 test_warm_m5_teardown_r15_control.py
  wc -c test_warm_m5_teardown_r15_control.py
  ```

### 5. Update the classifier, stop request, and cleanup hash pins

These three files each carry `WRAPPER_SHA256` for their `READY_LINE` construction. Update all three to the new wrapper hash:

- `warm_m5_teardown_r15_journal_classifier.py`
- `warm_m5_teardown_r15_stop_request.py`
- `warm_m5_teardown_r15_cleanup.py`

After editing, compute their new hashes and sizes:
```bash
shasum -a 256 warm_m5_teardown_r15_journal_classifier.py warm_m5_teardown_r15_stop_request.py warm_m5_teardown_r15_cleanup.py
wc -c warm_m5_teardown_r15_journal_classifier.py warm_m5_teardown_r15_stop_request.py warm_m5_teardown_r15_cleanup.py
```

### 6. Regenerate the derivation script

The derivation (`derive_warm_m5_teardown_r15.py`) reconstructs the wrapper from the R14 parent wrapper plus a line-based patch, and reconstructs the classifier and stop helper from base85-encoded payloads. All three need regeneration when the wrapper or classifier/stop helper change.

Run a regeneration script that:

1. Reads the R14 wrapper, normalizes it to R15 naming, and computes a new line-based patch (via `difflib.SequenceMatcher`) to the edited R15 wrapper.
2. Encodes the patch as `zlib.compress` + `base64.b85encode`.
3. Encodes the current classifier and stop helper files as base85 payloads.
4. Updates `WRAPPER_SHA256`, `WRAPPER_PATCH_RAW_SHA256`, `WRAPPER_PATCH_OPERATIONS`, `CLASSIFIER_SHA256`, `STOP_HELPER_SHA256`, `OUTER_SHA256`, and `LAUNCH_SHA256` in the derivation file.
5. Replaces the `WRAPPER_PATCH_B85`, `CLASSIFIER_PAYLOAD_B85`, and `STOP_HELPER_PAYLOAD_B85` string literal blocks.
6. Verifies that `derive_wrapper()`, `derive_classifier()`, `derive_stop_helper()`, `derive_outer()`, and `derive_launch()` each reconstruct the correct file with the correct hash.

After regeneration, compute the derivation file's new hash and size:
```bash
shasum -a 256 derive_warm_m5_teardown_r15.py
wc -c derive_warm_m5_teardown_r15.py
```

Update `DERIVATION_SHA256` in the test file to this new hash.

### 7. Compute the new launcher hash

The launcher file itself changed in step 3, so its SHA256 changed:
```bash
shasum -a 256 warm_m5_teardown_r15_launch.sh
```

Update `LAUNCH_SHA256` in the test file to this new hash.

### 8. Update the manifests and r25 loader

The four qualification manifests and the r25 loader pin every artifact's hash and size. Update all of them with the new hashes and sizes for every file that changed:

- `warm_m5_teardown_r15_author_manifest_*.json`
- `warm_m5_teardown_r15_root_manifest_*.json`
- `warm_m5_teardown_r15_final1_manifest_*.json`
- `warm_m5_teardown_r15_final2_manifest_*.json`
- `r25_live_fire_loader_v1_7aa0b55d.py`

Each manifest has a JSON array of artifacts with `path`, `sha256`, `size`, and `mode` fields. Update the `sha256` and `size` for each changed file.

The r25 loader has a `_spec()` call for each artifact with the hash, size, inode, and device. Update the hash and size. The inode should be re-verified (it may change if the file was recreated rather than edited in place):

```bash
ls -i warm_m5_teardown_r15_control.sh
```

### 9. Update secondary references

Search for any remaining files that reference old hashes:

```bash
cd sessions/fable/work
grep -rl "<old-wrapper-hash>" . | grep -v __pycache__
```

Update any matches (for example, `fire_diagnose_*.sh`, `test_r25_recovery_chain_*.py`).

### 10. Verify no stale hashes remain

```bash
for old in <old-wrapper-hash> <old-launcher-hash> <old-classifier-hash> <old-stop-helper-hash> <old-derivation-hash> <old-test-hash>; do
  hits=$(grep -rl "$old" . 2>/dev/null | grep -v __pycache__ | wc -l | tr -d ' ')
  [ "$hits" -gt 0 ] && echo "STALE: ${old:0:12}... in $hits file(s)" || echo "clean: ${old:0:12}..."
done
```

Every old hash must report clean before the chain is consistent.

### 11. Run the test suite

```bash
# Use the project venv that has pytest
/Users/arunsharma/code/spatial-atlas/.venv/bin/python -m pytest -q -p no:cacheprovider \
  sessions/fable/work/test_warm_m5_teardown_r15_control.py \
  sessions/fable/work/test_warm_m5_teardown_r15_cleanup.py
```

All tests should pass. The one exception is `test_outer_no_tty_is_fixed_stdout_and_silent_stderr`, which expects no controlling TTY and only passes in a headless environment. On a Mac with a terminal it reaches READY and times out; this is a pre-existing environment issue, not a regeneration regression.

### 12. Run the relaunch to verify

```bash
/Users/arunsharma/code/spatial-atlas/sessions/fable/work/warm_m5_teardown_r15_relaunch.sh
```

The cleanup reports `state=absent` (or `cleaned`), then the launcher verifies the new hash chain and starts SSH. If the SSH arguments are correct, the connection reaches the authentication step (Duo prompt for MSI). The `termination=ssh_reserved` failure (exit 255, auth denied) should no longer occur if the arguments fix the auth path.

## Safety constraints

- `sessions/` is gitignored and must never be force-added. The repo is public and this directory holds MSI operational details. See `FABLE_HANDOFF.md` and `CLAUDE_HANDOFF.md` for the hard rules.
- Never push the adjacent SpatialClaw working copy. Its remote is NVlabs upstream.
- Never weaken a gate or fabricate a result. The regeneration changes hashes, not gates.
- No Slurm job wider than two GPUs without explicit owner authorization.
- The wrapper hash chain is a security boundary. Only regenerate when the SSH arguments genuinely need to change, and always verify the full test suite passes before running the launcher against MSI.

## Summary of the hash propagation order

1. Edit the wrapper.
2. New wrapper hash and size.
3. Launcher pinned hash.
4. Test expected hash and expected args.
5. Classifier, stop helper, cleanup hash pins.
6. Derivation patch, payloads, and all hash pins (regenerate via script).
7. Test `DERIVATION_SHA256` and `LAUNCH_SHA256`.
8. Manifests and r25 loader (hash + size for every changed file).
9. Secondary references (diagnose scripts, recovery tests).
10. Verify no stale hashes, run tests, run relaunch.
