# Claude Handoff: Spatial Atlas Poster Narrative

## Objective

Audit and refine the public spoken narrative for the Spatial Atlas research
poster. Improve accuracy, cadence, transitions, and accessibility without
changing a verified number or blurring the boundary between implemented code,
source-reported preprint results, upstream SpatialClaw results, local
integration work, and proposed integration.

The poster is already print-preflighted. Narrative work should edit
`poster/POSTER_NARRATIVE.md` only unless Arun explicitly requests a new poster
layout or content pass.

## Read order

Work from the repository root and read these files in order:

1. `poster/CLAUDE_POSTER_NARRATIVE_HANDOFF.md`
2. `poster/POSTER_NARRATIVE.md`
3. `poster/spatial_atlas_poster.tex`
4. `poster/PRINT_PREFLIGHT.md`
5. `src/agent.py`
6. `src/fieldwork/handler.py`
7. `src/fieldwork/reasoner.py`
8. `src/fieldwork/spatial.py`
9. `src/fieldwork/vision.py`
10. `src/mlebench/handler.py`
11. `src/mlebench/codegen.py`
12. `src/mlebench/executor.py`
13. `src/entropy/engine.py`
14. `src/cost/router.py`
15. `src/cost/tracker.py`
16. `src/llm.py`
17. `src/executor.py`
18. `src/config.py`
19. `tests/test_agent.py`
20. `scenarios/fieldwork/scenario.toml`
21. `scenarios/mlebench/scenario.toml`
22. `paper/spatial_atlas.tex`
23. `paper/spatial_atlas.md`
24. `ARCHITECTURE.md`
25. `README.md`
26. `pyproject.toml`

Use only the official upstream sources for SpatialClaw:

- Repository: <https://github.com/NVlabs/SpatialClaw>
- Paper: <https://arxiv.org/abs/2606.13673>

Use the public Spatial Atlas preprint for source-reported paper claims:

- Preprint: <https://arxiv.org/abs/2604.12102v2>

Do not rely on prior chat context, private local paths, cluster logs, or recalled
experiment results.

## Evidence hierarchy

1. The current poster controls visible order, terminology, equations, tables,
   and presentation emphasis.
2. Current public source code and tests control claims about behavior
   implemented in public main.
3. The Spatial Atlas preprint controls theory and values explicitly labeled as
   source-reported preprint results.
4. Official SpatialClaw sources control upstream SpatialClaw mechanisms and
   upstream results.
5. Local bridge claims visible on the poster may be described only as local
   integration work that is not independently reproduced by public main.
6. No unpublished cluster result, job identifier, diagnostic measurement,
   private service state, or private experiment status may enter the public
   narrative.

If two sources disagree, show the disagreement. Do not silently reconcile it.

## Claim boundaries

### Implemented in public Spatial Atlas

- Spatial Atlas exposes an Agent-to-Agent server with FieldWorkArena and
  MLE-Bench handlers.
- The current public scene-graph path obtains `position_x` and `position_y`
  from a Strong-tier model through a structured JSON extraction prompt.
- The graph implements deterministic distance calculation, relation-distance
  completion, radius queries, constraint checks, and fact-sheet serialization.
- The graph arithmetic uses model-estimated 2D coordinates. It is not measured
  3D geometry.
- The current FieldWork path produces one Strong-tier answer, obtains a
  fast-tier confidence estimate, and performs at most one Strong-tier
  refinement below 0.6.
- The current FieldWork path does not invoke the expected information-gain
  action selector.
- The repository does not provide calibration evidence for the confidence
  estimate.
- The public MLE handler attempts bounded repair, parses
  `VALIDATION_SCORE`, requests at most two score-driven revisions under
  default configuration, and retains only an improvement under the configured
  metric direction.

### Public-main contradiction that must remain visible

The poster describes generated-code execution as requiring explicit operator
opt-in and isolated-worker attestation, with dummy fallback remaining opt-in.
Public main does not independently establish those exact controls.
`src/mlebench/executor.py` launches a subprocess without an explicit attestation
gate, and `src/mlebench/handler.py` creates a dummy submission after all initial
attempts fail.

Do not describe the stricter poster gate as public-main behavior unless the
matching hardened implementation is published and verified.

### Source-reported Spatial Atlas preprint

- FieldWorkArena overall configuration: Factory 72 percent, Warehouse 68
  percent, Retail 74 percent.
- Without the Spatial Scene Graph: Factory 51 percent, Warehouse 44 percent,
  Retail 55 percent.
- MLE-Bench overall: 82 percent valid submissions and 32 percent medals across
  75 competitions.
- The answer-entropy, expected information-gain, reflection, and complete
  model-routing equations are paper-level theory.

These values are source-reported preprint results. They are not reproduced by
public main or by public integration artifacts.

### Upstream SpatialClaw

- SpatialClaw is a separate, training-free framework.
- Its official implementation uses a persistent Jupyter kernel, one
  AST-checked Python cell per reasoning step, perception tools, scientific
  libraries, intermediate observation feedback, and `ReturnAnswer`.
- The SpatialClaw paper reports 59.9 percent average accuracy across 20 spatial
  reasoning benchmarks.
- The SpatialClaw paper reports an 11.2-point improvement over the prior best
  spatial agent.

These are upstream SpatialClaw results. They are not Spatial Atlas results.
The projects use different tasks, metrics, models, and perception stacks.

### Local bridge described by the poster

- The poster describes a metric bridge using object masks, reconstructed
  points, mask erosion, confidence-qualified finite points, deterministic XZ
  voxel representatives, and a symmetric directed fifth-percentile
  nearest-surface estimator.
- The poster describes checks for metric range, units, provenance, protocol
  identity, and evidence hashes.
- The public main branch does not contain the local bridge implementation or
  its experiment artifacts.

Describe this as local Spatial Atlas integration work that public main does not
independently reproduce. Do not add unpublished measurements or validation
status.

### Proposed integration

- The complete persistent-kernel connection between the SpatialClaw action
  loop and the Atlas evidence state remains proposed.
- The evidence-use journal remains proposed.
- The post-kernel verifier remains proposed.
- Do not state or imply that the complete Figure 2 pipeline is deployed.

## Narrative priorities

- Open with the compute-grounded reasoning thesis.
- Explain the architecture before the equations.
- Treat the scene-graph limitation as a central motivation.
- Explain the difference between deterministic arithmetic and measured
  geometry.
- Separate the paper's entropy policy from the current FieldWork controller.
- Label both result tables before quoting any value.
- Introduce SpatialClaw as a separate framework before discussing integration.
- Surface the public MLE gate contradiction instead of concealing it.
- Close on traceability, recomputation, revision, and fail-closed evidence.

## Required output

Return:

1. A polished four-to-six-minute spoken walk-through with light pointing cues.
2. A two-minute version.
3. A thirty-second opening.
4. Five likely audience questions with concise, evidence-bounded answers.
5. A claim-audit table with these categories:
   implemented public Spatial Atlas,
   source-reported Spatial Atlas preprint,
   upstream SpatialClaw,
   local integration unavailable in public main,
   proposed integration,
   unsupported or contradictory.
6. A short change log.

## Style and operating rules

- Use direct spoken language.
- Lead with the answer.
- Never use an em dash.
- Do not replace an em dash with a semicolon.
- Use `Prof.` rather than `Professor`.
- Do not change or extrapolate a number.
- Do not invent calibration, deployment, causal, or benchmark claims.
- If evidence is absent or conflicting, preserve the uncertainty and flag it.
- Do not add private paths, service addresses, raw logs, tokens, or job IDs.
- Do not edit source code, results, poster layout, or the PDF.
- Do not commit or push.
