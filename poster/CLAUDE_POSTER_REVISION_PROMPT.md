# Paste-Ready Claude Prompt

Copy the text below into Claude from the root of this repository.

```text
You are revising the public spoken narrative and audience Q&A for the Spatial
Atlas research poster.

Work from the root of the spatial-atlas repository. Do not rely on previous
chat context, private local paths, cluster logs, or recalled experiment
results.

Read these files in order:

1. poster/CLAUDE_POSTER_NARRATIVE_HANDOFF.md
2. poster/POSTER_NARRATIVE.md
3. poster/POSTER_QA_PACKET.md
4. poster/spatial_atlas_poster.tex
5. poster/PRINT_PREFLIGHT.md
6. src/agent.py
7. src/fieldwork/handler.py
8. src/fieldwork/reasoner.py
9. src/fieldwork/spatial.py
10. src/fieldwork/vision.py
11. src/mlebench/handler.py
12. src/mlebench/codegen.py
13. src/mlebench/executor.py
14. src/entropy/engine.py
15. src/cost/router.py
16. src/cost/tracker.py
17. src/llm.py
18. src/executor.py
19. src/config.py
20. tests/test_agent.py
21. scenarios/fieldwork/scenario.toml
22. scenarios/mlebench/scenario.toml
23. paper/spatial_atlas.tex
24. paper/spatial_atlas.md
25. ARCHITECTURE.md
26. README.md
27. pyproject.toml

Use the official upstream sources only for SpatialClaw claims:

SpatialClaw repository:
https://github.com/NVlabs/SpatialClaw

SpatialClaw paper:
https://arxiv.org/abs/2606.13673

Spatial Atlas preprint:
https://arxiv.org/abs/2604.12102v2

Apply this evidence hierarchy:

1. The current poster controls visible order, terminology, equations, tables,
   and presentation emphasis.
2. Current source code and tests control claims about behavior implemented in
   public main.
3. The Spatial Atlas preprint controls theory and values explicitly labeled as
   source-reported preprint results.
4. Official SpatialClaw sources control upstream SpatialClaw mechanisms and
   upstream results.
5. Local bridge claims visible on the poster may be described only as local
   integration work that is not independently reproduced by public main.
6. No unpublished cluster result, job identifier, diagnostic measurement,
   private service state, or private experiment status may enter the public
   narrative.

Preserve these claim boundaries:

Spatial Atlas is an Agent-to-Agent server with FieldWorkArena and MLE-Bench
handlers.

The public-main scene graph performs deterministic calculations over
model-estimated 2D coordinates. Do not describe those coordinates as measured
3D geometry. Its bulk relation method fills only missing distances and
preserves extractor-supplied values without recording whether each value was
supplied or derived.

The paper's expected information-gain policy is theoretical. Public-main
FieldWork code instead produces one Strong-tier answer, obtains a fast-tier
confidence estimate, and performs at most one Strong-tier refinement below
0.6. Do not claim calibrated confidence because the repository provides no
calibration evidence.

Treat the FieldWorkArena and MLE-Bench table values as source-reported preprint
results. The public manuscript contains both tables, but the repository does
not include the artifacts needed to reproduce them. Do not say that the GitHub
paper omits the tables.

SpatialClaw is a separate upstream framework. Its 59.9 percent average accuracy
and 11.2-point improvement are upstream SpatialClaw results, not Spatial Atlas
results.

The public main branch does not independently contain the local metric-bridge
implementation or its experiment artifacts. Do not claim that the public
repository reproduces that bridge.

The persistent-kernel connection, evidence-use journal, and post-kernel
verifier remain proposed integration work unless current public code proves
otherwise.

Audit this explicit contradiction:

The poster describes generated-code execution as requiring explicit operator
opt-in and isolated-worker attestation, with dummy fallback remaining opt-in.
Public main does not independently establish those controls.
src/mlebench/executor.py launches a subprocess without an explicit attestation
gate, and src/mlebench/handler.py creates a dummy submission after all initial
attempts fail. The usage tracker records provider-reported tokens but does not
enforce the configured token field as a hard admission budget. Keep these
discrepancies visible. Do not silently write around them.

Tasks:

1. Audit every factual and numerical claim in
   poster/POSTER_NARRATIVE.md and poster/POSTER_QA_PACKET.md.
2. For each claim, identify its supporting repository file or official
   upstream source.
3. Report contradictions before editing.
4. Revise poster/POSTER_NARRATIVE.md into a natural four-to-six-minute spoken
   walk-through that follows the poster from left to right.
5. Include a two-minute version and a thirty-second opening.
6. Refine poster/POSTER_QA_PACKET.md into a public rehearsal packet with
   concise, evidence-bounded answers.
7. Include a claim-audit table with these categories:
   implemented public Spatial Atlas,
   source-reported Spatial Atlas preprint,
   upstream SpatialClaw,
   local integration unavailable in public main,
   proposed integration,
   unsupported or contradictory.
8. Provide a short change log.
9. Edit only poster/POSTER_NARRATIVE.md and poster/POSTER_QA_PACKET.md.
10. Do not edit source code, result values, poster LaTeX, the PDF, or the
    print-preflight record.
11. Do not commit or push.

Writing rules:

Use direct spoken language.
Lead with compute-grounded reasoning.
Explain architecture before equations.
Distinguish deterministic arithmetic from measured geometry.
Label the origin of every numerical result before stating it.
Never use an em dash.
Do not replace an em dash with a semicolon.
Do not invent calibration, deployment, causal, or benchmark claims.
If evidence is absent or conflicting, preserve the uncertainty and flag it.
Use Prof. rather than Professor.
```
