# Spatial Atlas Poster Narrative

This is the final public speaking narrative for the poster. It follows the
poster from left to right and keeps five evidence classes separate:

1. behavior implemented in public Spatial Atlas
2. results reported by the Spatial Atlas v2 preprint
3. results and mechanisms reported by upstream SpatialClaw
4. local integration work described on the poster but unavailable in public main
5. proposed integration

## Fifteen-second hook

Language models can answer a distance question without computing the distance.
Spatial Atlas changes the order: write evidence into an inspectable state,
compute the available relations, then ask the model to answer.

## Thirty-second opening

Spatial Atlas starts from one question: when an agent can compute a fact, why
ask a language model to guess it? We call the alternative compute-grounded
reasoning. The system turns scene evidence into typed state, runs deterministic
spatial or machine-learning operations, and gives the resulting facts to the
model before answer generation. The poster shows the two-handler Atlas design,
results reported in the Spatial Atlas v2 preprint, and a path toward
SpatialClaw's stateful code-action loop and perception tool wrappers.

## Ninety-second version

[Point to Overview.]

Spatial Atlas is built around compute-grounded reasoning: compute what can be
computed before generating what must be generated. Answer-only evaluation
cannot tell whether a spatial answer came from evidence or a plausible guess.

[Point to Figure 1.]

One Agent-to-Agent server routes to two handlers. FieldWorkArena turns
model-extracted entities, relations, positions, and constraints into a typed
scene graph. Code then derives selected distances, nearby-object queries,
violations, and a fact sheet. MLE-Bench applies the same compute-first pattern
to generated pipelines and their validation scores.

[Point to Section 3.]

Here is the key qualification. Public main performs deterministic arithmetic
over model-estimated two-dimensional coordinates. The calculation is
repeatable and inspectable, but the input is not measured geometry.

[Point to Table 1.]

The v2 preprint reports FieldWorkArena values of 72, 68, and 74 percent. Its
scene-graph ablation differs by 21, 24, and 19 points. Those values are
source-reported, not reproduced by public main, and the public artifacts do not
show that geometry alone caused the difference.

[Point to Figure 2.]

SpatialClaw motivates the next step. Its persistent code-action loop lets an
agent inspect intermediate outputs and revise its analysis. The local metric
bridge described here is not in public main. The kernel link, evidence journal,
and verifier remain proposed.

The take-away is accountability: make evidence and computation visible enough
to inspect, recompute, and reject.

## Two-minute version

[Point to Overview.]

Spatial Atlas asks a simple question: when an agent can compute a fact, why ask
a language model to guess it? Compute-grounded reasoning builds explicit
evidence, performs deterministic operations, and passes the resulting facts to
the model before generation.

[Point to Figure 1.]

One Agent-to-Agent server routes to two handlers. FieldWorkArena builds a typed
scene graph, derives selected spatial facts, and serializes a fact sheet.
MLE-Bench generates and repairs a pipeline, parses its validation score, and
keeps a refinement only when that score improves.

[Point to the scene-graph DAG.]

The important distinction is between deterministic arithmetic and correct
geometry. In public main, a Strong-tier model supplies `position_x` and
`position_y`. The graph performs repeatable Euclidean arithmetic over those
values, but the coordinates remain model estimates.

[Point to the entropy equations and algorithm.]

The equations show the paper's information-theoretic policy. Public FieldWork
code is narrower: one Strong-tier answer, one fast-tier confidence estimate,
and at most one Strong-tier refinement below 0.6. The public path does not
invoke expected information gain or demonstrate calibration.

[Point to Tables 1 and 2.]

The v2 preprint reports FieldWorkArena overall values of 72, 68, and 74 percent.
The reported no-scene-graph row is 21, 24, and 19 points below the reported
full-configuration row. MLE-Bench reports 82 percent valid submissions and 32
percent medals across 75 competitions. These are source-reported results.
Public main contains the tables but not the run artifacts needed to reproduce
them.

[Point to Figure 2 and Section 7.1.]

SpatialClaw is a separate upstream project. Its paper reports 59.9 percent
average accuracy across 20 benchmarks and an 11.2-point improvement over the
prior best spatial agent. Those are SpatialClaw results. The poster describes a
local mask-and-point-map bridge that is absent from public main. The kernel
connection, evidence-use journal, and verifier remain proposed.

The contribution is the ordering and accountability of the reasoning process:
make evidence explicit, compute over it, then generate.

## Five-minute walk-through

### 1. Problem and thesis

[Point to the title and Section 1.]

The question behind this poster is simple. When an agent can compute a fact,
why ask a language model to guess it? Spatial Atlas calls the alternative
compute-grounded reasoning, or CGR. It resolves answerable subproblems through
code before a model interprets the evidence or writes the response.
Answer-only evaluation cannot distinguish a grounded estimate from a lucky
guess. CGR makes the intermediate evidence and operations inspectable.

### 2. Two handlers, one server

[Point to Figure 1.]

One Agent-to-Agent server routes to two benchmark-oriented handlers.
FieldWorkArena handles multimodal spatial questions. MLE-Bench handles
machine-learning engineering tasks. Both use the same entry point and
model-access layer.

The FieldWork branch turns extracted entities, positions, relations, zones, and
constraints into a typed graph. The MLE branch generates a pipeline, repairs
failed executions from error context, parses a `VALIDATION_SCORE`, and keeps a
refinement only when the parsed score improves under the analyzer-supplied
direction as interpreted by a keyword heuristic. These are two instances of
the same ordering: establish state, compute, then generate.

### 3. Scene-graph computation

[Point to the entity equations and scene-graph DAG.]

In public main, a Strong-tier model supplies `position_x` and `position_y`.
`compute_distance` applies Euclidean arithmetic. The bulk method fills only
missing relation distances. Radius queries, constraint checks, and fact-sheet
serialization then operate on the stored state.

This separation is useful, but incomplete. An extractor-supplied distance is
preserved without recording whether it was supplied or derived. More
fundamentally, the coordinates are model-estimated and two-dimensional. The
arithmetic is deterministic. The geometry is not measured. A repeatable
calculation can still be wrong when its inputs are wrong.

### 4. Paper policy and public behavior

[Point to the entropy equations, then the paragraph above the algorithm.]

The entropy equations state the paper's broader policy: choose the action
expected to reduce answer uncertainty, then route or reflect by confidence.
That is not the current FieldWork path. Public main starts with one Strong-tier
answer, requests one fast-tier confidence estimate, and performs at most one
Strong-tier refinement below 0.6. It does not invoke the expected
information-gain selector, and the confidence estimate has not been shown to be
calibrated.

### 5. MLE-Bench boundary

[Point to Section 4.]

The MLE path has another visible evidence boundary. The poster describes
explicit execution opt-in, isolated-worker attestation, and opt-in dummy
fallback. Public main does not establish those controls. It launches a
subprocess, inherits the surrounding environment, and creates a dummy
submission after all initial attempts fail.

Public main does support execution timeouts, repair from captured errors,
validation-score parsing, and score-gated rollback. But the score comes from
model-generated code, so the rollback is only as trustworthy as its validation
procedure. The leakage checks are prompted safeguards, not an independently
enforced audit.

### 6. Source-reported results

[Point to Tables 1 and 2.]

Both tables are source-reported results from the Spatial Atlas v2 preprint.
Public main contains the tables, but not the run artifacts needed to reproduce
them.

For FieldWorkArena, the preprint reports 72, 68, and 74 percent for the full
configuration. Its scene-graph ablation reports 51, 44, and 55 percent, a
difference of 21, 24, and 19 points. The paper labels this as a component
ablation. The public artifacts do not include a frozen run manifest proving
that only one factor changed, and the comparison does not separate correct
geometry from structured prompting. A schema-matched corruption control would
do that, but it has not been reported.

For MLE-Bench, the preprint reports 82 percent valid and 32 percent medals
across 75 competitions. The displayed category counts sum to 75, and weighted
means of the rounded rates round to 82 and 32. That is only an arithmetic
consistency check. It does not recover raw counts, attempt protocol, variance,
or confidence intervals.

### 7. Why SpatialClaw enters

[Point to Figure 2 and the Upstream Context box.]

The scene-graph limitation motivates the third column. SpatialClaw is a
separate, training-free framework from Cho and colleagues. It uses code as the
action interface: one AST-checked Python cell per step in a persistent kernel,
with intermediate outputs available to later actions.

Its paper reports 59.9 percent average accuracy across 20 benchmarks and an
11.2-point improvement over the prior best spatial agent. Those are upstream
SpatialClaw results, not Spatial Atlas results.

### 8. Integration status and close

[Point to Section 7.1, then the Figure 2 caption.]

The poster describes local work that combines object masks with a
Depth-Anything-3 point map, erodes the masks, builds horizontal XZ voxel
representatives, and estimates a symmetric surface gap from directed fifth
percentiles. It also describes range, unit, provenance, protocol, and evidence
hash checks.

That bridge and its artifacts are not in public main. The persistent-kernel
link, evidence-use journal, and post-kernel verifier also remain proposed.
Figure 2 is an integration design, not a deployed end-to-end claim.

[Point to the final take-away.]

The contribution is not that every estimate is correct. It is a boundary that
makes evidence explicit and derived facts inspectable. The direction is toward
answers that can be recomputed, revised, and traced to the evidence that
produced them.

## Closing options

### Technical close

The research question is no longer only whether the model returned the right
answer. It is whether the answer can be traced to evidence and recomputed when
that evidence changes.

### Systems close

The practical contribution is a boundary: models may propose evidence, but
deterministic code should own derived calculations, and a verifier should
decide whether the evidence is fresh enough to count.

### Collaboration close

The next experiment is a schema-matched corruption test that separates the
benefit of structured prompting from the benefit of correct geometry. That is
the evaluation design on which I would most value feedback.

## Pointing map

| Spoken topic | Poster location |
| --- | --- |
| CGR thesis and answer-only limitation | Section 1 |
| Two-handler architecture | Figure 1 |
| Typed state and deterministic operations | Section 3 |
| Paper policy versus public behavior | Top of column 2 |
| FieldWork and MLE procedures | Section 4 |
| Source-reported values | Tables 1 and 2 |
| Limitation of estimated coordinates | Bottom of Section 5 |
| Upstream SpatialClaw mechanism and results | Section 6 |
| Local bridge described by the poster | Section 7.1 |
| Proposed contracts and final take-away | Section 7.2 |

## Claim card

### Safe to state as public implementation

- One Agent-to-Agent server routes to FieldWorkArena and MLE-Bench handlers.
- The scene graph implements deterministic two-dimensional arithmetic, radius
  queries, constraint checks, and fact-sheet serialization. It fills only
  missing relation distances and does not record whether each distance was
  supplied or derived.
- Public FieldWork uses one Strong-tier answer, a fast-tier confidence
  estimate, and at most one Strong-tier refinement below 0.6.
- Public MLE code attempts repair, parses a pipeline-emitted validation score,
  requests at most two refinements by default, and retains only a parsed score
  improvement.

### State only as source-reported preprint results

- FieldWorkArena overall: 72, 68, and 74 percent.
- Without the Spatial Scene Graph: 51, 44, and 55 percent.
- MLE-Bench overall: 82 percent valid and 32 percent medals across 75
  competitions.
- The complete entropy, expected-information-gain, and multi-tier routing
  formulation.

### State only as upstream SpatialClaw

- Training-free code-action framework with a persistent kernel.
- 59.9 percent average accuracy across 20 benchmarks.
- 11.2-point improvement over the prior best spatial agent.

### State only as local poster-described integration

- Mask erosion, confidence-qualified reconstructed points, XZ voxel
  representatives, symmetric fifth-percentile surface-gap estimation, and
  strict evidence validation.
- Public main does not contain or reproduce this bridge.

### State only as proposed

- Complete persistent-kernel connection to Atlas evidence state.
- Evidence-use journal.
- Post-kernel verifier.

### Never say

- The public repository reproduces Tables 1 or 2.
- The public repository paper omits the tables. At the current public commit,
  both `paper/spatial_atlas.tex` and `paper/spatial_atlas.md` contain them.
- The scene-graph coordinates are measured geometry.
- The self-grade is calibrated.
- Public FieldWork executes the complete entropy-routing algorithm.
- Public main implements the poster's hardened MLE execution gate.
- Figure 2 is fully deployed.
- SpatialClaw's 59.9 percent or 11.2-point result belongs to Spatial Atlas.
- A shuffled-evidence or placebo arm has already been run.
