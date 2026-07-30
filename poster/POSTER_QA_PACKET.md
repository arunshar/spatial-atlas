# Spatial Atlas Poster Q&A Packet

This packet is designed for live preparation. It retains the strongest
questions from an earlier preparation draft, removes claims that do not match
public main, and adds direct answers for the source, geometry, systems, and
attribution seams visible on the poster.

Implementation claims were audited against source commit
`d29c8c30a3cbf463fa120c825b3c074a3a07e923`. The prep-document update that
contains this packet does not change the audited implementation.

## 0. Evidence rules

Use these labels before giving a number or capability:

- **Public implementation** means the behavior is visible in `src/` at the
  current public commit.
- **Source-reported preprint result** means the value appears in Spatial Atlas
  v2 but the public repository does not include the run artifacts needed to
  reproduce it.
- **Upstream SpatialClaw result** means the claim belongs to Cho and colleagues,
  not Spatial Atlas.
- **Local poster-described integration** means the poster says the work exists,
  but public main does not contain it.
- **Proposed** means it is a design or next step.

When a source is missing, say, "I do not have evidence for that claim in the
public artifact." Do not fill the silence with an estimate.

## 1. Numbers to know

### Table 1 arithmetic

| Comparison against Overall | Factory | Warehouse | Retail |
| --- | ---: | ---: | ---: |
| w/o Spatial Scene Graph | -21 | -24 | -19 |
| w/o Entropy-Guided Reasoning | -7 | -8 | -7 |
| w/o Florence-2 | -9 | -10 | -8 |
| GPT-4V Direct | -24 | -27 | -22 |

The GPT-4V Direct row is exactly three points below the no-scene-graph row in
all three columns. That is an arithmetic observation. The poster provides no
variance or task counts that would support a mechanistic interpretation of the
three-point pattern.

### Table 2 arithmetic

The displayed competition counts sum to 75:

```text
32 + 18 + 12 + 8 + 5 = 75
```

Weighting the displayed, rounded category rates by the displayed competition
counts gives:

```text
Valid = (91x32 + 78x18 + 65x12 + 85x8 + 72x5) / 75
      = 81.81, which rounds to 82

Medal = (42x32 + 28x18 + 15x12 + 35x8 + 20x5) / 75
      = 32.11, which rounds to 32
```

This is an internal arithmetic check. It does not prove that the source used
this aggregation procedure, and it does not reveal raw successful counts,
attempt counts, repeated runs, variance, or confidence intervals.

## 2. Fast opening questions

### 2.1 What is this?

**Answer**

Spatial Atlas asks language models to write evidence into an inspectable state,
then uses code to compute selected relations before the model generates an
answer. I call that compute-grounded reasoning.

**Point at**

Section 1, then Figure 1.

**Avoid**

Opening with the full paper title or claiming that every spatial value is
measured.

### 2.2 What is the contribution in one sentence?

**Answer**

The contribution is a compute-then-generate boundary: extraction proposes
evidence, deterministic code derives selected facts, and the final model
receives a fact sheet instead of being asked to regenerate the same relations
in prose.

**Point at**

Section 3 and the scene-graph DAG.

### 2.3 What reported comparison should I discuss?

**Answer**

The strongest contribution to lead with is the typed computation boundary. For
the table, the reported no-scene-graph row is 21, 24, and 19 points below the
reported full-configuration row across Factory, Warehouse, and Retail. The
paper presents this as a component ablation, but the public artifacts do not
include the run manifest needed to verify that only one factor changed. The
comparison also does not prove that correct geometry alone caused the
difference, and the public repository does not reproduce the table.

**Point at**

Table 1 and the Section 5 title, Source-Reported Results.

### 2.4 Why is MLE-Bench on a spatial poster?

**Answer**

MLE-Bench is not evidence about geometry. It tests whether the same
compute-first control pattern can manage a different deterministic object, a
pipeline-emitted validation score. The reusable claim is about the shared
Agent-to-Agent entry point and model-access layer. Public main does not
implement a shared hard-budget gate or general evidence journal. Two
benchmarks are not enough to prove generality.

**Point at**

Figure 1, then the rollback equation in Section 4.

### 2.5 Every agent calls tools. What is different here?

**Answer**

The intended difference is separation of stages. Extraction establishes
entities and initial relations. Deterministic code then derives or completes
selected fields before the final language model sees the fact sheet. The
current implementation still trusts model-estimated coordinates and can also
preserve a relation distance supplied directly by the extractor, so the
evidence provenance is not yet strong enough.

**Point at**

Section 3, then the limitation at the bottom of Section 5.

## 3. Evaluation and provenance

### 3.1 Are Tables 1 and 2 reproduced by the GitHub repository?

**Answer**

No. The public repository includes the v2 manuscript and the displayed tables,
but it does not include sealed task-level run artifacts or a matching current
pipeline that regenerates the values. I describe them as source-reported
preprint results.

**Avoid**

Saying the public repository paper omits the tables. At the current public
commit, both `paper/spatial_atlas.tex` and `paper/spatial_atlas.md` contain
them.

### 3.2 Which pipeline produced Table 1?

**Answer**

The v2 paper describes Florence-2 preprocessing and positions derived from
bounding-box centroids. Public main still uses Florence-2 detections as context
when available, but the scene graph itself requests `position_x` and
`position_y` from a Strong-tier JSON extraction call. The poster prints that
difference. Public main should not be presented as the exact pipeline behind
Table 1.

**Point at**

The FieldWorkArena paragraph in Section 4.

### 3.3 Where are the confidence intervals?

**Answer**

They are not reported. The poster also does not provide per-environment task
counts or repeated-run variance. I treat the rows as reported point estimates,
not as uncertainty-qualified comparisons.

**Avoid**

Saying the changes are too large for intervals to matter.

### 3.4 Are Factory, Warehouse, and Retail independent replications?

**Answer**

No. They are three reported domain columns evaluated with a shared system. The
poster provides no basis for treating the columns as independent replications.

### 3.5 What does GPT-4V Direct control for?

**Answer**

The v2 table presents it as a direct VLM baseline. It changes more than the
scene-graph component, so it is not a matched same-model comparison. I do not
interpret the 24, 27, and 22 point gaps to GPT-4V Direct as a clean estimate of
the graph effect.

### 3.6 Why are some Table 2 percentages incompatible with whole competition counts?

**Answer**

The displayed percentages are rounded, and the source does not expose the raw
counts or attempt protocol needed to reconstruct every cell. Under ordinary
nearest-percentage rounding, four of the ten category cells are compatible
with one binary outcome per listed competition: Tabular valid, NLP valid, NLP
medal, and Other medal. Six are not. I can also verify that the category counts
sum to 75 and that weighted means of the rounded rates round to the Overall
row. The public artifacts do not provide enough protocol detail to identify
the actual denominator.

### 3.7 Does Table 1 show that computing geometry caused the gain?

**Answer**

No. The paper labels the comparison as an ablation, but public artifacts do not
document the exact intervention or provide a frozen run manifest. Removing a
graph can change computed relations, prompt structure, fact-sheet content, and
token allocation together. A stronger test would keep the schema and prompt
budget fixed while corrupting only the coordinates or relation values. That
control has not been reported.

### 3.8 What result would weaken the CGR thesis?

**Answer**

If a schema-matched fact sheet with shuffled geometry recovered most of the
reported difference, then correct geometry would not explain most of that
difference. Further controls would be needed to identify what does. A second
test would compare the local reconstructed-geometry path with the
estimated-coordinate path on labeled measurements. Neither test appears on
this poster.

### 3.9 Is the no-entropy row actually an entropy ablation?

**Answer**

The v2 paper labels the row "Without EG." The public artifacts do not contain
the run manifest needed to establish exactly what the historical arm disabled.
Public FieldWork code does not evaluate the displayed expected-information-gain
objective. The closest current mechanism is confidence-gated reflection, but I
cannot claim that the historical row removed exactly that implementation.

### 3.10 Does Overall in Table 1 mean a pooled score?

**Answer**

No. In Table 1, Overall names the full-system configuration. It is not a pooled
aggregate across Factory, Warehouse, and Retail. The label can be
misinterpreted if that distinction is not stated.

### 3.11 What evidence is missing for a reproducible Table 1?

**Answer**

The paper and poster report the same FieldWorkArena table, but the public
repository does not provide the task-level outputs, dataset snapshot, run
configuration, model versions, prompts, seeds, or script that regenerates it.
The paper and current source also differ on the Strong-tier default and the
coordinate-extraction path. A frozen run manifest is required.

## 4. Scene graph and geometry

### 4.1 What does `compute_distance` actually compute?

**Answer**

It computes a two-dimensional Euclidean value between stored positions. Public
main labels the positions as estimated meters, but it provides no scale
recovery or metric validation for them. The result is deterministic for fixed
inputs and is not a physical measurement.

### 4.2 Does the graph recompute every relation distance?

**Answer**

No. `compute_all_distances()` fills a distance only when the relation distance
is missing. If extraction supplies `distance_meters`, public main preserves
that value. The relation does not store whether the number was supplied by the
model or derived from coordinates, which is a real provenance gap.

### 4.3 In what sense is this computation rather than a computation of an invented number?

**Answer**

Determinism guarantees repeatability for fixed inputs. It does not guarantee
that the inputs are geometrically correct. The contribution I defend on the
public path is an inspectable derivation and serialization process, not metric
accuracy.

### 4.4 Defend bounding-box centroids.

**Answer**

I do not defend them as surface geometry. They can be useful for coarse image
ordering, but a center-to-center value is not clearance between object
surfaces. Long, thin, partially visible, and overlapping objects expose that
failure directly.

### 4.5 What happens when a supplied relation distance conflicts with a coordinate-derived value?

**Answer**

In public main, the supplied relation value wins because missing values alone
are completed. There is no consistency check and no source field on
`SpatialRelation`. The repair is to store source, units, confidence, and
derivation method separately, then reject or reconcile conflicts before
constraint checking.

### 4.6 What uncertainty accompanies a reported distance?

**Answer**

None is calibrated or reported. A defensible uncertainty model would need
measured ground truth and calibration stratified by range, occlusion, object
extent, view support, and scene type. Dispersion within a reconstructed point
set would not by itself capture correlated segmentation or scale bias.

### 4.7 Why project onto XZ?

**Answer**

The local poster-described bridge uses XZ as the horizontal plane, which
focuses the estimator on horizontal clearance. That choice discards height.
Objects on different rack levels can therefore look close in the projected
plane even when the three-dimensional relationship is different.

### 4.8 Does the fifth percentile solve occlusion?

**Answer**

No. It reduces sensitivity to isolated low-distance points compared with a raw
minimum. It does not recover an unseen surface, correct segmentation leakage,
or remove systematic reconstruction bias.

### 4.9 Why use the minimum of the two directed fifth percentiles?

**Answer**

It produces a symmetric near-contact statistic while reducing the influence of
one extreme point. The poster does not show that this estimator satisfies the
axioms of a mathematical metric. Its behavior can depend on mask extent, point
density, voxel size, and the chosen quantile.

### 4.10 Is Depth-Anything-3 measured geometry?

**Answer**

No. The poster-described local bridge treats its reconstructed point map as
metric-scale input. That output is still model-estimated, and its scale
properties depend on the selected model and mode. Dense reconstructed points
provide a richer substrate than two generated coordinates, but they remain
vulnerable to scale, visibility, segmentation, and reconstruction errors.

### 4.11 Are you estimating true object clearance?

**Answer**

The local design estimates clearance between visible reconstructed surfaces.
It cannot recover unseen object extent without additional views or a shape
prior. A safety-oriented version should expose coverage and refuse a claim when
the relevant surfaces are not observed.

### 4.12 What happens when a PPE attribute is absent?

**Answer**

Public `check_constraints()` defaults missing PPE, hard-hat, and safety-vest
attributes to compliant values. Missing evidence can therefore suppress a
violation. That is not fail-closed behavior and should be changed to an
unknown-state policy before safety deployment.

## 5. Entropy and model routing

### 5.1 Does the deployed controller compute expected information gain?

**Answer**

No. The expected-information-gain equations and the fourteen-step policy are
the paper formulation. Public FieldWork code produces one Strong-tier answer,
uses a fast-tier call to estimate confidence, and performs at most one
Strong-tier refinement below 0.6.

### 5.2 Is the self-grade calibrated?

**Answer**

No calibration evidence is present in the repository or on the poster. If the
grading response cannot be parsed, the code returns 0.5, which also shows that
the value is a routing heuristic rather than a calibrated probability.

### 5.3 Where did 0.6 come from?

**Answer**

The public artifact provides no threshold sweep or reliability curve. I treat
0.6 as a hand-set operating point, not a tuned or statistically justified
threshold.

### 5.4 Why print the full theory if it is not deployed?

**Answer**

The equations state the intended objective and make the approximation gap
visible. The operational answer is shorter: one answer, one self-grade, and at
most one retry. If the visual emphasis suggests the theoretical policy is the
deployed system, that is a presentation limitation I should acknowledge.

## 6. MLE-Bench and system behavior

### 6.1 Is generated code execution fail-closed in public main?

**Answer**

Not in the strict form printed on the poster. Public main does not contain the
explicit execution opt-in or isolated-worker attestation gate described there.
Its executor starts a subprocess and inherits the surrounding process
environment. I treat the poster's hardened gate as a poster-described local
hardening path that is absent from the current public branch.

### 6.2 Is the executor a security sandbox?

**Answer**

No. Public main provides a subprocess and timeout. It is not a complete
isolation boundary, and it should not be described as a sandbox.

### 6.3 What happens after all initial MLE attempts fail?

**Answer**

Public main automatically creates a dummy submission. That contradicts the
poster sentence saying dummy fallback remains opt-in. The discrepancy is
documented in the public narrative and should not be hidden in conversation.

### 6.4 What does the rollback rule guarantee?

**Answer**

It guarantees that the handler keeps the better parsed `VALIDATION_SCORE`
under an analyzer-supplied direction interpreted by a keyword heuristic. The
heuristic defaults to maximize when it does not recognize a minimizing term.
The score is printed by model-generated pipeline code, so the rule is monotone
on that proxy, not necessarily on true held-out performance.

### 6.5 What prevents validation leakage?

**Answer**

The code-generation prompt can include checks for ID overlap, row
fingerprints, temporal ordering, and identical media bytes. Those are
prompt-specified behaviors. Public main does not independently enforce the
audit or verify the validity of the pipeline's split.

### 6.6 How many attempts and refinements occur?

**Answer**

Public default configuration allows three initial execution attempts in total.
After a successful run with a parseable validation score, it permits at most
two score-driven refinement attempts. The code checks the 900-second
refinement budget before starting an iteration. A started iteration can run
past that point, subject to its separate 600-second subprocess timeout, so 900
seconds is not a strict end-to-end wall-clock bound.

### 6.7 What are the p95 latency and cost?

**Answer**

I do not have a verified p95 or cost artifact for the poster. The v2 manuscript
prints average cost and latency values, but this public package does not include
sealed run evidence that lets me validate them. I will not turn those values
into a stronger live claim.

### 6.8 What breaks first under real traffic?

**Answer**

Public main does not expose enough validated load-test evidence to answer with a
measured bottleneck. The likely pressure points are long-running generated-code
subprocesses, persistent per-context agent state, provider rate limits, and
cost tracking that records usage but does not reserve a budget before a call.
Those are engineering hypotheses, not measured traffic results.

### 6.9 Is the configured 150,000-token field enforced?

**Answer**

No. Public main records provider-reported usage, but the active execution path
does not call the tracker's budget check. The field supports accounting, not
admission control.

### 6.10 Can a client cancel a long-running Agent-to-Agent task?

**Answer**

Not through the public executor. Its cancellation method reports that the
operation is unsupported.

### 6.11 Can a handler error still end with a completed task state?

**Answer**

Yes. The agent can catch an exception and add an error artifact without
propagating the exception. The executor can then mark the task complete. Task
status should therefore not be treated as a successful outcome without
checking the artifact.

### 6.12 Can the final model prove which facts it used?

**Answer**

No. Public main includes the fact sheet in the prompt, but it has no
evidence-use journal or verifier that records which facts supported the
answer. Section 7.2 proposes those controls.

## 7. SpatialClaw integration and attribution

### 7.1 Is SpatialClaw part of Spatial Atlas?

**Answer**

No. SpatialClaw is a separate upstream project from Cho and colleagues at
NVIDIA and KAIST. The poster presents its mechanism and results as upstream
context, then proposes an integration with Atlas.

### 7.2 What is SpatialClaw's contribution?

**Answer**

SpatialClaw treats code as the action interface. Its agent writes one
AST-checked Python cell per step into a persistent Jupyter kernel, observes
intermediate outputs, and can compose or revise operations before returning an
answer.

### 7.3 Whose results are 59.9 percent and 11.2 points?

**Answer**

They are SpatialClaw results. Its paper reports 59.9 percent average accuracy
across 20 spatial reasoning benchmarks and an 11.2-point improvement over the
prior best spatial agent. They are not Spatial Atlas results and should never
be compared directly with Tables 1 or 2.

### 7.4 Which parts of Figure 2 are deployed?

**Answer**

The poster describes the local metric bridge as implemented. Public main does
not contain that bridge, so the public artifact cannot demonstrate it. The
persistent-kernel link, evidence-use journal, and post-kernel verifier are
proposed.

### 7.5 What exactly does the local bridge add?

**Answer**

According to Section 7.1, it adds mask erosion, confidence-qualified point
filtering, deterministic horizontal voxel representatives, a symmetric
fifth-percentile surface-gap estimator, and strict validation before returning
evidence. These are poster-described local claims, not public-main
implementation claims.

### 7.6 Does SpatialClaw supply all the perception models?

**Answer**

SpatialClaw exposes wrappers around SAM3 and Depth-Anything-3, plus geometry
utilities. The underlying models have their own authors and licenses. The
accurate attribution is to SpatialClaw's action interface and wrappers, not to
NVIDIA as the sole author of every perception model.

### 7.7 Are you affiliated with NVIDIA?

**Answer**

My listed affiliation on this poster is the University of Minnesota.
SpatialClaw is separate upstream work by Cho and colleagues from NVIDIA
Research and KAIST. The NVIDIA mark acknowledges that upstream context. It
does not assign me authorship of SpatialClaw or establish sponsorship.

### 7.8 Is this just SpatialClaw with a wrapper?

**Answer**

No. The left two columns describe Spatial Atlas: the two-handler A2A server,
typed scene graph, graph operations, FieldWork control path, MLE repair and
score-comparison loop, and source-reported preprint tables. The third column is
an integration direction. Public main does not port SpatialClaw's persistent
code-action runtime.

## 8. Skeptic and recruiter questions

### 8.1 Take away OpenAI, SpatialClaw, and the A2A SDK. What did you build?

**Answer**

I built the system layer that structures evidence and decides which work code
must perform before generation. In public main that includes task routing, the
typed scene graph and its query methods, fact-sheet serialization, the current
confidence-gated FieldWork path, MLE repair, score parsing, and score-gated
rollback. The local bridge described on the poster adds a surface-gap
estimator, but it is not in public main.

### 8.2 Does a larger model make this unnecessary?

**Answer**

A larger model may improve spatial answers. It does not solve the evaluation
problem that motivates CGR. Answer-only scoring still cannot distinguish an
answer supported by inspected evidence from a plausible guess. The claim is
about evidence accountability, not a permanent ceiling on model capability.

### 8.3 What is the weakest claim on the poster?

**Answer**

The clearest weakness is the gap between the complete entropy policy and the
current one-refinement controller. The second is the gap between the hardened
MLE gate described by the poster and public main. I would rather name both
before a visitor finds them.

### 8.4 What is the strongest defensible contribution?

**Answer**

The typed, inspectable computation path. It separates model-proposed evidence
from deterministic derivation and final generation. The public code is
imperfect, especially on provenance and missing evidence, but the boundary is
specific enough to inspect and improve.

### 8.5 What should happen next?

**Answer**

First, run a schema-matched corruption control that holds prompt structure
fixed and changes only geometric correctness. Second, publish the metric bridge
with tests and refusal accounting. Third, connect a persistent action loop,
evidence journal, and verifier so an answer counts only when its consumed
evidence is explicit and current.

### 8.6 How do you answer a question you cannot verify?

**Answer**

Say: "That value is not available in the public artifact, and I do not want to
guess. I can tell you what file or experiment would answer it." Then name the
missing evidence in one sentence and return to the poster.

## 9. Standing at the poster

### A visitor is still walking

Say:

> Language models can answer distance questions without measuring or computing
> a distance. Spatial Atlas writes evidence down, computes selected relations,
> and only then lets the model answer.

Stop after that sentence. Let the visitor choose whether to continue.

### A visitor gives you one minute

Use the ninety-second narrative in `poster/POSTER_NARRATIVE.md`, stopping after
Table 1 if time runs out.

### A visitor challenges a source

Use this order:

1. agree with the part that is correct
2. name the claim category
3. state what the source supports
4. state what it does not support
5. point to the visible poster boundary

### A visitor is leaving

Ask for feedback on one technical decision:

> The next experiment holds the fact-sheet schema fixed and corrupts only the
> geometry. Is that the control you would trust, or would you design it
> differently?

For a professional follow-up, ask for the best contact address only after the
technical conversation has ended.

## 10. Red-card claims

Do not say any of the following:

- Public main reproduces either result table.
- The public repository paper omits benchmark results.
- The displayed FieldWork columns are independent replications.
- The graph recomputes every relation distance.
- The scene-graph coordinates have recovered metric scale.
- The local reconstructed geometry is physical ground truth.
- The fifth-percentile estimator is proven to be a mathematical metric.
- The self-grade is calibrated.
- The public controller executes the complete entropy algorithm.
- Public main implements isolated-worker attestation or opt-in dummy fallback.
- The public executor is a sandbox.
- Figure 2 is a deployed end-to-end system.
- The metric bridge is available in public main.
- The evidence-use journal or post-kernel verifier is implemented.
- SpatialClaw's results are Spatial Atlas results.
- An unpublished shuffled-evidence, placebo, or external benchmark arm appears
  on this poster.
- A cost, latency, refusal, or uncertainty number exists when no artifact is
  available.

## 11. Final rehearsal checklist

- Read the fifteen-second, ninety-second, and five-minute versions aloud.
- Rehearse Questions 3.1, 3.7, 4.2, 5.1, 6.1, 7.3, and 7.7.
- Know the Table 1 deltas without looking.
- Remember that the Table 2 weighted arithmetic is a consistency check, not
  proof of the source aggregation protocol.
- Use "model-estimated two-dimensional coordinates," not "measured geometry,"
  for the public scene graph.
- Attribute metric-scale reconstructed geometry only to the poster-described
  local bridge. Do not call Depth-Anything-3 output ground truth.
- Attribute 59.9 percent and 11.2 points to SpatialClaw before stating them.
- Keep the private cluster history and unpublished diagnostics out of public
  answers.
- Do not improvise a number.

## 12. Public source index

- Spatial Atlas v2 preprint:
  <https://arxiv.org/abs/2604.12102v2>
- Spatial Atlas public repository:
  <https://github.com/arunshar/spatial-atlas>
- SpatialClaw paper:
  <https://arxiv.org/abs/2606.13673>
- SpatialClaw official repository:
  <https://github.com/NVlabs/SpatialClaw>
- Visible poster source:
  `poster/spatial_atlas_poster.tex`
- Public implementation:
  `src/`
- Public manuscript:
  `paper/spatial_atlas.tex` and `paper/spatial_atlas.md`
