# Spatial Atlas Poster Narrative

## Thirty-second opening

Spatial Atlas starts from one question: when an agent can compute a fact, why
ask a language model to guess it? We call the alternative compute-grounded
reasoning. The system first turns scene evidence into inspectable state, runs
deterministic spatial or machine-learning operations, and only then asks a
model to generate the answer. The poster shows the current two-handler Atlas
design, source-reported preprint results, and a proposed path toward
SpatialClaw's metric perception and stateful action loop.

## Five-minute walk-through

[Point to the title and Overview.]

The question behind this poster is simple: when an agent can compute something,
why ask a language model to guess it? Spatial Atlas calls the alternative
compute-grounded reasoning, or CGR. The goal is to resolve answerable
subproblems through deterministic computation before asking a language model
to interpret evidence or produce the final response.

[Point to Figure 1.]

Spatial Atlas exposes one Agent-to-Agent server with two benchmark-oriented
handlers. The FieldWorkArena branch handles multimodal spatial questions. The
MLE-Bench branch handles end-to-end machine-learning tasks. Both use shared
model access, budgets, and provenance-oriented controls, then apply
domain-specific processing.

The FieldWork branch converts extracted scene evidence into a typed graph.
Vertices represent entities with labels, estimated positions, attributes, and
zones. Edges represent relations and optional distances. Once the graph exists,
the system can compute pairwise distances, populate relation distances, query
nearby entities, check constraints, and serialize the state into a fact sheet.

[Point to the scene-graph DAG.]

The important distinction is between evidence construction and graph
arithmetic. In public main, a Strong-tier model supplies `position_x` and
`position_y` through a structured JSON extraction prompt. The graph then
performs deterministic calculations over those estimated coordinates. The
arithmetic is repeatable for fixed inputs, but the coordinates are not measured
3D geometry.

The DAG shows the computational path. `compute_distance` supplies a bulk path
through `compute_all_distances`, `check_constraints`, and `to_fact_sheet`. It
also supports the independent, read-only `query_near` branch. The final
language model can receive structured distances and violations instead of
being asked to invent those relations.

[Point to the entropy equations and algorithm.]

The preprint extends CGR with an information-theoretic policy. At step t, the
knowledge state contains observations, computed facts, and intermediate
conclusions. Answer entropy represents uncertainty over candidate answers. The
proposed policy selects the next action expected to reduce that uncertainty
the most.

The complete paper formulation returns a Fast-tier answer above 0.8 confidence,
uses the Standard tier from 0.6 through 0.8, and reflects or escalates below
0.6 for at most two rounds. Public-main FieldWork behavior is narrower. It
produces one Strong-tier answer, requests a confidence estimate from the fast
tier, and performs at most one Strong-tier refinement below 0.6. That path does
not invoke the expected information-gain action selector, and the repository
does not provide calibration evidence for the confidence estimate.

[Point to the Evaluation section.]

The MLE-Bench branch applies a compute-first loop to generated machine-learning
pipelines. It analyzes the task, generates code, attempts execution, and uses
captured error and output context for bounded repair. After a successful run,
it parses a machine-readable validation score. It can request up to two
targeted revisions and retains a revision only when the parsed score improves
under the configured metric direction.

There is one important public-code qualification. The poster describes a
hardened, fail-closed execution path with explicit worker attestation and an
opt-in dummy fallback. Public main does not independently establish that exact
gate. Its current executor launches a subprocess, and its handler creates a
dummy submission after all initial attempts fail. The spoken narrative should
not present the stricter poster wording as behavior verified by public main
unless the corresponding hardened implementation is published.

[Point to Tables 1 and 2.]

These tables are source-reported results from the Spatial Atlas preprint. They
are not a reproduction by public main or by the local integration work.

The preprint reports overall FieldWorkArena accuracy of 72 percent for Factory,
68 percent for Warehouse, and 74 percent for Retail. Without the Spatial Scene
Graph, the corresponding values are 51, 44, and 55 percent. For MLE-Bench, the
preprint reports 82 percent valid submissions and 32 percent medals across 75
competitions.

[Point to Figure 2.]

The third column addresses a limitation of the scene-graph path.
Model-estimated 2D coordinates can propagate localization and depth errors into
downstream distances and constraints.

SpatialClaw provides a complementary action interface. It is a separate,
training-free framework in which a vision-language model writes one
AST-checked Python cell at a time inside a persistent kernel. Variables and
observations persist across steps, so later actions can inspect tool outputs,
compose operations, recompute evidence, and revise the analysis before
returning an answer.

The SpatialClaw paper reports 59.9 percent average accuracy across 20 spatial
reasoning benchmarks and an 11.2-point improvement over the prior best spatial
agent. Those are upstream SpatialClaw results. They are not Spatial Atlas
results, and the two projects use different tasks, models, metrics, and
perception components.

[Point to the Measured Geometry Bridge.]

The poster describes local Spatial Atlas integration work that obtains object
masks and a Depth-Anything-3 point map, erodes masks, retains finite
confidence-qualified points, constructs deterministic horizontal XZ voxel
representatives, and estimates a symmetric surface gap from directed fifth
percentile nearest-neighbor distances. It also describes validation of range,
units, provenance, protocol identity, and evidence hashes.

That metric bridge and its experiment artifacts are not present in public main,
so this repository does not independently reproduce the local bridge. The
complete persistent-kernel link, evidence-use journal, and post-kernel verifier
also remain proposed integration steps. Do not present Figure 2 as a fully
deployed public pipeline.

[Point to the final take-away.]

The contribution is not a claim that every geometric estimate is correct. It
is a design for making spatial evidence explicit, computable, inspectable, and
rejectable. Spatial Atlas structures evidence and performs deterministic
operations over it. SpatialClaw contributes upstream metric-perception
primitives and a stateful action model. The combined direction moves spatial
answers toward evidence that can be recomputed, revised, and traced.

## Two-minute version

Spatial Atlas asks a simple question: when an agent can compute a fact, why ask
a language model to guess it? Its answer is compute-grounded reasoning. The
system builds explicit evidence, performs deterministic operations, and gives
the resulting facts to the model before answer generation.

Figure 1 shows one Agent-to-Agent server with FieldWorkArena and MLE-Bench
handlers. FieldWork creates a typed scene graph. Its five visible operations
compute distances, populate relations, run radius queries, check constraints,
and serialize an inspectable fact sheet. In public main, this arithmetic runs
over coordinates estimated by a model through structured JSON extraction. It
is deterministic arithmetic, not measured 3D geometry.

The entropy equations show the paper's broader theory. The proposed policy
selects actions by expected uncertainty reduction and routes across model
tiers. Public-main FieldWork behavior is narrower: one Strong-tier answer, one
confidence estimate, and at most one Strong-tier refinement below 0.6. The
repository does not demonstrate calibration.

The MLE path generates, executes, repairs, and score-gates pipeline revisions.
The poster describes a stricter execution gate than public main independently
verifies, so that claim needs source-code confirmation before it is stated as
deployed behavior.

Tables 1 and 2 are source-reported preprint results, not a public-repository
reproduction. The overall rows report 72, 68, and 74 percent across the three
FieldWork environments, plus 82 percent valid submissions and 32 percent medals
across 75 MLE-Bench competitions.

The final column motivates integration with SpatialClaw, a separate upstream
framework with metric perception and a persistent code-action loop. Its paper
reports 59.9 percent average accuracy and an 11.2-point improvement. Those are
SpatialClaw results. The local metric bridge described by the poster is not
reproducible from public main, and the persistent-kernel link, evidence journal,
and verifier remain proposed. The take-away is traceability: compute first,
preserve evidence, then generate.

## Audience questions

### Is the scene graph measured geometry?

No. Public main computes deterministic Euclidean relations over model-estimated
2D coordinates. Fixed graph inputs produce repeatable arithmetic, but the input
coordinates can contain perception error.

### Are the poster's benchmark values reproduced here?

No. The FieldWorkArena and MLE-Bench values are source-reported results from
the Spatial Atlas preprint.

### Is the entropy policy deployed exactly as written in the algorithm?

No. The displayed multi-tier policy is the paper formulation. Public-main
FieldWork code uses a Strong-tier answer, a fast-tier confidence estimate, and
at most one Strong-tier refinement below 0.6.

### Is SpatialClaw part of Spatial Atlas?

No. SpatialClaw is a separate upstream framework. The poster presents an
integration design and attributes SpatialClaw mechanisms and results to the
upstream project.

### Is Figure 2 fully deployed in public main?

No. The local metric bridge described by the poster is not present in public
main. The persistent-kernel link, evidence-use journal, and post-kernel
verifier remain proposed integration steps.

## Claim audit

| Category | Claims that may be stated | Evidence boundary |
| --- | --- | --- |
| Implemented in public Spatial Atlas | A2A routing, FieldWork and MLE handlers, typed scene graph, deterministic 2D arithmetic, fact-sheet serialization, current FieldWork confidence gate, public MLE repair and score comparison | `src/`, `tests/`, `ARCHITECTURE.md`, and `README.md` |
| Source-reported Spatial Atlas preprint | FieldWorkArena and MLE-Bench table values, entropy equations, complete multi-tier routing formulation | `paper/spatial_atlas.tex`, `paper/spatial_atlas.md`, and arXiv:2604.12102v2 |
| Upstream SpatialClaw | Persistent code-action loop, perception tools, 59.9 percent average accuracy, 11.2-point improvement | Official NVlabs repository and arXiv:2606.13673 |
| Local integration described by poster | Mask and point-map bridge, erosion, XZ voxel representatives, fifth-percentile surface-gap estimator, strict evidence validation | Visible poster claim and private local work, not independently reproducible from public main |
| Proposed integration | Persistent-kernel link to Atlas evidence state, evidence-use journal, post-kernel verifier | Figure 2 and Sections 7.1 to 7.2 of the poster |
| Public-main contradiction to resolve | Poster says isolated-worker attestation and opt-in dummy fallback, while public main shows subprocess execution and automatic dummy fallback after failed attempts | Compare `poster/spatial_atlas_poster.tex`, `src/mlebench/executor.py`, and `src/mlebench/handler.py` |
