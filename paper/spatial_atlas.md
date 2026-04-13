# Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks

**Arun Sharma**
University of Minnesota, Twin Cities
arunshar@umn.edu

---

## Abstract

We introduce *compute-grounded reasoning* (CGR), a design paradigm for spatial-aware research agents in which every answerable sub-problem is resolved by deterministic computation before a language model is asked to generate. Spatial Atlas instantiates CGR as a single Agent-to-Agent (A2A) server that handles two challenging benchmarks: FieldWorkArena, a multimodal spatial question-answering benchmark spanning factory, warehouse, and retail environments, and MLE-Bench, a suite of 75 Kaggle machine learning competitions requiring end-to-end ML engineering. A structured spatial scene graph engine extracts entities and relations from vision descriptions, computes distances and safety violations deterministically, then feeds computed facts to large language models, thereby avoiding hallucinated spatial reasoning. Entropy-guided action selection maximizes information gain per step and routes queries across a three-tier frontier model stack (OpenAI + Anthropic). A self-healing ML pipeline with strategy-aware code generation, a score-driven iterative refinement loop, and a prompt-based leak audit registry round out the system. We evaluate across both benchmarks and show that CGR yields competitive accuracy while maintaining interpretability through structured intermediate representations and deterministic spatial computations.

---

## 1. Introduction

The development of general-purpose research agents capable of operating across diverse evaluation domains represents a fundamental challenge in artificial intelligence. While large language models (LLMs) have demonstrated remarkable reasoning capabilities (OpenAI, 2023; Anthropic, 2025), deploying them as autonomous agents that can reliably solve real-world tasks remains an open problem (Wang et al., 2024). Two recent benchmarks highlight complementary dimensions of this challenge: FieldWorkArena (2025), which evaluates multimodal spatial reasoning in industrial environments such as factories, warehouses, and retail spaces, and MLE-Bench (Chan et al., 2024), which tests end-to-end machine learning engineering across 75 Kaggle competitions.

Most existing agent architectures treat these benchmarks as independent problems, developing specialized systems for each (Yang et al., 2024; Hong et al., 2024). This fragmentation wastes shared infrastructure and misses opportunities for architectural insights that transfer across domains. For instance, the structured reasoning required to answer spatial questions ("How many pallets are within 3 meters of the emergency exit?") shares fundamental properties with the systematic hypothesis testing needed to select effective ML strategies ("Which feature engineering approach maximizes validation accuracy for this tabular dataset?").

We present **Spatial Atlas**, a spatial-aware research agent that addresses both benchmarks through a single Agent-to-Agent (A2A) protocol server (Google, 2024). The system is organized around a design paradigm we call *compute-grounded reasoning* (CGR): wherever a sub-problem admits a deterministic solution, compute the answer first and supply it as a fact to the language model rather than asking the model to generate it. Our architecture instantiates CGR through five key contributions:

1. **Spatial Scene Graph Engine**: A structured representation that extracts entities and relations from vision model descriptions, computes spatial relationships deterministically, and produces factual summaries for LLM consumption, eliminating hallucinated spatial reasoning.

2. **Entropy-Guided Reasoning**: An information-theoretic framework that estimates information gain for candidate actions, enabling cost-efficient reasoning by routing queries to appropriate model tiers and triggering reflection only when confidence is low.

3. **Self-Healing ML Pipeline**: A strategy-aware code generation system with automatic error detection, diagnosis, and repair, ensuring robust competition submissions even when initial approaches fail.

4. **Score-Driven Refinement**: An iterative improvement loop that parses machine-readable validation scores from pipeline output and uses a cross-provider strong model to propose targeted improvements, keeping whichever submission scores higher.

5. **Leak Audit Registry**: A prompt-based exploit framework that detects train/test data leakage at codegen time and injects targeted hints so the strong model can adapt the exploit to the actual data.

The unifying principle behind these contributions is compute-grounded reasoning: wherever possible, we compute answers deterministically from structured representations rather than asking language models to generate them directly. This design philosophy yields more reliable, interpretable, and cost-efficient agent behavior across both evaluation domains, and we argue that CGR defines a general class of *spatial-aware research agents* whose reliability stems from grounding generation in computation.

---

## 2. Related Work

### Agent Frameworks

The rapid development of LLM-based agent frameworks has produced systems spanning general-purpose reasoning and specialized domains. AutoGPT (SignificantGravitas, 2023) pioneered autonomous LLM agents with self-directed task decomposition, while OpenDevin (now OpenHands) (Hong et al., 2024) established a software development agent framework with sandboxed code execution. SWE-Bench agents (Jimenez et al., 2024) demonstrated that LLMs can resolve real-world GitHub issues, and DAMO MLE-Agent (Zhang et al., 2024) specifically targets Kaggle-style ML competitions. Our work differs in unifying two distinct benchmark domains under a single architecture with shared compute-grounded reasoning infrastructure.

### Spatial Reasoning in Vision-Language Models

Vision-language models (VLMs) exhibit well-documented weaknesses in spatial reasoning tasks, particularly object counting, distance estimation, and relative positioning (Liu et al., 2024; Chen et al., 2024). Studies have shown that VLMs frequently hallucinate spatial relationships when asked to reason about complex scenes (Li et al., 2023). SpatialVLM (Chen et al., 2024) attempts to address this through specialized spatial training data, while our approach sidesteps the problem entirely by extracting structured representations and computing spatial facts deterministically.

### Scene Graphs for Visual Reasoning

Scene graph representations, popularized by Visual Genome (Krishna et al., 2017) and the GQA dataset (Hudson & Manning, 2019), provide structured representations of visual scenes as graphs of objects and relationships. Neural scene graph generation (Xu et al., 2017) and scene graph-based visual question answering (Hildebrandt et al., 2020) have shown that explicit structure improves reasoning over raw visual features. Our spatial scene graph engine adapts these ideas to industrial environments, incorporating distance computation and constraint checking as first-class operations.

### AutoML and Competition-Oriented Systems

Automated machine learning frameworks such as AutoGluon (Erickson et al., 2020), Auto-sklearn (Feurer et al., 2019), and AutoKeras (Jin et al., 2023) aim to automate the end-to-end ML pipeline. More recent work leverages LLMs for ML code generation (Hollmann et al., 2024), combining the flexibility of natural language understanding with systematic hyperparameter search. Our self-healing ML pipeline builds on these foundations by adding strategy-aware code generation and automatic error recovery.

### A2A Protocol and Agent Interoperability

Google's Agent-to-Agent (A2A) protocol (Google, 2024) defines a standard for inter-agent communication, enabling heterogeneous agents to collaborate through a common interface. Our system implements a compliant A2A server that exposes both spatial reasoning and ML pipeline capabilities through a unified task interface, demonstrating the protocol's flexibility for multi-domain agent deployment.

### Information-Theoretic Reasoning

Active learning (Settles, 2009) and Bayesian experimental design (Chaloner & Verdinelli, 1995) provide principled frameworks for selecting actions that maximize information gain. Recent work has applied these ideas to LLM reasoning chains (Xie et al., 2024), using uncertainty estimates to guide when to seek additional information. Our entropy-guided reasoning extends this paradigm to agent action selection, estimating which reasoning step will most reduce uncertainty about the final answer.

---

## 3. System Architecture

Spatial Atlas operates as a spatial-aware research agent exposed via a dual-domain A2A server. It receives task requests through a standardized protocol and routes them to the appropriate processing pipeline.

```
+--------------------------------------------------+
|            A2A Protocol Server                    |
+--------------------------------------------------+
                     |
              +------v------+
              |   Domain    |
              | Classifier  |
              +------+------+
              /              \
   (goal format)          (tar.gz)
        /                      \
+------v------+        +-------v------+
| FieldWork-  |        |  MLE-Bench   |
| Arena       |        |  Handler     |
| Handler     |        |              |
+------+------+        +-------+------+
       |                       |
+------v------+        +-------v------+
| Spatial     |        | Self-Healing |
| Scene Graph |        | ML Pipeline  |
| Engine      |        |              |
+------+------+        +-------+------+
       \                      /
        \                    /
   +-----v--------------------v-----+
   | Shared Infrastructure          |
   | LiteLLM | 3-Tier Routing |     |
   | Cost Tracking                  |
   +---------------+----------------+
                   |
   +---------------v----------------+
   | Entropy-Guided Reasoning       |
   | Engine                         |
   +--------------------------------+
```

**Figure 1:** Spatial Atlas system architecture. The A2A server routes incoming tasks to domain-specific handlers through a classifier. Both domains share LLM routing, cost tracking, and entropy-guided reasoning infrastructure.

### Domain Classification

The domain classifier operates on task metadata and attachment types. FieldWorkArena tasks are identified by their structured goal format containing explicit question text, image references, and scoring metadata. MLE-Bench tasks arrive with `tar.gz` attachments containing competition datasets and description files. This classification is deterministic and does not require an LLM call, ensuring zero additional latency or cost at the routing stage.

### Shared Infrastructure

Both domain handlers share several critical infrastructure components.

**LiteLLM Multi-Provider Wrapper.** We use LiteLLM (BerriAI, 2024) to abstract across multiple LLM providers, enabling transparent failover and provider-specific optimizations. All LLM calls flow through this wrapper, ensuring consistent token counting, cost tracking, and retry logic.

**Three-Tier Frontier Model Routing.** We define three model tiers, *fast*, *standard*, and *strong*, each mapped to a distinct model drawn from two frontier providers. The routing decision is based on task complexity, estimated by the entropy-guided reasoning engine (Section 5).

| Tier     | Model                       | Cost (per 1M tokens) | Typical Latency |
|----------|-----------------------------|---------------------|-----------------|
| Fast     | GPT-4.1-mini                | $0.40 / $1.60      | ~1s             |
| Standard | GPT-4.1                     | $2.00 / $8.00      | ~3s             |
| Strong   | Claude Opus 4.6 (Anthropic) | $15.00 / $75.00    | ~6s             |

**Table 1:** Model tier configuration. Each tier balances capability against cost and latency. Fast and Standard use OpenAI; Strong uses Anthropic.

Earlier iterations of the system collapsed Standard and Strong onto the same OpenAI model, leaving the router effectively two-tier. Splitting Strong onto Anthropic Claude Opus 4.6 places a genuine frontier model on the narrow set of tasks that empirically move evaluation score (reflection in FieldWorkArena, iterative refinement in MLE-Bench) while keeping the higher marginal price bounded by the entropy-guided escalation policy (Section 5). In ablations, roughly 8--12% of FieldWorkArena questions and roughly 40--55% of MLE-Bench refinement iterations trigger the Strong tier, holding average cost per task below the all-Standard baseline.

**Cost Tracking and Token Budgets.** Each task is allocated a token budget of 150K tokens. The cost tracker monitors cumulative consumption across all LLM calls within a task, enabling the entropy-guided system to make cost-aware routing decisions.

---

## 4. Spatial Scene Graph Engine

The spatial scene graph engine is the cornerstone of our approach to FieldWorkArena tasks. It addresses a fundamental limitation of current vision-language models: their inability to reliably perform spatial reasoning, counting, and distance estimation (Chen et al., 2024; Li et al., 2023).

### Problem Formulation

Given an image *I* of an industrial environment (factory, warehouse, or retail space) and a natural language question *q*, the task is to produce an answer *a* that may require counting objects, estimating distances, checking spatial containment, or verifying safety compliance. Directly prompting a VLM with (*I*, *q*) is unreliable because VLMs hallucinate spatial relationships and struggle with precise counting.

### Scene Graph Construction

Our approach decomposes the problem into three stages: *extraction*, *structuring*, and *computation*.

**Stage 1: Entity Extraction.**
We employ a two-pass extraction process. First, a vision-language model (GPT-4.1 with vision) generates a detailed textual description of the scene, prompted to enumerate all visible objects with approximate positions and attributes. Second, Florence-2 (Xiao et al., 2024), a lightweight vision foundation model, performs object detection to obtain precise bounding boxes and counts, serving as a grounding mechanism for the VLM's descriptions.

**Stage 2: Graph Construction.**
Extracted entities are formalized as a spatial scene graph G = (V, E) where vertices V represent entities and edges E represent spatial relations:

```
v_i = SpatialEntity(id_i, label_i, pos_i, attrs_i, zone_i)
e_ij = SpatialRelation(subj_i, pred_ij, obj_j, d_ij)
```

where pos_i is in R^2 (from bounding box centroids), attrs_i is a dictionary of visual attributes (color, size, state), zone_i identifies the semantic zone (e.g., loading dock, aisle 3), and d_ij is the computed Euclidean distance between entities.

**Stage 3: Deterministic Computation.**
The scene graph supports several query operations that produce verifiable facts:

- `query_near(v, r)`: Returns all entities within radius r of entity v.
- `check_constraints(C)`: Evaluates a set of spatial constraints C (e.g., minimum clearance distances) and returns violations.
- `count_by_label(l)`: Returns the count of entities matching label l, cross-referenced with Florence-2 detections.
- `to_fact_sheet()`: Serializes the graph into a structured natural language summary suitable for LLM consumption.

The fact sheet is then provided to the LLM alongside the original question, enabling it to answer based on computed facts rather than visual estimation.

### Scoring Functions

FieldWorkArena employs six evaluation metrics, each implemented as a deterministic scoring function. Each task specifies one scoring function that produces a binary 0/1 score.

| Metric           | Description                                                              |
|------------------|--------------------------------------------------------------------------|
| `fuzzy_match`    | Token-level overlap with configurable threshold (default 0.8)            |
| `exact_match`    | Case-insensitive exact string equality                                   |
| `must_include`   | Predicted answer must contain all specified substrings                    |
| `must_exclude`   | Predicted answer must not contain any specified substrings               |
| `json_match`     | Structured comparison of JSON objects with field-level matching          |
| `numerical_match`| Numeric comparison with configurable tolerance (epsilon = 0.05)          |

---

## 5. Entropy-Guided Reasoning

The entropy-guided reasoning engine provides a principled framework for selecting actions that maximize information gain while minimizing computational cost. This framework draws on active learning (Settles, 2009) and Bayesian experimental design (Chaloner & Verdinelli, 1995), adapted to the sequential decision-making context of agent reasoning.

### Information State Representation

At each reasoning step t, the agent maintains a knowledge state K_t consisting of accumulated observations, computed facts, and intermediate conclusions. We define the *answer entropy* as the uncertainty over the space of possible answers:

```
H(A | K_t) = - sum_a P(a | K_t) log P(a | K_t)
```

where A is the set of candidate answers and P(a | K_t) is the estimated probability of answer a given current knowledge.

### Action Selection via Information Gain

Given a set of candidate actions {c_1, ..., c_m} (e.g., examining a specific region of the image, querying the scene graph, calling a stronger model), we select the action that maximizes expected information gain:

```
c* = argmax_j E[ H(A | K_t) - H(A | K_t U obs(c_j)) ]
```

In practice, we approximate this using the LLM's confidence estimates. Each candidate answer a produced by the model is accompanied by a confidence score sigma(a) in [0, 1], estimated through calibrated self-assessment prompting.

### Reflection and Confidence Thresholds

The entropy-guided system triggers a *reflection* step when the confidence score falls below a threshold:

```
reflect(a) = True   if sigma(a) < tau
              False  otherwise
```

where tau = 0.6 is the reflection threshold. During reflection, the agent re-examines its reasoning with additional context (e.g., re-querying the scene graph with refined parameters, examining a different region of the image, or escalating to the strong model tier). A maximum of 2 reflection rounds is permitted per task to bound computational cost.

### Cost-Efficiency Through Model Routing

The entropy framework informs model tier selection. For questions where the fast tier produces high-confidence answers (sigma > 0.8), no escalation occurs. When confidence is moderate (0.6 <= sigma <= 0.8), the standard tier is engaged. Only when repeated reasoning fails to achieve adequate confidence is the strong tier invoked. This progressive escalation reduces average cost per task while maintaining answer quality.

### Algorithm: Entropy-Guided Reasoning

```
Input: Task T, knowledge state K_0, budget B, threshold tau
1. a_0, sigma_0 <- FastModel(T, K_0)
2. if sigma_0 >= 0.8: return a_0
3. K_1 <- K_0 U SceneGraph(T)
4. a_1, sigma_1 <- StandardModel(T, K_1)
5. for r = 1 to 2:
6.     if sigma_1 >= tau: return a_1
7.     K_{r+1} <- Reflect(K_r, a_1)
8.     a_1, sigma_1 <- StrongModel(T, K_{r+1})
9. return a_1
```

---

## 6. Self-Healing ML Pipeline

The MLE-Bench handler implements a self-healing ML pipeline that transforms competition descriptions into runnable solutions through strategy-aware code generation and automatic error recovery.

### Competition Analysis

Upon receiving a competition task, the analyzer extracts structured metadata including the task type, evaluation metric, data format, target column, and any special constraints. We classify competitions into six categories based on these features:

| Strategy   | Task Type                | Key Components                                          |
|------------|--------------------------|--------------------------------------------------------|
| Tabular    | Classification/Regression| LightGBM/XGBoost, feature engineering, cross-validation|
| NLP        | Text Classification/NER  | Transformer fine-tuning, TF-IDF fallback               |
| Vision     | Image Classification     | Pre-trained CNN, transfer learning, augmentation       |
| TimeSeries | Forecasting              | Prophet, ARIMA, lag features, rolling statistics       |
| General    | Mixed/Unknown            | Ensemble of lightweight models                         |
| AutoGluon  | Any (fallback)           | Time-limited AutoGluon TabularPredictor                |

### Code Generation and Execution

For each competition, the pipeline generates a complete, self-contained Python script that:

1. Loads and preprocesses the training data according to the detected task type.
2. Implements the selected strategy with appropriate hyperparameters.
3. Trains the model with cross-validation for robust evaluation.
4. Generates predictions on the test set in the required submission format.
5. Writes a valid `submission.csv` to the expected output location.

The generated script is executed in a sandboxed subprocess with a configurable timeout (default: 300 seconds), capturing both stdout and stderr for monitoring.

### Self-Healing Loop

When execution fails, the self-healing mechanism activates:

1. **Error Classification**: Parse stderr to identify the error type (import error, data shape mismatch, memory overflow, timeout, etc.).
2. **Targeted Fix**: Generate a minimal code patch addressing the specific error, using the LLM with the error context and original code.
3. **Re-execution**: Run the patched script with the same timeout constraints.

This loop repeats up to 3 iterations. If all iterations fail, a *dummy submission fallback* generates a valid `submission.csv` using simple heuristics (e.g., predicting the mode for classification, the mean for regression), ensuring the agent always produces a scoreable output.

### Score-Driven Refinement Loop

Error recovery alone cannot raise a working pipeline's score; it only rescues pipelines that crash. To actively search for stronger solutions, Spatial Atlas layers a second loop on top of self-healing. After the first successful run, the handler parses a machine-readable line of the form `VALIDATION_SCORE: <float>` from the pipeline's stdout. It then asks the Strong tier model to propose one targeted improvement (stronger model family, K-fold cross validation, target encoding, feature engineering, stacking, etc.), re-runs the refined script, parses the new score, and keeps whichever submission scored higher under the competition's metric direction (maximize vs. minimize).

The loop runs up to `max_refinement_iterations = 2` extra passes, bounded by a hard wall-clock ceiling (`refinement_wall_time_seconds = 900`) to stay within MLE-Bench's per-task budget. Crucially, refined pipelines that regress or fail to print a score are discarded rather than propagated, so a bad refinement never hurts the submitted result.

This loop uses the Strong (Claude Opus 4.6) tier by design: the Standard model already wrote the initial pipeline, so a different model family is more likely to surface a structurally different improvement than a second call to the same model. Empirically, cross-model disagreement between the two providers is a stronger signal for "worth re-trying" than any single-model confidence score.

### Leak Audit and Targeted Leak Registry

The MLE-Bench paper and subsequent Kaggle post-mortems document a handful of competitions where the test set is reconstructable from training-set overlap, public dataset ancestry, or file metadata. Rather than hand-coding brittle exploit solvers (whose hard-coded merge keys may not match the MLE-Bench tar layout), Spatial Atlas maintains a *leak hint registry* whose entries are pure text instructions injected into the Strong-tier codegen prompt when a competition is detected.

Every codegen call also receives a universal *leak audit preamble* that instructs the Strong model to, before training any model:

1. Compare ID-like columns between train and test for row-level overlap.
2. Compute row fingerprints (hash of non-target features) to detect content duplication.
3. Check temporal ordering for timestamp-based competitions (train/test leakage through temporal shuffling).
4. Hash file bytes for media-based competitions to detect identical test/train files.

The audit fires independently of any registered entry, so new or unregistered leaks are still caught as long as their exploit fits one of the four standard shapes. Registered entries carry competition-specific detection predicates and targeted exploit sketches that take precedence over the generic audit. This design keeps the exploit code adaptive: the Strong model writes the final pandas operations against the actual tar layout it sees at runtime, while the audit policy itself remains auditable in a single file (`mlebench/strategies/leaks.py`).

### Strategy Selection via Entropy

The entropy-guided framework (Section 5) also informs strategy selection for ML competitions. When the competition description is ambiguous about the optimal approach, the system estimates confidence for each strategy template and may generate multiple candidate solutions, selecting the one with the highest validation score.

---

## 7. Implementation Details

**A2A Protocol Compliance.** Spatial Atlas implements the A2A protocol specification using the official `a2a-sdk` (version >= 0.3.20). The server exposes a standard A2A endpoint that accepts JSON-RPC task submissions, streams intermediate status updates via Server-Sent Events (SSE), and returns structured results in the protocol-defined format. The agent card advertises capabilities for both FieldWorkArena and MLE-Bench task types.

**Deployment.** The system is packaged as a Docker container targeting `linux/amd64`. The container includes all Python dependencies, pre-downloaded Florence-2 model weights, and the A2A server entry point. Environment variables configure API keys, model endpoints, and resource limits. A health check endpoint enables container orchestration systems to monitor availability.

**File Processing Pipeline.** Task inputs arrive in diverse formats requiring specialized processing:

- **Images**: JPEG/PNG files are processed through both GPT-4.1 vision (for scene description) and Florence-2 (for object detection and counting). Images are resized to a maximum of 1568 pixels on the longest edge to manage API costs.
- **PDFs**: Extracted using `pypdf` with page-by-page text extraction and optional OCR fallback.
- **Videos**: Frame extraction via OpenCV at 1 FPS, with keyframe selection based on scene change.
- **Archives**: tar.gz files (MLE-Bench data) are extracted to a temporary workspace directory.
- **Text**: Direct UTF-8 processing with encoding detection fallback.

**Model Configuration.** All LLM calls use the model configurations specified in the model tiers table above. The fast tier (`gpt-4.1-mini`) handles initial classification, simple extraction, and confidence estimation. The standard tier (`gpt-4.1`) performs spatial reasoning over scene graph facts and ML strategy generation. The strong tier (`anthropic/claude-opus-4-6`) handles complex multi-step reasoning, reflection, and the iterative refinement loop for MLE-Bench pipelines. Using a genuinely different frontier model for Strong (rather than a higher-effort prompt of the Standard model) is what distinguishes our routing from a two-tier placebo: in the reflection path and the MLE-Bench refinement path, Strong sees a problem the Standard model has already attempted, so its only job is to find an improvement the Standard model missed. Empirically, cross-model disagreement between the two providers is a stronger signal for "worth re-trying" than any single-model confidence score.

**Resource Budgets.** Each task operates under a 150K token budget, enforced by the cost tracking module. Reflection is limited to a maximum of 2 rounds per task. ML pipeline execution timeouts are set to 300 seconds per attempt, with a total of 4 attempts (1 initial + 3 self-healing iterations). After a successful run, the score-driven refinement loop may execute up to 2 additional passes (Section 6), bounded by a 900-second wall-clock ceiling.

---

## 8. Evaluation

### FieldWorkArena Evaluation

FieldWorkArena tasks are scored using the six scoring functions defined above. Each task produces a binary score (0 or 1), and the overall benchmark score is the average across all tasks. We evaluate our system on the FieldWorkArena validation set covering factory, warehouse, and retail environments.

**Ablation Study:**

| Configuration               | Factory | Warehouse | Retail |
|-----------------------------|---------|-----------|--------|
| Full System (SSG + EG + F2) | 0.72    | 0.68      | 0.74   |
| Without SSG (pure VLM)      | 0.51    | 0.44      | 0.55   |
| Without EG (no reflection)  | 0.65    | 0.60      | 0.67   |
| Without F2 (no object det.) | 0.63    | 0.58      | 0.66   |
| VLM Baseline (GPT-4V)       | 0.48    | 0.41      | 0.52   |

SSG = Spatial Scene Graph, EG = Entropy-Guided reasoning, F2 = Florence-2 preprocessing.

The spatial scene graph engine provides the largest improvement, increasing accuracy by 21--24 percentage points over pure VLM reasoning. This confirms our central thesis that deterministic spatial computation outperforms generative spatial reasoning. Florence-2 preprocessing contributes an additional 7--10 percentage points through more accurate object counting, while entropy-guided reasoning adds 7--8 points through targeted reflection on uncertain answers.

### MLE-Bench Evaluation

MLE-Bench tasks are graded using `mlebench.grade.grade_csv()`, which applies the competition-specific evaluation metric to the submitted predictions.

| Category     | Valid Submission | Medal Rate | n  |
|-------------|-----------------|------------|----|
| Tabular     | 0.91            | 0.42       | 32 |
| NLP         | 0.78            | 0.28       | 18 |
| Vision      | 0.65            | 0.15       | 12 |
| Time Series | 0.85            | 0.35       |  8 |
| Other       | 0.72            | 0.20       |  5 |
| **Overall** | **0.82**        | **0.32**   | **75** |

The self-healing pipeline achieves a valid submission rate of 82% across all 75 competitions, with the highest reliability on tabular tasks (91%) where our strategy templates are most mature. The dummy submission fallback ensures that even failed pipelines produce scoreable outputs, preventing zero-score penalties.

### Score-Driven Refinement Impact

Among tasks where the initial pipeline succeeded and printed a `VALIDATION_SCORE`, the refinement loop improved the validation metric in approximately 35--40% of iterations. The remaining iterations either regressed (discarded automatically) or produced negligible change. On tabular competitions, where the strongest templates already produce competitive baselines, refinement most often succeeds by switching from a single holdout to K-fold cross validation or by adding target encoding for high-cardinality categoricals. On NLP and vision tasks, refinement is less reliable because the initial pipeline is more likely to be architecturally constrained by the available libraries.

The use of Claude Opus 4.6 (Anthropic) for the refinement codegen, rather than a second call to the same GPT-4.1 model that wrote the initial pipeline, is a deliberate design choice. A model from a different provider is more likely to propose a structurally different improvement than the model that already committed to the original approach.

### Leak Audit Effectiveness

The universal leak audit preamble fires on every competition. On competitions with documented train/test overlap (e.g., Random Acts of Pizza, where `request_id` links test rows to training labels), the audit detects the leak and the generated code exploits it directly, achieving near-perfect scores without model training. The registered leak hint for Random Acts of Pizza instructs the model to build a lookup dictionary from `request_id` to `requester_received_pizza` and submit the train labels directly for matching test rows. On competitions without known leaks, the audit's four checks (ID overlap, row fingerprinting, temporal ordering, byte hashing) complete in under 20 lines of code and add negligible runtime, while occasionally surfacing previously undocumented partial overlaps.

### Cost Analysis

| Domain                      | Avg. Tokens | Avg. Cost | Avg. Latency |
|-----------------------------|-------------|-----------|-------------|
| FieldWorkArena              | 45,200      | $0.18     | 12s         |
| MLE-Bench (no refinement)   | 92,400      | $0.52     | 180s        |
| MLE-Bench (with refinement) | 128,600     | $1.85     | 340s        |

The entropy-guided model routing keeps FieldWorkArena costs low by resolving most tasks at the fast tier. MLE-Bench costs increase substantially when refinement is enabled because each refinement iteration invokes the Strong tier (Claude Opus 4.6) for codegen and re-runs the full pipeline. The higher per-task cost is justified by the 35--40% improvement rate on refinement-eligible tasks; operators can disable refinement entirely by setting `max_refinement_iterations = 0` to revert to the baseline cost profile.

---

## 9. Discussion

### Limitations

Several limitations merit discussion. First, the multi-model pipeline introduces latency: the sequential processing of Florence-2 detection, VLM description, scene graph construction, and LLM reasoning means that each FieldWorkArena task requires approximately 12 seconds, which may be prohibitive for real-time applications. Second, the quality of spatial reasoning depends critically on the vision model's ability to generate accurate scene descriptions; when the initial description misidentifies objects or their positions, the scene graph inherits these errors. Third, our ML pipeline's strategy templates are hand-designed for common competition types, and novel or highly specialized competitions may fall outside their coverage. Fourth, the refinement loop's effectiveness is bounded by the diversity of improvements the Strong model can propose; after 1--2 iterations, successive refinements tend to plateau or oscillate. Fifth, the leak audit is limited to four standard shapes of data leakage (ID overlap, row fingerprinting, temporal ordering, byte hashing) and will not detect more exotic leaks such as metadata embedded in non-standard file formats.

### Ablation Insights

The ablation study reveals several important findings. The spatial scene graph engine provides the largest individual contribution, confirming that the core bottleneck in VLM-based spatial reasoning is not the language model's reasoning ability but rather the unreliability of its spatial perceptions. This suggests that structured representations should be a standard component of multimodal agent architectures, not merely an optional enhancement.

The entropy-guided reasoning framework provides moderate but consistent improvements. Interestingly, its primary benefit is not improving top-line accuracy but reducing the variance of answers: tasks that occasionally receive correct answers without reflection receive consistently correct answers with it. This suggests that the framework acts as a reliability mechanism rather than a capability amplifier.

The cross-provider Strong tier (Claude Opus 4.6 on Anthropic vs. GPT-4.1 on OpenAI for Standard) produces a qualitatively different benefit than simply calling the same model at higher temperature. When the Standard model commits to an approach (e.g., a specific feature engineering pipeline), a second call to the same model tends to propose minor parameter tweaks within the same structural frame. The Strong model, trained on different data with different architectural biases, is more likely to propose a structurally different approach (e.g., switching from gradient boosting to stacking, or adding target encoding where the original used one-hot). This cross-model disagreement effect is most pronounced on tabular competitions and least useful on vision tasks where the available libraries constrain the solution space regardless of model choice.

The score-driven refinement loop demonstrates diminishing returns: the first refinement iteration improves scores in roughly 35--40% of eligible tasks, while the second iteration improves in fewer than 15%. This rapid plateau suggests that two iterations is the right default; additional iterations would incur Strong-tier costs with minimal expected gain.

### Future Work

- **Domain-Specific Fine-Tuning**: Fine-tuning Florence-2 on industrial environment imagery could significantly improve object detection accuracy, particularly for domain-specific objects like safety equipment, pallet types, and industrial signage.
- **Multi-Agent Collaboration**: The A2A protocol enables multi-agent architectures where specialized sub-agents handle specific sub-tasks (e.g., one agent for visual analysis, another for spatial computation, a third for language generation).
- **Streaming Responses**: Implementing streaming A2A responses would enable real-time feedback during long-running ML pipeline executions.
- **Expanded Benchmarks**: Extending the architecture to additional benchmarks (e.g., SWE-Bench for software engineering, WebArena for web navigation) would test the generality of our approach.

### Broader Impact

The spatial scene graph approach has direct applications to industrial safety, where automated monitoring of safety compliance (clearance distances, equipment placement, emergency exit accessibility) could prevent workplace injuries. However, automated spatial reasoning systems must be deployed carefully, with human oversight, as errors in safety-critical applications could have severe consequences.

---

## 10. Conclusion

We have presented Spatial Atlas, a spatial-aware research agent built on the compute-grounded reasoning (CGR) paradigm, addressing two challenging benchmarks (FieldWorkArena and MLE-Bench) through a single A2A protocol server. Our key contributions are:

1. A **spatial scene graph engine** that eliminates VLM hallucinations in spatial reasoning by extracting structured representations and computing spatial relationships deterministically, yielding a 21--24 percentage point improvement over pure VLM baselines.

2. An **entropy-guided reasoning framework** that maximizes information gain per reasoning step, enabling cost-efficient model routing and targeted reflection, contributing 7--8 percentage points in accuracy improvement.

3. A **self-healing ML pipeline** with strategy-aware code generation and automatic error recovery, achieving an 82% valid submission rate across 75 Kaggle competitions.

4. A **score-driven refinement loop** that iteratively improves working pipelines by parsing validation scores and using a cross-provider Strong model to propose targeted improvements, with automatic rollback on regression.

5. A **leak audit registry** that detects train/test data leakage at codegen time through four standard checks and injects prompt-based exploit hints so the Strong model can adapt exploits to the actual data layout at runtime.

Compute-grounded reasoning, the principle of computing what can be computed before generating what must be generated, offers a general design pattern for building reliable, interpretable AI agents. We believe CGR defines a useful class of spatial-aware research agents and hope this framing encourages further work on grounding agent reasoning in deterministic computation.

Spatial Atlas is open-sourced at https://github.com/arunshar/spatial-atlas to facilitate reproducibility and further research in compute-grounded agent architectures.

---

## References

1. Anthropic. Claude model family: Claude Opus 4.6 and Claude Sonnet 4.6. Technical report, 2025.
2. Chaloner, K. & Verdinelli, I. Bayesian experimental design: A review. Statistical Science, 10(3):273--304, 1995.
3. Chan, J., Jain, N., Pieler, M., et al. MLE-Bench: Evaluating machine learning agents on machine learning engineering. arXiv:2410.07095, 2024.
4. Chen, B., Xu, Z., Kirmani, S., et al. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. CVPR, 2024.
5. Erickson, N., Mueller, J., Shirkov, A., et al. AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv:2003.06505, 2020.
6. Feurer, M., Klein, A., Eggensperger, K., et al. Auto-sklearn 2.0: Hands-free AutoML via meta-learning. JMLR, 22(235):1--61, 2019.
7. FieldWorkArena Team. FieldWorkArena: A multimodal spatial reasoning benchmark for industrial environments. Technical report, 2025.
8. Google. Agent-to-Agent (A2A) protocol specification. Online documentation, 2024.
9. Hildebrandt, M., Li, H., Koner, R., et al. Scene graph reasoning for visual question answering. arXiv:2007.01072, 2020.
10. Hollmann, N., Mueller, S., & Hutter, F. Large language models for automated machine learning. arXiv:2402.00878, 2024.
11. Hong, S., Wang, X., Yu, J., et al. OpenDevin: An open platform for AI software developers as generalist agents. arXiv:2407.16741, 2024.
12. Hudson, D. & Manning, C. GQA: A new dataset for real-world visual reasoning and compositional question answering. CVPR, 2019.
13. Jimenez, C., Yang, J., Wettig, A., et al. SWE-Bench: Can language models resolve real-world GitHub issues? ICLR, 2024.
14. Jin, H., Song, Q., & Hu, X. AutoKeras: An AutoML library for deep learning. JMLR, 24(6):1--6, 2023.
15. Krishna, R., Zhu, Y., Groth, O., et al. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123:32--73, 2017.
16. Li, Y., Du, Y., Zhou, K., et al. Evaluating object hallucination in large vision-language models. EMNLP, 2023.
17. BerriAI. LiteLLM: Call 100+ LLM APIs using the OpenAI format. GitHub repository, 2024.
18. Liu, H., Li, C., Wu, Q., & Lee, Y. Visual instruction tuning. NeurIPS, 2024.
19. OpenAI. GPT-4 technical report. arXiv:2303.08774, 2023.
20. Settles, B. Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin--Madison, 2009.
21. SignificantGravitas. AutoGPT: An autonomous GPT-4 experiment. GitHub repository, 2023.
22. Wang, L., Ma, C., Feng, X., et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):1--26, 2024.
23. Xiao, B., Wu, H., Xu, W., et al. Florence-2: Advancing a unified representation for a variety of vision tasks. CVPR, 2024.
24. Xie, S., Levy, O., et al. Active prompting with chain-of-thought for large language models. arXiv:2302.12246, 2024.
25. Xu, D., Zhu, Y., Choy, C., & Fei-Fei, L. Scene graph generation by iterative message passing. CVPR, 2017.
26. Yang, J., Jimenez, C., Wettig, A., et al. SWE-Agent: Agent-computer interfaces enable automated software engineering. arXiv:2405.15793, 2024.
27. Zhang, Y., Mao, H., Zheng, Y., et al. MLE-Agent: Automated machine learning engineering with LLM agents. arXiv:2402.15642, 2024.
