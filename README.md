# Spatial Atlas

**Spatial-aware research agent for AgentX-AgentBeats Phase 2, Sprint 2: Research Agent Track**

## What It Does

Spatial Atlas is a spatial-aware research agent built on **compute-grounded reasoning (CGR)**: compute what can be computed from explicit inputs, then let LLMs reason about what must be generated. It exposes two benchmark-oriented handler surfaces through a single A2A server. FieldWorkArena remains unevaluated because its data were inaccessible. MLE-Bench generated-code execution is disabled by default.

| Benchmark | What | Input | Output |
|-----------|------|-------|--------|
| **FieldWorkArena** | Multimodal spatial QA (factory/warehouse/retail) | Text goal + images, PDFs, videos | Formatted answer text |
| **MLE-Bench** | 75 Kaggle ML competitions | Instructions + competition.tar.gz | submission.csv |

## Architecture

```
                     ┌──────────────────────────────────┐
                     │        A2A Protocol Layer         │
                     │  server.py → executor.py → agent  │
                     └──────────────┬───────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                                           ▼
     ┌────────────────┐                        ┌──────────────────┐
     │  FieldWork      │                        │  MLE-Bench        │
     │  Handler        │                        │  Handler          │
     ├────────────────┤                        ├──────────────────┤
     │ Vision Pipeline │                        │ Competition       │
     │ Spatial Scene   │ ◄── Crown Jewel        │   Analyzer        │
     │   Graph Engine  │                        │ ML Code Generator │
     │ Entropy Reasoner│                        │ Strategy Library  │
     │ Answer Formatter│                        │ Fail-Closed       │
     └────────────────┘                        │   Code Executor   │
              │                                 └──────────────────┘
              └──────────┬────────────────────────────┘
                         ▼
              ┌──────────────────────┐
              │  Shared Infrastructure│
              │  LLM (litellm)       │
              │  Cost Router (3-tier)│
              │  Entropy Engine      │
              └──────────────────────┘
```

## Key Innovations

### 1. Structured Spatial Scene Graphs (FieldWorkArena)
Instead of asking LLMs to hallucinate spatial relationships:
- **Extract** entities + positions from vision descriptions
- **Store** in a queryable graph with typed relations
- **Compute** distances, containment, violations *deterministically*
- **Feed** computed facts back to LLM for natural language answers

This makes coordinate assumptions and arithmetic inspectable. Incorrect detection, depth, scale, or correspondence can still produce incorrect results.

### 2. Entropy-Guided Reasoning
Retained from the Sprint 1 design:
- Estimate confidence of initial answers
- Trigger self-reflection when confidence is low
- Prioritize high-information-gain reasoning paths

### 3. Fail-Closed ML Pipelines (MLE-Bench)
- Generate complete Python scripts from competition descriptions
- Fail closed unless an operator sets both execution and isolated-worker attestation flags
- Execute in a bounded subprocess with a 600-second timeout only after that explicit opt-in
- On failure: read the error, fix the code, and make at most 3 total attempts
- Fail the task after those attempts unless dummy submissions are separately enabled
- Guard each public A2A execution with a concurrency-safe heuristic reservation before provider calls
- Strategy library: tabular, NLP, vision, time series, general

### Public Agent Token Boundary

The public A2A `Agent` uses `BudgetedLLMClient` with the legacy configuration field `Config.max_tokens_per_task = 150_000`. Before a provider call, it heuristically estimates prompt usage and reserves that estimate plus the allowed maximum completion under a lock. Concurrent calls within one A2A execution cannot oversubscribe that estimated reservation counter. A new execution receives a new counter, even when it references the same A2A task ID. This is not exact tokenizer accounting and not a hard provider-token boundary, because provider tokenizers and image accounting vary. Provider-reported usage remains authoritative. Frozen benchmark drivers continue to use the base observational `LLMClient`; benchmark batch usage is reconciled from immutable artifacts.

### 4. 3-Tier Model Routing (frontier-routed)
- **Fast** (`openai/gpt-4.1-mini`): classification, parsing, formatting
- **Standard** (`openai/gpt-4.1`): code generation, mid-complexity analysis
- **Strong** (`openai/gpt-4.1`): spatial reasoning, reflection, hard MLE iterations

The strong tier is intended for reflection and code refinement. Its effect on evaluation score and cost remains an evaluation question. Override it via `ATLAS_STRONG_MODEL` to use a different provider.

## Quick Start

```bash
# Clone
git clone https://github.com/arunshar/spatial-atlas.git
cd spatial-atlas

# Setup
cp sample.env .env
# Edit .env with your OPENAI_API_KEY

# Run locally
uv run src/server.py --host 127.0.0.1 --port 9019

# Verify
curl http://localhost:9019/.well-known/agent-card.json
```

## Docker

```bash
# Build
docker build -t spatial-atlas --platform linux/amd64 .

# Run
docker run -p 9019:9019 --env-file .env spatial-atlas --host 0.0.0.0
```

## Testing

```bash
uv run pytest -v
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | (none) | OpenAI API key (all tiers default to OpenAI) |
| `ATLAS_FAST_MODEL` | No | `openai/gpt-4.1-mini` | Fast tier model |
| `ATLAS_STANDARD_MODEL` | No | `openai/gpt-4.1` | Standard tier model |
| `ATLAS_STRONG_MODEL` | No | `openai/gpt-4.1` | Strong tier model |
| `ATLAS_VISION_MODEL` | No | `openai/gpt-4.1` | Vision tier model |
| `ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION` | No | disabled | Explicitly allow generated-code execution in an isolated, trusted worker |
| `ATLAS_TRUSTED_ISOLATED_WORKER` | No | disabled | Attest that the process is running in an isolated, trusted worker |
| `ATLAS_ALLOW_DUMMY_SUBMISSION` | No | disabled | Explicitly allow a dummy submission after all real pipeline attempts fail |
| `ATLAS_BEARER_TOKEN` | Conditional | (none) | Protect non-read-only requests; required and at least 32 characters when MLE execution is enabled |
| `ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC` | No | disabled | Test-only override permitting a non-loopback bind without bearer authentication |
| `ATLAS_MAX_REQUEST_BYTES` | No | 64 MiB | Maximum HTTP request body size |
| `ATLAS_MAX_CONCURRENT_REQUESTS` | No | 4 | Maximum active HTTP requests; excess requests receive HTTP 503 |

To swap a tier to another provider, set the env var with the provider prefix (e.g. `ATLAS_STRONG_MODEL=gemini/gemini-3-pro-preview` with `GEMINI_API_KEY`).

Enabling MLE execution requires both `ATLAS_ENABLE_MLEBENCH_CODE_EXECUTION=true` and `ATLAS_TRUSTED_ISOLATED_WORKER=true`. Server startup then also requires `ATLAS_BEARER_TOKEN` with at least 32 characters. A normal non-loopback bind always requires that bearer token, even when code execution is disabled. Only loopback development may omit it. `ATLAS_ALLOW_UNAUTHENTICATED_PUBLIC=true` bypasses this startup gate for test-only use and must not be used for normal deployment. When a token is configured, non-read-only requests must authenticate. The server rejects request bodies above 64 MiB and rejects active requests above the default concurrency limit of 4.

## Project Structure

```
src/
├── server.py              # A2A entry point
├── executor.py            # AgentExecutor lifecycle
├── agent.py               # Core orchestrator (THE BRAIN)
├── config.py              # Centralized configuration
├── llm.py                 # LiteLLM wrapper with cost tracking
├── budgeted_llm.py        # Heuristic per-execution reservation for the public A2A Agent
├── fieldwork/             # FieldWorkArena domain
│   ├── handler.py         # Pipeline orchestrator
│   ├── parser.py          # Goal string parser
│   ├── vision.py          # Multimodal file processing
│   ├── spatial.py         # Spatial scene graph engine
│   ├── reasoner.py        # Entropy-guided reasoning
│   └── formatter.py       # Output format matching
├── mlebench/              # MLE-Bench domain
│   ├── handler.py         # Pipeline orchestrator
│   ├── analyzer.py        # Competition analysis
│   ├── codegen.py         # ML code generator
│   ├── executor.py        # Bounded subprocess execution for a trusted isolated worker
│   └── strategies/        # ML strategy templates
├── entropy/               # Entropy-guided reasoning
│   └── engine.py          # Information gain estimation
└── cost/                  # Cost optimization
    ├── router.py          # 3-tier model selection
    └── tracker.py         # Provider-reported token/cost observation
```

## License

MIT

---

Built for Berkeley RDI AgentX-AgentBeats Competition. Green agent infrastructure provided by the Entropic team.
