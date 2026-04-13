# Spatial Atlas

**Spatial-aware research agent for AgentX-AgentBeats Phase 2, Sprint 2 — Research Agent Track**

> Builds on the 1st Place Sprint 1 Business Process Track approach

## What It Does

Spatial Atlas is a spatial-aware research agent built on **compute-grounded reasoning (CGR)**: compute what can be computed deterministically, then let LLMs reason only about what must be generated. It handles **two benchmarks** through a single A2A server:

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
     │ Answer Formatter│                        │ Self-Healing      │
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

This yields correct distance measurements, accurate violation counts, and consistent JSON for `json_match` evaluation.

### 2. Entropy-Guided Reasoning
Continued from the winning Sprint 1 approach:
- Estimate confidence of initial answers
- Trigger self-reflection when confidence is low
- Prioritize high-information-gain reasoning paths

### 3. Self-Healing ML Pipelines (MLE-Bench)
- Generate complete Python scripts from competition descriptions
- Execute in sandboxed subprocess with timeout
- On failure: read error, fix code, retry (up to 3 iterations)
- Strategy library: tabular, NLP, vision, time series, general

### 4. 3-Tier Model Routing (frontier-routed)
- **Fast** (`openai/gpt-4.1-mini`): classification, parsing, formatting
- **Standard** (`openai/gpt-4.1`): code generation, mid-complexity analysis
- **Strong** (`openai/gpt-4.1`): spatial reasoning, reflection, hard MLE iterations

The strong tier handles the tasks that move evaluation score the most (reflection and code refinement). Override via `ATLAS_STRONG_MODEL` to use a different provider.

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

To swap a tier to another provider, set the env var with the provider prefix (e.g. `ATLAS_STRONG_MODEL=gemini/gemini-3-pro-preview` with `GEMINI_API_KEY`).

## Project Structure

```
src/
├── server.py              # A2A entry point
├── executor.py            # AgentExecutor lifecycle
├── agent.py               # Core orchestrator (THE BRAIN)
├── config.py              # Centralized configuration
├── llm.py                 # LiteLLM wrapper with cost tracking
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
│   ├── executor.py        # Safe code execution
│   └── strategies/        # ML strategy templates
├── entropy/               # Entropy-guided reasoning
│   └── engine.py          # Information gain estimation
└── cost/                  # Cost optimization
    ├── router.py          # 3-tier model selection
    └── tracker.py         # Token/cost budget tracking
```

## License

MIT

---

Built for Berkeley RDI AgentX-AgentBeats Competition. Green agent infrastructure provided by the Entropic team.
