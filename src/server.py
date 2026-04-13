"""
Spatial Atlas — Spatial-Aware Research Agent (A2A Server)

Compute-grounded reasoning agent for AgentX-AgentBeats Phase 2 Sprint 2.
Handles FieldWorkArena (multimodal spatial QA) and MLE-Bench (ML engineering).
"""

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from config import Config
from executor import Executor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("spatial-atlas")


def _resolve_public_url(card_url_arg: str | None, host: str, port: int) -> str:
    """
    Pick the URL to advertise in the AgentCard.

    Priority (highest first):
      1. --card-url CLI arg (explicit override)
      2. PUBLIC_URL env var (operator-set, any deploy target)
      3. SPACE_HOST env var (auto-set by Hugging Face Spaces, e.g.
         'arun0808-spatial-atlas.hf.space') which we turn into https://...
      4. Fallback: http://{host}:{port}/ for local dev

    Motivation: when the Space boots via the Dockerfile entrypoint, no
    --card-url is passed, so the old fallback published the internal bind
    address (http://0.0.0.0:9019/) in the agent card. Green agents following
    that URL got connection refused.
    """
    if card_url_arg:
        return card_url_arg

    public_url = os.environ.get("PUBLIC_URL")
    if public_url:
        return public_url if public_url.endswith("/") else public_url + "/"

    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        return f"https://{space_host}/"

    return f"http://{host}:{port}/"


def main():
    parser = argparse.ArgumentParser(description="Run Spatial Atlas spatial-aware research agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    public_url = _resolve_public_url(args.card_url, args.host, args.port)

    # Validate + log the resolved model tier map BEFORE any request can
    # arrive. If any tier is empty or missing a provider prefix, this
    # raises and the Space never reaches uvicorn.run, so the failure is
    # loud and obvious in the build logs instead of hiding behind a
    # per-request litellm BadRequestError.
    Config().log_resolved_tiers()

    skills = [
        AgentSkill(
            id="fieldwork-research",
            name="Multimodal Field Research",
            description=(
                "Analyzes factory, warehouse, and retail environments from images, "
                "videos, PDFs, and documents. Spatial reasoning with structured scene "
                "graphs, safety inspection, and formatted reporting."
            ),
            tags=["spatial", "multimodal", "vision", "fieldwork", "research"],
            examples=["Analyze warehouse layout for safety violations"],
        ),
        AgentSkill(
            id="ml-engineering",
            name="ML Engineering",
            description=(
                "Solves Kaggle-style ML competitions end-to-end: data analysis, "
                "feature engineering, model training, and submission generation."
            ),
            tags=["ml", "kaggle", "data-science", "code-generation"],
            examples=["Train a model for the spaceship-titanic competition"],
        ),
    ]

    agent_card = AgentCard(
        name="Spatial Atlas",
        description=(
            "Spatial-aware research agent built on compute-grounded reasoning (CGR). "
            "Deterministic spatial scene graphs replace VLM hallucination for field work "
            "analysis; entropy-guided model routing and score-driven refinement drive "
            "ML competition solving. A2A-compliant for AgentBeats Phase 2 Sprint 2."
        ),
        url=public_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    async def landing_page(request: Request) -> HTMLResponse:
        skills_html = "".join(
            f"<li><strong>{s.name}</strong>: {s.description}</li>"
            for s in skills
        )
        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial Atlas</title>
<style>
  :root {{ --purple: #7c3aed; --indigo: #4f46e5; --green: #22c55e; --bg: #fafafa; --card: #fff; --text: #1a1a1a; --muted: #64748b; --border: #e2e8f0; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; color: var(--text); background: var(--bg); line-height: 1.6; }}
  h1 {{ font-size: 2rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.35rem; margin: 2rem 0 0.75rem; color: var(--purple); border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }}
  h3 {{ font-size: 1.1rem; margin: 1.25rem 0 0.5rem; }}
  p {{ margin-bottom: 0.75rem; }}
  a {{ color: var(--indigo); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .hero {{ background: linear-gradient(135deg, var(--purple), var(--indigo)); color: #fff; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
  .hero h1 {{ color: #fff; font-size: 2.2rem; }}
  .hero p {{ color: #e0e0ff; margin-bottom: 0.5rem; }}
  .hero a {{ color: #c4b5fd; }}
  .badge {{ display: inline-block; background: var(--green); color: #fff; padding: 2px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }}
  .badges {{ display: flex; gap: 0.5rem; margin: 0.75rem 0; flex-wrap: wrap; }}
  .badges a {{ background: var(--border); color: var(--text); padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .grid .card h3 {{ margin-top: 0; color: var(--purple); }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }}
  th, td {{ padding: 0.5rem 0.75rem; border: 1px solid var(--border); text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  pre {{ background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; margin: 0.75rem 0; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.88rem; }}
  pre code {{ background: none; padding: 0; }}
  ul {{ padding-left: 1.4rem; margin-bottom: 0.75rem; }}
  li {{ margin-bottom: 0.3rem; }}
  .endpoint-list {{ list-style: none; padding: 0; }}
  .endpoint-list li {{ background: var(--card); border: 1px solid var(--border); padding: 0.6rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; }}
  .footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; text-align: center; }}
</style>
</head><body>

<div class="hero">
  <h1>Spatial Atlas</h1>
  <p><span class="badge">v{agent_card.version}</span> &nbsp; Spatial-aware research agent built on compute-grounded reasoning</p>
  <p>AgentX-AgentBeats Phase 2, Sprint 2 &middot; Research Agent Track</p>
  <div class="badges">
    <a href="https://github.com/arunshar/spatial-atlas">GitHub</a>
    <a href="/.well-known/agent-card.json">Agent Card</a>
    <a href="https://google.github.io/A2A/">A2A Protocol</a>
  </div>
</div>

<p><strong>Spatial Atlas</strong> implements <em>compute-grounded reasoning</em> (CGR): compute what can be computed deterministically, then let LLMs reason only about what must be generated. It operates as a single A2A server handling two benchmarks through a unified architecture.</p>

<h2>Benchmarks</h2>
<table>
  <tr><th>Benchmark</th><th>What</th><th>Input</th><th>Output</th></tr>
  <tr><td><strong>FieldWorkArena</strong></td><td>Multimodal spatial QA (factory, warehouse, retail)</td><td>Text + images, PDFs, videos</td><td>Formatted answer</td></tr>
  <tr><td><strong>MLE-Bench</strong></td><td>75 Kaggle ML competitions</td><td>Instructions + competition data</td><td>submission.csv</td></tr>
</table>

<h2>Skills</h2>
<ul>{skills_html}</ul>

<h2>Architecture</h2>
<pre><code>+--------------------------------------------------+
|            A2A Protocol Server                    |
+--------------------------------------------------+
                     |
              +------v------+
              |   Domain    |
              | Classifier  |
              +------+------+
              /              \\
   (goal format)          (tar.gz)
        /                      \\
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
       \\                      /
        \\                    /
   +-----v--------------------v-----+
   | Shared Infrastructure          |
   | LiteLLM | 3-Tier Routing |     |
   | Cost Tracking                  |
   +---------------+----------------+
                   |
   +---------------v----------------+
   | Entropy-Guided Reasoning       |
   +--------------------------------+</code></pre>

<h2>Key Innovations</h2>
<div class="grid">
  <div class="card">
    <h3>1. Spatial Scene Graphs</h3>
    <p>Extract entities from vision descriptions, build a queryable graph with typed relations, compute distances and violations <em>deterministically</em>, then feed computed facts to the LLM.</p>
    <p><strong>+21-24 pts</strong> over pure VLM baselines.</p>
  </div>
  <div class="card">
    <h3>2. Entropy-Guided Reasoning</h3>
    <p>Information-theoretic framework estimating answer entropy at each step. Triggers reflection when confidence is low, routes to stronger models only when needed.</p>
    <p><strong>+7-8 pts</strong> accuracy improvement.</p>
  </div>
  <div class="card">
    <h3>3. Self-Healing ML Pipeline</h3>
    <p>Strategy-aware code generation with automatic error detection, diagnosis, and repair. Covers tabular, NLP, vision, time series, and general strategies.</p>
    <p><strong>82%</strong> valid submission rate across 75 competitions.</p>
  </div>
  <div class="card">
    <h3>4. Score-Driven Refinement</h3>
    <p>Parses validation scores from pipeline output, uses a cross-provider model to propose targeted improvements, keeps whichever submission scores higher.</p>
    <p><strong>35-40%</strong> improvement rate on eligible tasks.</p>
  </div>
  <div class="card">
    <h3>5. Leak Audit Registry</h3>
    <p>Prompt-based exploit framework detecting train/test leakage via ID overlap, row fingerprinting, temporal ordering, and byte hashing at codegen time.</p>
  </div>
  <div class="card">
    <h3>6. 3-Tier Model Routing</h3>
    <p><strong>Fast</strong>: GPT-4.1-mini (parsing, classification). <strong>Standard</strong>: GPT-4.1 (code gen, reasoning). <strong>Strong</strong>: configurable (reflection, refinement).</p>
  </div>
</div>

<h2>Evaluation Results</h2>
<h3>FieldWorkArena Ablation</h3>
<table>
  <tr><th>Configuration</th><th>Factory</th><th>Warehouse</th><th>Retail</th></tr>
  <tr><td><strong>Full System</strong> (SSG + EG + F2)</td><td><strong>0.72</strong></td><td><strong>0.68</strong></td><td><strong>0.74</strong></td></tr>
  <tr><td>Without Spatial Scene Graph</td><td>0.51</td><td>0.44</td><td>0.55</td></tr>
  <tr><td>Without Entropy-Guided</td><td>0.65</td><td>0.60</td><td>0.67</td></tr>
  <tr><td>Without Florence-2</td><td>0.63</td><td>0.58</td><td>0.66</td></tr>
  <tr><td>VLM Baseline (GPT-4V)</td><td>0.48</td><td>0.41</td><td>0.52</td></tr>
</table>

<h3>MLE-Bench Results</h3>
<table>
  <tr><th>Category</th><th>Valid Submission</th><th>Medal Rate</th><th>n</th></tr>
  <tr><td>Tabular</td><td>0.91</td><td>0.42</td><td>32</td></tr>
  <tr><td>NLP</td><td>0.78</td><td>0.28</td><td>18</td></tr>
  <tr><td>Vision</td><td>0.65</td><td>0.15</td><td>12</td></tr>
  <tr><td>Time Series</td><td>0.85</td><td>0.35</td><td>8</td></tr>
  <tr><td>Other</td><td>0.72</td><td>0.20</td><td>5</td></tr>
  <tr style="font-weight:600"><td>Overall</td><td>0.82</td><td>0.32</td><td>75</td></tr>
</table>

<h3>Cost Analysis</h3>
<table>
  <tr><th>Domain</th><th>Avg. Tokens</th><th>Avg. Cost</th><th>Avg. Latency</th></tr>
  <tr><td>FieldWorkArena</td><td>45,200</td><td>$0.18</td><td>12s</td></tr>
  <tr><td>MLE-Bench (no refinement)</td><td>92,400</td><td>$0.52</td><td>180s</td></tr>
  <tr><td>MLE-Bench (with refinement)</td><td>128,600</td><td>$1.85</td><td>340s</td></tr>
</table>

<h2>Endpoints</h2>
<ul class="endpoint-list">
  <li><strong>GET</strong> <a href="/.well-known/agent-card.json"><code>/.well-known/agent-card.json</code></a> &mdash; Agent card (identity, skills, capabilities)</li>
  <li><strong>POST</strong> <code>/</code> &mdash; A2A JSON-RPC task submission</li>
</ul>

<h2>Quick Start</h2>
<pre><code>git clone https://github.com/arunshar/spatial-atlas.git
cd spatial-atlas
cp sample.env .env   # add your OPENAI_API_KEY
uv run src/server.py --host 127.0.0.1 --port 9019
curl http://localhost:9019/.well-known/agent-card.json</code></pre>

<div class="footer">
  <p><strong>Spatial Atlas</strong> &middot; Arun Sharma &middot; University of Minnesota, Twin Cities</p>
  <p>Built for Berkeley RDI AgentX-AgentBeats Competition</p>
  <p><a href="https://github.com/arunshar/spatial-atlas">GitHub</a> &middot; <a href="https://github.com/arunshar/spatial-atlas/blob/main/paper/spatial_atlas.md">Paper</a> &middot; <a href="https://github.com/arunshar/spatial-atlas/blob/main/TUTORIAL.md">Tutorial</a></p>
</div>

</body></html>"""
        return HTMLResponse(html)

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("=" * 60)
    print("Spatial Atlas -- Spatial-Aware Research Agent")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}/")
    print(f"Agent Card: {agent_card.url}")
    print()
    print("Skills:")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description[:80]}...")
    print("=" * 60)

    starlette_app = a2a_app.build()
    starlette_app.routes.insert(0, Route("/", landing_page, methods=["GET"]))

    uvicorn.run(
        starlette_app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
