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
  body {{ font-family: system-ui, sans-serif; max-width: 680px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .badge {{ display: inline-block; background: #22c55e; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }}
  a {{ color: #2563eb; }}
  ul {{ padding-left: 1.2rem; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.9rem; }}
</style>
</head><body>
<h1>Spatial Atlas</h1>
<p><span class="badge">v{agent_card.version}</span> Spatial-aware research agent (A2A)</p>
<h2>Skills</h2>
<ul>{skills_html}</ul>
<h2>Endpoints</h2>
<ul>
  <li><a href="/.well-known/agent-card.json">Agent Card</a> (GET)</li>
  <li><code>POST /</code> for A2A JSON-RPC requests</li>
</ul>
<p><a href="https://github.com/arunshar/spatial-atlas">GitHub</a></p>
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
