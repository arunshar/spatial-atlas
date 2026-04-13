# Spatial Atlas — Ultra-Fine Step-by-Step Running Guide

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | >= 3.12 | `brew install python@3.12` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | latest | [docker.com/get-started](https://docker.com/get-started) |
| git | any | pre-installed on macOS |
| gh (GitHub CLI) | any | `brew install gh` |

## 1. Clone the Repository

```bash
git clone https://github.com/arunshar/spatial-atlas.git
cd spatial-atlas
```

## 2. Set Up Environment Variables

```bash
# Copy the sample env file
cp sample.env .env

# Edit with your API key
nano .env
```

Your `.env` file must contain:
```
OPENAI_API_KEY=sk-your-openai-key-here
```

Optional overrides:
```
# Add other provider keys here if overriding model tiers
ATLAS_FAST_MODEL=openai/gpt-4.1-mini   # Fast tier (classification, parsing)
ATLAS_STANDARD_MODEL=openai/gpt-4.1    # Standard tier (code gen, analysis)
ATLAS_STRONG_MODEL=openai/gpt-4.1      # Strong tier (spatial reasoning)
ATLAS_VISION_MODEL=openai/gpt-4.1      # Vision tier (image analysis)
```

## 3. Install Dependencies

```bash
# This resolves all 93 packages and creates uv.lock
uv sync
```

Expected output:
```
Resolved 93 packages in 3ms
Installed 93 packages in 2s
```

To also install test dependencies:
```bash
uv sync --extra test
```

## 4. Run the Agent Server Locally

```bash
uv run src/server.py --host 127.0.0.1 --port 9019
```

You should see:
```
============================================================
Spatial Atlas — Purple Agent
============================================================
Server: http://127.0.0.1:9019/
Agent Card: http://127.0.0.1:9019/

Skills:
  - Multimodal Field Research: Analyzes factory, warehouse, and retail...
  - ML Engineering: Solves Kaggle-style ML competitions end-to-end...
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:9019
```

## 5. Verify the Agent Card

In a separate terminal:
```bash
curl -s http://localhost:9019/.well-known/agent-card.json | python3 -m json.tool
```

Expected response (formatted):
```json
{
    "name": "Spatial Atlas",
    "version": "1.0.0",
    "protocolVersion": "0.3.0",
    "capabilities": {"streaming": true},
    "skills": [
        {
            "id": "fieldwork-research",
            "name": "Multimodal Field Research",
            "tags": ["spatial", "multimodal", "vision", "fieldwork", "research"]
        },
        {
            "id": "ml-engineering",
            "name": "ML Engineering",
            "tags": ["ml", "kaggle", "data-science", "code-generation"]
        }
    ]
}
```

## 6. Run Tests

```bash
# Run all tests
uv run pytest -v

# Run specific test class
uv run pytest tests/test_agent.py::TestSpatialScene -v

# Run with output
uv run pytest -v -s
```

## 7. Build the Docker Image

```bash
# Build for linux/amd64 (required for AgentBeats platform)
docker build -t spatial-atlas --platform linux/amd64 .

# Run the container
docker run -p 9019:9019 --env-file .env spatial-atlas --host 0.0.0.0

# Verify
curl http://localhost:9019/.well-known/agent-card.json
```

## 8. Push to GitHub Container Registry (GHCR)

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u arunshar --password-stdin

# Tag and push
docker tag spatial-atlas ghcr.io/arunshar/spatial-atlas:latest
docker push ghcr.io/arunshar/spatial-atlas:latest
```

Or let GitHub Actions do it automatically — just push a tag:
```bash
git tag v1.0.0
git push origin v1.0.0
```

## 9. Deploy to HuggingFace Spaces

```bash
# Login to HuggingFace (one-time setup)
uv run python -c "from huggingface_hub import login; login(token='YOUR_HF_WRITE_TOKEN')"

# Deploy
./deploy_to_hf.sh
```

Then add your `OPENAI_API_KEY` as a Space Secret:
1. Go to https://huggingface.co/spaces/Arun0808/spatial-atlas/settings
2. Scroll to "Repository secrets"
3. Add: Name = `OPENAI_API_KEY`, Value = your key

## 10. Submit to AgentBeats Competition

Submit this URL as your purple agent endpoint:
```
https://Arun0808-spatial-atlas.hf.space/
```

Or use the GHCR Docker image:
```
ghcr.io/arunshar/spatial-atlas:latest
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: litellm` | Run `uv sync` first |
| `OPENAI_API_KEY not set` | Create `.env` file with your key |
| Docker build fails | Ensure `uv.lock` exists: `uv lock` |
| "Method Not Allowed" in browser | Expected — the root `/` doesn't serve HTML. Check `/.well-known/agent-card.json` instead |
| Port 9019 in use | Change port: `uv run src/server.py --port 9020` |
| Tests fail with import errors | Run from project root with `uv run pytest` |
