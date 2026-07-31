# Spatial Atlas Poster Package

This directory contains the print-ready Spatial Atlas poster, its reproducible
LaTeX source, the exact visual assets used by the source, a screen preview, the
print preflight record, and evidence-bounded narrative materials.

## File map

| Path | Purpose |
| --- | --- |
| `poster/spatial_atlas_poster.tex` | Authoritative poster source |
| `poster/spatial_atlas_poster.pdf` | Authoritative 48 by 36 inch print PDF |
| `poster/spatial_atlas_poster_preview.png` | Screen preview rendered from the authoritative PDF |
| `poster/PRINT_PREFLIGHT.md` | Measured PDF, font, image, margin, link, and QR checks |
| `poster/POSTER_NARRATIVE.md` | Current evidence-bounded spoken narrative |
| `poster/POSTER_QA_PACKET.md` | Audited audience questions, answers, arithmetic checks, and red-card claims |
| `poster/SPATIAL_ATLAS_EXPLAINED.md` | Detailed public explainer with implementation, theory, evaluation, attribution, and limitation boundaries |
| `poster/SPATIAL_ATLAS_EXPLAINED.html` | Self-contained browser version of the detailed explainer |
| `poster/CLAUDE_POSTER_NARRATIVE_HANDOFF.md` | Claim boundaries and source hierarchy for narrative revision |
| `poster/CLAUDE_POSTER_REVISION_PROMPT.md` | Paste-ready task prompt for Claude |
| `poster/assets/umn_seal.pdf` | Vector University of Minnesota seal used in the header |
| `poster/assets/nvidia_logo.pdf` | Vector NVIDIA logo used in the header |
| `poster/assets/berkeley_rdi.png` | Official Berkeley RDI horizontal wordmark used in the header |

The PDF is the print authority. The PNG is only a preview.
`SPATIAL_ATLAS_EXPLAINED.md` is the text authority for the detailed guide. The
HTML file is a self-contained browser export of that guide.

## Build

Build from the repository root with XeLaTeX:

```bash
cd poster
xelatex -interaction=nonstopmode -halt-on-error -file-line-error spatial_atlas_poster.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error spatial_atlas_poster.tex
```

The source uses relative asset paths, so it must be compiled from `poster/`.
The expected output is a one-page, 48 by 36 inch landscape PDF.

## Evidence hierarchy

Use this order when auditing or revising poster claims:

1. `poster/spatial_atlas_poster.tex` controls the visible poster.
2. `src/` and `tests/` control claims about behavior implemented in public main.
3. `paper/spatial_atlas.tex` and `paper/spatial_atlas.md` control theory and
   values explicitly labeled as source-reported Spatial Atlas preprint results.
4. The official [SpatialClaw repository](https://github.com/NVlabs/SpatialClaw)
   and [SpatialClaw paper](https://arxiv.org/abs/2606.13673) control upstream
   SpatialClaw claims.
5. Private cluster runs, local worktrees, and recalled chat context are not
   public evidence.

The public manuscript contains the FieldWorkArena and MLE-Bench tables. Their
values are source-reported preprint results, and this repository snapshot does
not contain the run artifacts needed to reproduce them.
The public main branch also does not independently contain the local
SpatialClaw metric-bridge implementation or its experiment artifacts. Figure 2
must therefore retain its implemented versus proposed boundary, and any local
bridge claim must be attributed to local Spatial Atlas integration work rather
than public-main reproducibility.

## Source-code entry points

The main narrative-relevant files are:

- `src/agent.py`
- `src/fieldwork/handler.py`
- `src/fieldwork/reasoner.py`
- `src/fieldwork/spatial.py`
- `src/fieldwork/vision.py`
- `src/mlebench/handler.py`
- `src/mlebench/codegen.py`
- `src/mlebench/executor.py`
- `src/entropy/engine.py`
- `src/cost/router.py`
- `src/cost/tracker.py`
- `src/llm.py`
- `src/executor.py`
- `src/config.py`
- `tests/test_agent.py`
- `scenarios/fieldwork/scenario.toml`
- `scenarios/mlebench/scenario.toml`
- `paper/spatial_atlas.tex`
- `paper/spatial_atlas.md`
- `ARCHITECTURE.md`
- `README.md`
- `pyproject.toml`

## Asset notes

The Berkeley RDI wordmark is byte-identical to Berkeley RDI's official
720 by 148 pixel website asset:

<https://raw.githubusercontent.com/rdi-berkeley/rdi-berkeley.github.io/main/assets/images/rdi_logo_horizontal_720.png>

No exact higher-resolution or vector version of that newer blue wordmark was
found in the official repository, its history, or its press kit. The current
placement renders at 93 ppi. It is acceptable at normal poster viewing
distance, but it is the weakest raster asset under close inspection. The other
two header marks are embedded as vector PDFs.

See `poster/PRINT_PREFLIGHT.md` for the final measured checks.
