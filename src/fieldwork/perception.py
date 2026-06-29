"""Metric perception backend: replace LLM-guessed coordinates with measured 3D.

Uses NVIDIA SpatialClaw's GPU tool server (SAM3 segmentation + Depth-Anything-3 / Pi3
reconstruction) to produce REAL metric (x, z) ground-plane positions for scene entities,
then reuses Spatial Atlas's existing deterministic SpatialScene pipeline.

Requires (runs on MSI only): the `spatial_agent` package importable (SpatialClaw, NVIDIA
Source Code License-NC) and a running SpatialClaw GPU tool server (registry
logs/gpu_server.json reachable). Raises RuntimeError on any unavailability so the handler
can fall back to the scene-graph engine.

Pipeline for a FieldWorkArena 2.x distance task:
  reconstruct(image) -> metric point map (meters, gravity-aligned)
  for each entity phrase: SAM3 segment_image_by_text -> mask
                          -> PerFrameMask.get_centroid_3d(recon) -> (x, y, z) meters
  SpatialScene with real ground-plane (x, z) positions -> deterministic metric distance.
"""
import io
import json
import logging
from typing import List

from PIL import Image

from llm import LLMClient
from fieldwork.spatial import SpatialScene, SpatialEntity, SpatialRelation

logger = logging.getLogger("spatial-atlas.fieldwork.perception")

ENTITY_PROMPT = (
    "Extract the physical entities whose spatial position or distance the question asks about.\n"
    'Return JSON {{"entities": ["<short visual phrase>", ...]}} with 1-6 concise, visually '
    'groundable phrases (color + clothing/type), e.g. "woman wearing a yellow vest and black '
    'shirt", "black forklift".\nQuestion: {question}'
)


class MetricPerceptionBackend:
    """Thin client over SpatialClaw's SAM3 + Reconstruct GPU tools."""

    def __init__(self, config):
        self.config = config
        self._sam3 = None
        self._recon = None

    def _ensure_tools(self):
        if self._recon is not None:
            return
        # Imported lazily: only available where the SpatialClaw package is installed
        # (the MSI agent env), so the API service stays importable without it.
        from spatial_agent.tools.sam3_tool import SAM3Tool
        from spatial_agent.tools.reconstruct_tool import ReconstructTool

        class _ReconCfg:
            reconstruct_max_frames = getattr(self.config, "reconstruct_max_frames", 32)

        self._sam3 = SAM3Tool()
        self._recon = ReconstructTool(self._sam3, _ReconCfg())
        if not (self._sam3.is_available and self._recon.is_available):
            raise RuntimeError(
                "SpatialClaw GPU tool server not reachable (logs/gpu_server.json)."
            )

    async def extract_entities(self, llm: LLMClient, question: str) -> List[str]:
        out = await llm.generate(
            ENTITY_PROMPT.format(question=question),
            model_tier="fast", json_mode=True, max_tokens=512,
        )
        try:
            return [e for e in json.loads(out).get("entities", []) if e][:6]
        except Exception:  # noqa: BLE001 - tolerate non-JSON; caller handles empty list
            return []

    def ground(self, image_bytes: bytes, phrases: List[str]) -> SpatialScene:
        self._ensure_tools()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        recon = self._recon.Reconstruct([img])
        fi = recon.frame_indices[0]
        scene = SpatialScene()
        for i, phrase in enumerate(phrases):
            seg = self._sam3.segment_image_by_text(img, phrase)
            if getattr(seg, "num_objects", 0) == 0:
                logger.warning("SAM3 empty mask for %r", phrase)
                continue
            c = seg.get_centroid_3d(recon, frame=fi, object=0)   # (x, y, z) meters, +Y up
            if c is None:
                continue
            scene.add_entity(SpatialEntity(
                id=f"e{i}", label=phrase,
                position=(float(c[0]), float(c[2])),             # ground-plane (x, z)
                attributes={"y_height_m": round(float(c[1]), 2), "source": "sam3+da3"},
            ))
        ids = list(scene.entities)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d = scene.compute_distance(ids[a], ids[b])       # deterministic, metric
                if d is not None:
                    scene.add_relation(
                        SpatialRelation(ids[a], "distance_to", ids[b], round(d, 2))
                    )
        return scene


async def build_metric_scene(query: str, image_bytes_list: List[bytes],
                             llm: LLMClient, config) -> SpatialScene:
    """Entry point used by the handler. Single-image (FieldWorkArena 2.x); extendable."""
    backend = MetricPerceptionBackend(config)
    phrases = await backend.extract_entities(llm, query)
    if not phrases or not image_bytes_list:
        raise RuntimeError("metric backend: no entities extracted or no image present")
    return backend.ground(image_bytes_list[0], phrases)
