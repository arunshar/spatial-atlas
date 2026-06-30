"""Unit tests for the metric perception backend (hermetic; fakes injected)."""
import pytest
from unittest.mock import AsyncMock

from config import Config
from fieldwork.perception import MetricPerceptionBackend, build_metric_scene
from fakes import FakePerFrameMask, FakeReconstructTool, FakeSAM3

pytestmark = pytest.mark.unit


def _backend(by_phrase):
    b = MetricPerceptionBackend(Config())
    b._sam3 = FakeSAM3(by_phrase)        # set private handles so _ensure_tools no-ops
    b._recon = FakeReconstructTool()
    return b


def test_ground_measures_metric_distance(png_bytes):
    b = _backend({
        "worker": FakePerFrameMask(1, (0.0, 1.7, 0.0)),
        "black forklift": FakePerFrameMask(1, (3.0, 0.5, 4.0)),
    })
    scene = b.ground(png_bytes, ["worker", "black forklift"])
    assert scene.entity_count == 2
    # ground-plane uses (x, z): (0,0) and (3,4) -> 5.0 m. The differing y heights
    # (1.7 vs 0.5) must NOT change the distance (regression guard vs guessed coords).
    assert scene.relations[0].distance == 5.0
    assert scene.entities["e0"].attributes["source"] == "sam3+da3"
    assert scene.entities["e0"].attributes["y_height_m"] == 1.7


def test_empty_mask_is_skipped(png_bytes):
    b = _backend({"ghost": FakePerFrameMask(num_objects=0)})
    assert b.ground(png_bytes, ["ghost"]).entity_count == 0


def test_none_centroid_is_skipped(png_bytes):
    mask = FakePerFrameMask(1)
    mask.get_centroid_3d = lambda recon, frame, object: None  # noqa: A002
    b = _backend({"thing": mask})
    assert b.ground(png_bytes, ["thing"]).entity_count == 0


@pytest.mark.asyncio
async def test_extract_entities_parses_json():
    b = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["worker", "forklift"]}'
    assert await b.extract_entities(llm, "q?") == ["worker", "forklift"]


@pytest.mark.asyncio
async def test_extract_entities_caps_at_six():
    b = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["a","b","c","d","e","f","g","h"]}'
    assert len(await b.extract_entities(llm, "q?")) == 6


@pytest.mark.asyncio
async def test_extract_entities_tolerates_nonjson():
    b = MetricPerceptionBackend(Config())
    llm = AsyncMock()
    llm.generate.return_value = "not json at all"
    assert await b.extract_entities(llm, "q?") == []


@pytest.mark.asyncio
async def test_build_metric_scene_requires_entities(png_bytes):
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": []}'
    with pytest.raises(RuntimeError):
        await build_metric_scene("q?", [png_bytes], llm, Config())


@pytest.mark.asyncio
async def test_build_metric_scene_requires_image():
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": ["worker"]}'
    with pytest.raises(RuntimeError):
        await build_metric_scene("q?", [], llm, Config())


def _inject_spatial_agent(monkeypatch, sam3_available=True, recon_available=True):
    """Inject fake spatial_agent.tools modules so _ensure_tools can import them."""
    import sys
    import types

    sam3_mod = types.ModuleType("spatial_agent.tools.sam3_tool")
    recon_mod = types.ModuleType("spatial_agent.tools.reconstruct_tool")

    class _SAM3Tool:
        is_available = sam3_available

    class _ReconstructTool:
        def __init__(self, sam3, cfg):
            self.is_available = recon_available

    sam3_mod.SAM3Tool = _SAM3Tool
    recon_mod.ReconstructTool = _ReconstructTool
    for name, mod in [
        ("spatial_agent", types.ModuleType("spatial_agent")),
        ("spatial_agent.tools", types.ModuleType("spatial_agent.tools")),
        ("spatial_agent.tools.sam3_tool", sam3_mod),
        ("spatial_agent.tools.reconstruct_tool", recon_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_ensure_tools_constructs_when_available(monkeypatch):
    _inject_spatial_agent(monkeypatch)
    b = MetricPerceptionBackend(Config())
    b._ensure_tools()
    assert b._sam3 is not None and b._recon is not None


def test_ensure_tools_raises_when_server_unavailable(monkeypatch):
    _inject_spatial_agent(monkeypatch, sam3_available=False)
    b = MetricPerceptionBackend(Config())
    with pytest.raises(RuntimeError, match="GPU tool server"):
        b._ensure_tools()
