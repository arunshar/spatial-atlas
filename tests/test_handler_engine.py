"""Integration tests for the FieldWork handler's engine selection + fallback.

All sub-components are mocked; we assert which scene path runs under each
ATLAS_FIELDWORK_ENGINE setting and that the metric path degrades gracefully."""
import pytest
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.integration


def _handler(engine):
    from config import Config
    from fieldwork.handler import FieldWorkHandler

    cfg = Config()
    cfg.fieldwork_engine = engine

    h = FieldWorkHandler.__new__(FieldWorkHandler)
    h.config = cfg
    h.llm = MagicMock()
    h.parser = MagicMock()
    h.parser.parse.return_value = MagicMock(query="distance?", output_format="text")
    h.vision = MagicMock()
    h.vision.process_file = AsyncMock(return_value="ctx")
    h.vision._decode_data = lambda d: b"imgbytes"
    h.spatial = MagicMock()
    h.spatial.build_scene = AsyncMock(
        return_value=MagicMock(entity_count=1, relations=[], violation_count=0)
    )
    h.reasoner = MagicMock()
    h.reasoner.reason = AsyncMock(return_value="2.0 m")
    h.formatter = MagicMock()
    h.formatter.format_answer = lambda a, f: a
    return h


def _updater():
    up = MagicMock()
    up.update_status = AsyncMock()
    return up


@pytest.mark.asyncio
async def test_scenegraph_path_skips_metric():
    h = _handler("scenegraph")
    out = await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())
    assert out == "2.0 m"
    h.spatial.build_scene.assert_awaited_once()


@pytest.mark.asyncio
async def test_metric_success_uses_metric_scene(monkeypatch):
    h = _handler("metric")
    import fieldwork.perception as perception
    fake_scene = MagicMock(entity_count=2, relations=[1], violation_count=0)
    monkeypatch.setattr(perception, "build_metric_scene", AsyncMock(return_value=fake_scene))

    await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())

    h.spatial.build_scene.assert_not_awaited()       # metric path won
    _, kwargs = h.reasoner.reason.call_args
    assert kwargs["scene"] is fake_scene


@pytest.mark.asyncio
async def test_metric_failure_falls_back_to_scenegraph(monkeypatch):
    h = _handler("metric")
    import fieldwork.perception as perception
    monkeypatch.setattr(
        perception, "build_metric_scene",
        AsyncMock(side_effect=RuntimeError("no gpu server")),
    )

    await h.handle("t", [("a.jpg", "image/jpeg", b"x")], _updater())

    h.spatial.build_scene.assert_awaited_once()      # fell back cleanly
