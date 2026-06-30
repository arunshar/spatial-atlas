"""Config tests for the ATLAS_FIELDWORK_ENGINE flag."""
import importlib

import pytest

pytestmark = pytest.mark.unit


def test_default_engine_is_scenegraph(monkeypatch):
    monkeypatch.delenv("ATLAS_FIELDWORK_ENGINE", raising=False)
    import config as c
    importlib.reload(c)
    cfg = c.Config()
    assert cfg.fieldwork_engine == "scenegraph"
    assert cfg.reconstruct_max_frames == 32


def test_env_override_metric(monkeypatch):
    monkeypatch.setenv("ATLAS_FIELDWORK_ENGINE", "metric")
    import config as c
    importlib.reload(c)
    assert c.Config().fieldwork_engine == "metric"


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ATLAS_FIELDWORK_ENGINE", "")
    import config as c
    importlib.reload(c)
    assert c.Config().fieldwork_engine == "scenegraph"
