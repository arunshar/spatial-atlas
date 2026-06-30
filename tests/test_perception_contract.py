"""Contract test: the metric backend's output must satisfy the SpatialScene shape
that the downstream FieldWorkReasoner + AnswerFormatter consume unchanged."""
import pytest

from config import Config
from fieldwork.perception import MetricPerceptionBackend
from fakes import FakePerFrameMask, FakeReconstructTool, FakeSAM3

pytestmark = pytest.mark.unit


def test_scene_factsheet_carries_measured_distance(png_bytes):
    b = MetricPerceptionBackend(Config())
    b._sam3 = FakeSAM3({
        "worker": FakePerFrameMask(1, (0.0, 1.7, 0.0)),
        "forklift": FakePerFrameMask(1, (3.0, 0.5, 4.0)),
    })
    b._recon = FakeReconstructTool()
    scene = b.ground(png_bytes, ["worker", "forklift"])

    # to_fact_sheet() is fed verbatim into the reasoner prompt; it must be
    # non-empty and contain the measured distance.
    sheet = scene.to_fact_sheet()
    assert sheet
    assert "5.0" in sheet
    # downstream-consumed properties exist
    assert scene.entity_count == 2
    assert hasattr(scene, "violation_count")
