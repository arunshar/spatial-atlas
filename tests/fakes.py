"""In-memory fakes for SpatialClaw's GPU tools (no GPU, no server, no network).

Used to test the metric perception backend's orchestration + geometry by behavior,
not by patching internals (fakes-over-mocks, per Google testing guidance).
"""


class FakePerFrameMask:
    """Stand-in for SpatialClaw's PerFrameMask."""

    def __init__(self, num_objects=1, centroid=(0.0, 0.0, 0.0)):
        self.num_objects = num_objects
        self.frame_indices = [0]
        self._centroid = centroid

    def get_centroid_3d(self, recon, frame, object):  # noqa: A002 - mirrors real signature
        return None if self.num_objects == 0 else self._centroid


class FakeSAM3:
    """Stand-in for SAM3Tool: maps a text phrase to a preset mask."""

    def __init__(self, by_phrase):
        self._by_phrase = by_phrase
        self.is_available = True

    def segment_image_by_text(self, img, phrase):
        return self._by_phrase[phrase]


class _FakeRecon:
    frame_indices = [0]
    metric_scale = 1.0


class FakeReconstructTool:
    """Stand-in for ReconstructTool."""

    is_available = True

    def Reconstruct(self, frames):  # noqa: N802 - mirrors real method name
        return _FakeRecon()
