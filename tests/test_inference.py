"""
Unit tests for End-to-End Inference Engine.
"""

from pathlib import Path
import numpy as np
import pytest

from src.inference import InspectionEngine
from src.postprocessing import FalseAlarmFilter
from models.patchcore_detector import PatchCoreDetector


class DummyAnomalyModel:
    """Mock model fixture for deterministic inference pipeline tests."""
    def __init__(self):
        self.calibrated_threshold = 0.50

    def predict_anomaly_map(self, tensor):
        # Returns synthetic map with a small defect blob in center
        h, w = tensor.shape[2], tensor.shape[3]
        m = np.zeros((h, w), dtype=np.float32)
        m[h//4:h//2, w//4:w//2] = 0.85
        return m, 0.85


def test_inspection_engine_pipeline():
    dummy_model = DummyAnomalyModel()
    engine = InspectionEngine(model=dummy_model, model_type="patchcore")

    # Generate synthetic image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    res = engine.inspect_image(img)

    assert "decision" in res
    assert "image_score" in res
    assert "overlay_image" in res
    assert "binary_mask" in res
    assert res["image_score"] == 0.85
    assert res["decision"] in ("DEFECT DETECTED", "POSSIBLE UNKNOWN DEFECT", "MANUAL REVIEW REQUIRED")
    assert res["latency_ms"] > 0.0
