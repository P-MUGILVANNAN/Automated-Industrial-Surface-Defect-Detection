"""
Unit tests for Preprocessing, Enhancement, and Image Quality Validation.
"""

import numpy as np
import pytest
import torch

from src.preprocessing import PreprocessingPipeline


def test_preprocessing_tensor_output():
    pipeline = PreprocessingPipeline(target_size=(128, 128))
    # Synthetic RGB image [100, 100, 3]
    fake_img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    res = pipeline.preprocess(fake_img)
    assert "tensor" in res
    assert "quality" in res
    assert isinstance(res["tensor"], torch.Tensor)
    assert res["tensor"].shape == (1, 3, 128, 128)
    assert res["original_shape"] == (100, 100)


def test_quality_assessment_warnings():
    pipeline = PreprocessingPipeline(blur_threshold=50.0, min_brightness=30.0, max_brightness=230.0)

    # Completely flat image (extreme blur / zero gradient)
    flat_img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    report = pipeline.assess_quality(flat_img)

    assert report["sharpness"] < 1.0
    assert not report["is_acceptable"]
    assert any("blurry" in w for w in report["warnings"])


def test_scale_mask_to_original():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    scaled = PreprocessingPipeline.scale_mask_to_original(mask, (128, 256))
    assert scaled.shape == (128, 256)
    assert np.max(scaled) == 255
