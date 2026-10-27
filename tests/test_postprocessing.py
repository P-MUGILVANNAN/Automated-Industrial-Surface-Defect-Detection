"""
Unit tests for False-Alarm Filtering and Postprocessing.
"""

import numpy as np
import pytest

from src.postprocessing import FalseAlarmFilter


def test_connected_component_area_filtering():
    filter_obj = FalseAlarmFilter(min_component_area=50, use_smoothing=False, use_morphology=False)

    mask = np.zeros((100, 100), dtype=np.uint8)
    # Tiny noise spot: 3x3 = 9 pixels (< 50)
    mask[10:13, 10:13] = 255
    # Genuine defect blob: 10x10 = 100 pixels (>= 50)
    mask[40:50, 40:50] = 255

    cleaned_mask, comps = filter_obj.filter_components(mask)

    # The 3x3 spot should be removed, the 10x10 preserved
    assert len(comps) == 1
    assert comps[0]["area"] >= 50
    assert cleaned_mask[11, 11] == 0
    assert cleaned_mask[45, 45] == 255


def test_gaussian_smoothing():
    filter_obj = FalseAlarmFilter(gaussian_blur_kernel=5, gaussian_blur_sigma=1.0)
    # Raw map with single pixel spike of 1.0
    raw_map = np.zeros((30, 30), dtype=np.float32)
    raw_map[15, 15] = 1.0

    smoothed = filter_obj.smooth_anomaly_map(raw_map)
    # Spike should be diffused
    assert smoothed[15, 15] < 1.0
    assert smoothed[15, 16] > 0.0


def test_inspection_decision_rules():
    filter_obj = FalseAlarmFilter(min_component_area=20)
    # Case 1: Normal score well below threshold
    raw_map = np.zeros((50, 50), dtype=np.float32)
    res = filter_obj.process(raw_map, threshold=0.5, image_score=0.2)
    assert res["decision"] == "NORMAL"

    # Case 2: Borderline uncertainty score (within uncertainty margin of 0.08)
    res_uncertain = filter_obj.process(raw_map, threshold=0.5, image_score=0.48)
    assert res_uncertain["decision"] == "MANUAL REVIEW REQUIRED"
