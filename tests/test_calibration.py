"""
Unit tests for Threshold Calibration.
"""

import numpy as np
import pytest

from src.threshold_calibration import ThresholdCalibrator


def test_normal_quantile_calibration():
    calibrator = ThresholdCalibrator(target_fpr=0.05)
    # Generate 100 normal scores uniformly distributed between 0.1 and 0.4
    normal_scores = list(np.linspace(0.1, 0.4, 100))

    meta = calibrator.calibrate_on_normal_validation(normal_scores, target_fpr=0.05)
    thresh = meta["threshold"]

    # 95th percentile should be close to 0.385
    assert 0.35 <= thresh <= 0.41
    # Check that empirical validation false alarms are <= 5%
    false_alarms = sum(s > thresh for s in normal_scores)
    assert false_alarms <= 5


def test_optimal_f1_calibration():
    calibrator = ThresholdCalibrator()
    # 50 normal scores around 0.2, 50 defect scores around 0.8
    val_scores = [0.15] * 25 + [0.25] * 25 + [0.75] * 25 + [0.85] * 25
    val_labels = [0] * 50 + [1] * 50

    meta = calibrator.calibrate_optimal_f1(val_scores, val_labels)
    thresh = meta["threshold"]

    # Threshold should separate 0.25 and 0.75 cleanly
    assert 0.20 <= thresh <= 0.75
    assert meta["best_f1"] == 1.0
