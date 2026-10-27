"""
Unit tests for Dataset Discovery, Validation, and Loading.
"""

from pathlib import Path
import numpy as np
import pytest

from src.dataset_loader import MVTecDatasetManager
from src.dataset_validation import DatasetValidator


def test_mvtec_discovery():
    manager = MVTecDatasetManager("data/mvtec_anomaly_detection")
    categories = manager.get_available_categories()
    assert isinstance(categories, list)
    if categories:
        assert "bottle" in categories


def test_category_files_and_masks():
    manager = MVTecDatasetManager("data/mvtec_anomaly_detection")
    categories = manager.get_available_categories()
    if not categories:
        pytest.skip("No MVTec categories available.")

    cat = categories[0]
    files = manager.get_category_files(cat)
    assert "train_images" in files
    assert "test_good_images" in files
    assert "test_defects" in files
    assert len(files["train_images"]) > 0


def test_dataset_validator():
    validator = DatasetValidator("data/mvtec_anomaly_detection")
    report = validator.validate_dataset()
    assert "is_available" in report
    if report["is_available"] and report["categories"]:
        assert len(report["categories"]) > 0
        cat = report["categories"][0]
        cat_detail = report["category_details"][cat]
        assert cat_detail["is_valid"] is True


def test_mask_to_bounding_boxes():
    # Synthetic mask with 50x50 white square
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 50:100] = 255

    boxes = MVTecDatasetManager.mask_to_bounding_boxes(mask, min_area=20, normalize_yolo=False)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert 48 <= x1 <= 52
    assert 48 <= y1 <= 52
    assert 98 <= x2 <= 102
    assert 98 <= y2 <= 102
