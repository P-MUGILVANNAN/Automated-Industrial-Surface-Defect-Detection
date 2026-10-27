"""
Dataset Validation, Statistics, and Data Integrity Module.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.dataset_loader import MVTecDatasetManager
from src.utils import setup_logger

logger = setup_logger("dataset_validation")


class DatasetValidator:
    """
    Validates MVTec AD dataset directory integrity, checks image and mask dimensions,
    detects potential data leaks, and extracts comprehensive statistics.
    """

    def __init__(self, dataset_root: Union[str, Path] = "data/mvtec_anomaly_detection"):
        self.manager = MVTecDatasetManager(dataset_root)

    def validate_category(self, category: str) -> Dict[str, Any]:
        """
        Validates structure and file consistency for a single category.
        Returns detailed status, statistics, and any errors found.
        """
        errors = []
        warnings = []

        try:
            files_info = self.manager.get_category_files(category)
        except Exception as e:
            return {
                "category": category,
                "is_valid": False,
                "errors": [str(e)],
                "warnings": [],
                "statistics": {}
            }

        train_imgs = files_info["train_images"]
        test_good = files_info["test_good_images"]
        test_defects = files_info["test_defects"]

        if len(train_imgs) == 0:
            errors.append(f"No training images found in 'train/good' for {category}.")

        if len(test_good) == 0:
            warnings.append(f"No normal test images found in 'test/good' for {category}.")

        if len(test_defects) == 0:
            warnings.append(f"No defective test images found for {category}.")

        # Check for train-test file overlap (data leakage)
        train_names = {p.name for p in train_imgs}
        test_names = {p.name for p in test_good}
        leakage = train_names.intersection(test_names)
        if leakage:
            warnings.append(f"Potential filename collision between train and test: {list(leakage)[:5]}")

        # Dimension validation on sample images
        dimensions = set()
        for p in (train_imgs[:5] + test_good[:5]):
            try:
                with Image.open(p) as img:
                    dimensions.add(img.size)  # (width, height)
            except Exception as e:
                errors.append(f"Cannot read image {p}: {e}")

        # Check defect masks existence and dimension consistency
        missing_masks = 0
        mismatched_mask_dims = 0
        for item in test_defects[:20]:  # sample check
            img_path = item["image_path"]
            mask_path = item.get("mask_path")
            if mask_path is None or not Path(mask_path).exists():
                missing_masks += 1
            else:
                try:
                    with Image.open(img_path) as im, Image.open(mask_path) as m:
                        if im.size != m.size:
                            mismatched_mask_dims += 1
                except Exception as e:
                    errors.append(f"Error checking mask {mask_path}: {e}")

        if missing_masks > 0:
            warnings.append(f"{missing_masks} defective samples are missing ground-truth masks.")
        if mismatched_mask_dims > 0:
            errors.append(f"{mismatched_mask_dims} masks have dimensions different from source images.")

        # Compute category distribution
        defect_distribution = {}
        for item in test_defects:
            dt = item["defect_type"]
            defect_distribution[dt] = defect_distribution.get(dt, 0) + 1

        stats = {
            "num_train_good": len(train_imgs),
            "num_test_good": len(test_good),
            "num_test_defective": len(test_defects),
            "total_test": len(test_good) + len(test_defects),
            "defect_types": list(defect_distribution.keys()),
            "defect_distribution": defect_distribution,
            "detected_dimensions": list(dimensions),
        }

        is_valid = len(errors) == 0

        return {
            "category": category,
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "statistics": stats
        }

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validates the entire dataset across all discovered categories.
        """
        if not self.manager.is_available():
            return {
                "is_available": False,
                "message": (
                    f"Dataset directory '{self.manager.root}' does not exist. "
                    "Please download MVTec AD or generate a synthetic demo dataset."
                ),
                "categories": [],
                "summary": {}
            }

        categories = self.manager.get_available_categories()
        if not categories:
            return {
                "is_available": True,
                "message": (
                    f"Directory '{self.manager.root}' exists but no valid MVTec categories "
                    "with 'train/good/' folders were found."
                ),
                "categories": [],
                "summary": {}
            }

        cat_results = {}
        total_train = 0
        total_test_good = 0
        total_test_defective = 0

        for cat in categories:
            res = self.validate_category(cat)
            cat_results[cat] = res
            st = res.get("statistics", {})
            total_train += st.get("num_train_good", 0)
            total_test_good += st.get("num_test_good", 0)
            total_test_defective += st.get("num_test_defective", 0)

        return {
            "is_available": True,
            "message": f"Successfully verified {len(categories)} MVTec categories.",
            "categories": categories,
            "category_details": cat_results,
            "summary": {
                "total_categories": len(categories),
                "total_train_good": total_train,
                "total_test_good": total_test_good,
                "total_test_defective": total_test_defective,
                "total_images": total_train + total_test_good + total_test_defective,
            }
        }
