"""
Dataset preparation, validation, and demo data generator for MVTec AD.
"""

import argparse
import os
import sys
from pathlib import Path
import random

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.dataset_validation import DatasetValidator
from src.utils import setup_logger, load_config

logger = setup_logger("prepare_dataset")


def generate_synthetic_industrial_category(
    output_root: Path,
    category: str = "bottle",
    num_train: int = 30,
    num_test_good: int = 10,
    num_test_defective: int = 15,
    img_size: int = 256,
):
    """
    Generates a realistic procedural synthetic industrial dataset conforming strictly
    to the MVTec AD format. Enables instant testing and end-to-end training without
    forcing an immediate 5GB download.
    """
    cat_dir = output_root / category
    train_dir = cat_dir / "train" / "good"
    test_good_dir = cat_dir / "test" / "good"
    test_scratch_dir = cat_dir / "test" / "scratch"
    test_contamination_dir = cat_dir / "test" / "contamination"
    gt_scratch_dir = cat_dir / "ground_truth" / "scratch"
    gt_contamination_dir = cat_dir / "ground_truth" / "contamination"

    for d in [train_dir, test_good_dir, test_scratch_dir, test_contamination_dir,
              gt_scratch_dir, gt_contamination_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic MVTec category '{category}' at {cat_dir.resolve()}...")

    def make_base_surface(seed_val: int) -> np.ndarray:
        np.random.seed(seed_val)
        # Create subtle smooth metallic/glass surface gradient
        x = np.linspace(0, 1, img_size)
        y = np.linspace(0, 1, img_size)
        xx, yy = np.meshgrid(x, y)
        base = 140 + 40 * np.sin(2 * np.pi * xx) * np.cos(np.pi * yy)
        # Add fine texture noise
        noise = np.random.normal(0, 4, (img_size, img_size))
        surface = np.clip(base + noise, 0, 255).astype(np.uint8)
        # Convert to 3-channel BGR (simulate bottle or metal part with subtle tint)
        b = np.clip(surface * 1.05, 0, 255).astype(np.uint8)
        g = np.clip(surface * 0.98, 0, 255).astype(np.uint8)
        r = np.clip(surface * 0.95, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    # 1. Generate Training Normal Images
    for i in range(num_train):
        img = make_base_surface(i + 100)
        cv2.imwrite(str(train_dir / f"{i:03d}.png"), img)

    # 2. Generate Test Normal Images
    for i in range(num_test_good):
        img = make_base_surface(i + 500)
        cv2.imwrite(str(test_good_dir / f"{i:03d}.png"), img)

    # 3. Generate Test Scratch Defect Images + Masks
    num_scratch = num_test_defective // 2
    for i in range(num_scratch):
        img = make_base_surface(i + 1000)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Draw a randomized jagged scratch
        pt1 = (random.randint(40, img_size - 40), random.randint(40, img_size - 40))
        pt2 = (pt1[0] + random.randint(-50, 50), pt1[1] + random.randint(-50, 50))
        thickness = random.randint(2, 4)

        cv2.line(mask, pt1, pt2, 255, thickness)
        # Dark scratch on surface
        scratch_color = (30, 30, 35)
        for c in range(3):
            img[:, :, c] = np.where(mask > 0, scratch_color[c], img[:, :, c])

        cv2.imwrite(str(test_scratch_dir / f"{i:03d}.png"), img)
        cv2.imwrite(str(gt_scratch_dir / f"{i:03d}_mask.png"), mask)

    # 4. Generate Test Contamination Defect Images + Masks
    num_contam = num_test_defective - num_scratch
    for i in range(num_contam):
        img = make_base_surface(i + 2000)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Circular or elliptical contamination stain
        center = (random.randint(60, img_size - 60), random.randint(60, img_size - 60))
        axes = (random.randint(8, 22), random.randint(8, 22))
        angle = random.randint(0, 180)

        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        # Bright or dark oil stain
        stain_color = (random.randint(20, 50), random.randint(80, 140), random.randint(180, 230))
        for c in range(3):
            img[:, :, c] = np.where(mask > 0, stain_color[c], img[:, :, c])

        cv2.imwrite(str(test_contamination_dir / f"{i:03d}.png"), img)
        cv2.imwrite(str(gt_contamination_dir / f"{i:03d}_mask.png"), mask)

    logger.info(f"Synthetic dataset for category '{category}' generated successfully!")
    logger.info(f" - Train good: {num_train}")
    logger.info(f" - Test good: {num_test_good}")
    logger.info(f" - Test defects: {num_test_defective} (scratch: {num_scratch}, contamination: {num_contam})")


def main():
    parser = argparse.ArgumentParser(description="MVTec Dataset Preparation & Validation")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--generate-demo", action="store_true", help="Generate synthetic MVTec category for testing")
    parser.add_argument("--category", type=str, default="bottle", help="Category name")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(cfg["paths"]["dataset_root"])

    if args.generate_demo:
        generate_synthetic_industrial_category(dataset_root, category=args.category)

    validator = DatasetValidator(dataset_root)
    val_report = validator.validate_dataset()
    print("\n--- DATASET VALIDATION REPORT ---")
    print(f"Status: {val_report['message']}")
    if val_report["is_available"] and val_report["categories"]:
        print(f"Discovered Categories ({len(val_report['categories'])}): {val_report['categories']}")
        for cat, det in val_report["category_details"].items():
            print(f"Category [{cat}]: Valid={det['is_valid']}, Train={det['statistics'].get('num_train_good')}, "
                  f"Test Good={det['statistics'].get('num_test_good')}, Test Defects={det['statistics'].get('num_test_defective')}")
            if det["warnings"]:
                print(f"  Warnings: {det['warnings']}")
            if det["errors"]:
                print(f"  Errors: {det['errors']}")


if __name__ == "__main__":
    main()
