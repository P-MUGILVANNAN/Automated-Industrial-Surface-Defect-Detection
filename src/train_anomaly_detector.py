"""
Training Pipeline for PatchCore Anomaly Detector.
Trains normal-only feature memory banks and calibrates baseline thresholds.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.patchcore_detector import PatchCoreDetector
from src.dataset_loader import MVTecDatasetManager, MVTecTrainDataset
from src.threshold_calibration import ThresholdCalibrator
from src.utils import setup_logger, load_config, get_device, set_seed

logger = setup_logger("train_anomaly_detector")


def train_patchcore(
    category: str = "bottle",
    config_path: Union[str, Path] = "config/config.yaml",
    dataset_root: Optional[Union[str, Path]] = None,
    output_checkpoint: Optional[Union[str, Path]] = None,
    coreset_ratio: Optional[float] = None,
    device: Optional[torch.device] = None,
    val_split_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Trains PatchCore on normal training images for the given category.
    Uses a partitioned validation subset of normal training images for threshold calibration.
    """
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    set_seed(cfg.get("project", {}).get("random_seed", 42))

    dev = device or get_device(cfg.get("hardware", {}).get("device", "auto"))
    ds_root = Path(dataset_root or cfg["paths"]["dataset_root"])

    # Load dataset files
    manager = MVTecDatasetManager(ds_root)
    files = manager.get_category_files(category)
    train_imgs = files["train_images"]

    if len(train_imgs) == 0:
        raise ValueError(f"No normal training images found for category '{category}' in {ds_root}")

    # Split train normal images into pure train and validation-normal for threshold calibration
    # Strictly ensures NO test data is used for threshold calibration
    num_val = max(1, int(len(train_imgs) * val_split_ratio))
    val_normal_imgs = train_imgs[:num_val]
    fit_normal_imgs = train_imgs[num_val:] if len(train_imgs) > num_val else train_imgs

    img_size = tuple(cfg.get("dataset", {}).get("image_size", [256, 256]))
    train_dataset = MVTecTrainDataset(fit_normal_imgs, image_size=img_size)
    val_dataset = MVTecTrainDataset(val_normal_imgs, image_size=img_size)

    batch_size = cfg.get("models", {}).get("cnn_classifier", {}).get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize PatchCore model
    p_cfg = cfg.get("models", {}).get("patchcore", {})
    backbone = p_cfg.get("backbone", "resnet18")
    layers = p_cfg.get("layers", ["layer2", "layer3"])
    c_ratio = coreset_ratio if coreset_ratio is not None else p_cfg.get("coreset_ratio", 0.10)
    n_neighbors = p_cfg.get("n_neighbors", 9)

    logger.info(f"Initializing PatchCore ({backbone}, layers={layers}, coreset={c_ratio}) on device: {dev}")
    detector = PatchCoreDetector(
        backbone_name=backbone,
        layers=layers,
        coreset_ratio=c_ratio,
        n_neighbors=n_neighbors,
        device=dev
    )

    # Build memory bank
    logger.info(f"Extracting features and building memory bank from {len(fit_normal_imgs)} normal images...")
    detector.fit(train_loader)
    logger.info(f"Memory bank constructed. Features count: {len(detector.memory_bank)}")

    # Calibrate threshold on normal validation partition
    logger.info(f"Calibrating threshold on {len(val_normal_imgs)} normal validation images...")
    val_normal_scores = []
    with torch.no_grad():
        for batch in val_loader:
            tensor = batch["tensor"].to(dev)
            _, score = detector.predict_anomaly_map(tensor)
            val_normal_scores.append(score)

    target_fpr = cfg.get("calibration", {}).get("target_fpr", 0.05)
    calibrator = ThresholdCalibrator(target_fpr=target_fpr)
    calib_meta = calibrator.calibrate_on_normal_validation(val_normal_scores)
    detector.calibrated_threshold = calib_meta["threshold"]
    detector.calibrate_scores(val_normal_scores)

    logger.info(f"Calibrated threshold: {detector.calibrated_threshold:.4f} (target FPR: {target_fpr*100:.1f}%)")

    # Save checkpoint
    cp_dir = Path(cfg["paths"]["checkpoints_dir"])
    cp_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_checkpoint) if output_checkpoint else cp_dir / f"{category}_patchcore.pkl"

    train_duration = time.perf_counter() - t0
    metadata = {
        "category": category,
        "dataset_root": str(ds_root),
        "num_train_images": len(fit_normal_imgs),
        "num_val_images": len(val_normal_imgs),
        "coreset_size": len(detector.memory_bank),
        "training_time_seconds": train_duration,
        "calibration": calib_meta,
        "device": str(dev),
    }

    detector.save_checkpoint(out_path, metadata=metadata)
    logger.info(f"PatchCore checkpoint saved successfully to {out_path.resolve()}")

    return {
        "status": "success",
        "category": category,
        "checkpoint_path": str(out_path),
        "calibrated_threshold": detector.calibrated_threshold,
        "training_time_seconds": train_duration,
        "num_normal_samples": len(fit_normal_imgs),
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PatchCore Anomaly Detector")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    train_patchcore(category=args.category, config_path=args.config)
