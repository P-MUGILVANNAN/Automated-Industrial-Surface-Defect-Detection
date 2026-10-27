"""
YOLO Object Detection Dataset Preparation and Training Pipeline.
Converts defect masks to YOLO bounding box annotations and trains Ultralytics model.
"""

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.dataset_loader import MVTecDatasetManager
from src.utils import setup_logger, load_config, get_device

logger = setup_logger("train_detector")


def prepare_yolo_dataset(
    category: str = "bottle",
    dataset_root: Union[str, Path] = "data/mvtec_anomaly_detection",
    output_dir: Union[str, Path] = "artifacts/yolo_dataset"
) -> Path:
    """
    Constructs a YOLO-compatible dataset from MVTec defective images and masks.
    Extracts bounding boxes from ground truth masks.
    """
    out_path = Path(output_dir) / category
    images_train = out_path / "images" / "train"
    images_val = out_path / "images" / "val"
    labels_train = out_path / "labels" / "train"
    labels_val = out_path / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    manager = MVTecDatasetManager(dataset_root)
    files = manager.get_category_files(category)

    defective_items = files["test_defects"]
    if not defective_items:
        raise ValueError(f"No defective images with masks available for category '{category}'.")

    # Map defect types to class IDs
    defect_types = files["defect_types"]
    class_to_id = {dt: i for i, dt in enumerate(defect_types)}

    items_processed = 0
    for idx, item in enumerate(defective_items):
        img_p = item["image_path"]
        mask_p = item["mask_path"]
        defect_type = item["defect_type"]
        cls_id = class_to_id[defect_type]

        # 80/20 train/val split
        is_val = (idx % 5 == 0)
        target_img_dir = images_val if is_val else images_train
        target_lbl_dir = labels_val if is_val else labels_train

        # Copy or symlink image
        dest_img = target_img_dir / f"{item['defect_type']}_{img_p.name}"
        shutil.copy2(img_p, dest_img)

        # Generate YOLO bounding boxes from mask
        dest_lbl = target_lbl_dir / f"{item['defect_type']}_{img_p.stem}.txt"
        mask = MVTecDatasetManager.load_mask(mask_p)
        boxes = MVTecDatasetManager.mask_to_bounding_boxes(mask, min_area=20, normalize_yolo=True)

        with open(dest_lbl, "w") as f:
            for b in boxes:
                # xc, yc, w, h
                f.write(f"{cls_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

        items_processed += 1

    # Write data.yaml for YOLO
    data_yaml_path = out_path / "data.yaml"
    data_dict = {
        "path": str(out_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for name, i in class_to_id.items()}
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    logger.info(f"YOLO dataset prepared at {out_path.resolve()} with {items_processed} defect samples.")
    return data_yaml_path


def train_yolo_detector(
    category: str = "bottle",
    config_path: Union[str, Path] = "config/config.yaml",
    dataset_root: Optional[Union[str, Path]] = None,
    output_checkpoint: Optional[Union[str, Path]] = None,
    epochs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains Ultralytics YOLO model on defect bounding boxes.
    """
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    ds_root = dataset_root or cfg["paths"]["dataset_root"]
    y_cfg = cfg.get("models", {}).get("yolo_detector", {})

    data_yaml = prepare_yolo_dataset(category=category, dataset_root=ds_root)

    base_model = y_cfg.get("base_model", "yolov8n.pt")
    num_epochs = epochs if epochs is not None else y_cfg.get("epochs", 15)
    imgsz = y_cfg.get("image_size", 256)

    logger.info(f"Initializing YOLO model '{base_model}' for defect localization...")
    model = YOLO(base_model)

    results = model.train(
        data=str(data_yaml),
        epochs=num_epochs,
        imgsz=imgsz,
        batch=y_cfg.get("batch_size", 16),
        device="cpu",  # safe default
        verbose=False,
        project="artifacts/experiments",
        name=f"yolo_{category}"
    )

    cp_dir = Path(cfg["paths"]["checkpoints_dir"])
    cp_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_checkpoint) if output_checkpoint else cp_dir / f"{category}_yolo.pt"

    # Copy best model weights
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, out_path)
    else:
        model.save(str(out_path))

    duration = time.perf_counter() - t0
    logger.info(f"YOLO Defect Detector saved to {out_path.resolve()}")

    return {
        "status": "success",
        "category": category,
        "checkpoint_path": str(out_path),
        "epochs": num_epochs,
        "training_time_seconds": duration,
    }


if __name__ == "__main__":
    train_yolo_detector()
