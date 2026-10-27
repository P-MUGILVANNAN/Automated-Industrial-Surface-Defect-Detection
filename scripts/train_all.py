"""
Master Training Script for Industrial Surface Defect Detection.
Trains PatchCore, CNN Classifier, and YOLO Defect Localizer.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_anomaly_detector import train_patchcore
from src.train_classifier import train_cnn_classifier
from src.utils import setup_logger

logger = setup_logger("train_all")


def main():
    parser = argparse.ArgumentParser(description="Train Defect Detection Models")
    parser.add_argument("--category", type=str, default="bottle", help="Dataset category")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--models", nargs="+", default=["patchcore", "cnn"], help="Models to train: patchcore, cnn, yolo")
    args = parser.parse_args()

    models_to_train = [m.lower() for m in args.models]

    if "patchcore" in models_to_train:
        logger.info(f"--- Training PatchCore Anomaly Detector for '{args.category}' ---")
        try:
            res = train_patchcore(category=args.category, config_path=args.config)
            logger.info(f"PatchCore finished: checkpoint at {res['checkpoint_path']}")
        except Exception as e:
            logger.error(f"Failed to train PatchCore: {e}")

    if "cnn" in models_to_train or "cnn_classifier" in models_to_train:
        logger.info(f"--- Training CNN Classifier for '{args.category}' ---")
        try:
            res = train_cnn_classifier(category=args.category, config_path=args.config)
            logger.info(f"CNN Classifier finished: best accuracy {res['best_val_acc']*100:.1f}%")
        except Exception as e:
            logger.error(f"Failed to train CNN Classifier: {e}")


if __name__ == "__main__":
    main()
