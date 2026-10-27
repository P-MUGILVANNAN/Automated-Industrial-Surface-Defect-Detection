"""
Unified Model Loader and Checkpoint Registry.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from models.patchcore_detector import PatchCoreDetector
from models.cnn_classifier import CNNDefectClassifier
from models.yolo_detector import YOLODefectDetector
from src.utils import setup_logger, get_device

logger = setup_logger("model_loader")


class ModelRegistry:
    """Discovers, instantiates, and loads industrial defect models."""

    @staticmethod
    def get_available_checkpoints(checkpoints_dir: Union[str, Path] = "artifacts/checkpoints") -> Dict[str, List[Path]]:
        """Scans checkpoints folder and categorizes available model files."""
        cp_dir = Path(checkpoints_dir)
        if not cp_dir.exists():
            return {"patchcore": [], "cnn_classifier": [], "yolo": []}

        patchcore_cps = sorted(list(cp_dir.glob("*.pkl")) + list(cp_dir.glob("*patchcore*.pt")))
        cnn_cps = sorted([p for p in cp_dir.glob("*cnn*.pt") if "patchcore" not in p.name])
        yolo_cps = sorted([p for p in cp_dir.glob("*yolo*.pt") if "cnn" not in p.name])

        return {
            "patchcore": patchcore_cps,
            "cnn_classifier": cnn_cps,
            "yolo": yolo_cps,
        }

    @staticmethod
    def load_model(
        model_type: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> Any:
        """
        Loads specified model type ('patchcore', 'cnn_classifier', 'yolo').
        If checkpoint_path is None, instantiates a fresh model using config parameters.
        """
        dev = device or get_device()
        m_type = model_type.lower()

        if m_type == "patchcore":
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Loading PatchCore checkpoint from {checkpoint_path}")
                return PatchCoreDetector.load_checkpoint(checkpoint_path, device=dev)
            else:
                cfg_patch = (config or {}).get("models", {}).get("patchcore", {})
                backbone = cfg_patch.get("backbone", "resnet18")
                layers = cfg_patch.get("layers", ["layer2", "layer3"])
                coreset = cfg_patch.get("coreset_ratio", 0.10)
                n_neighbors = cfg_patch.get("n_neighbors", 9)
                return PatchCoreDetector(
                    backbone_name=backbone,
                    layers=layers,
                    coreset_ratio=coreset,
                    n_neighbors=n_neighbors,
                    device=dev
                )

        elif m_type in ("cnn", "cnn_classifier"):
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Loading CNN Classifier checkpoint from {checkpoint_path}")
                return CNNDefectClassifier.load_checkpoint(checkpoint_path, device=dev)
            else:
                cfg_cnn = (config or {}).get("models", {}).get("cnn_classifier", {})
                backbone = cfg_cnn.get("backbone", "resnet18")
                num_classes = cfg_cnn.get("num_classes", 2)
                return CNNDefectClassifier(backbone_name=backbone, num_classes=num_classes, device=dev)

        elif m_type in ("yolo", "yolo_detector"):
            yolo_path = str(checkpoint_path) if (checkpoint_path and Path(checkpoint_path).exists()) else "yolov8n.pt"
            device_str = "cuda" if dev.type == "cuda" else "cpu"
            return YOLODefectDetector(model_path=yolo_path, device=device_str)

        else:
            raise ValueError(f"Unknown model type '{model_type}'. Choose from 'patchcore', 'cnn_classifier', 'yolo'.")
