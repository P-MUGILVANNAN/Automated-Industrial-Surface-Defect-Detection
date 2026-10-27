"""
CNN-Based Supervised Defect Classifier Baseline (ResNet18).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNDefectClassifier(nn.Module):
    """
    Supervised CNN classifier baseline for surface defect classification.
    Distinguishes between normal surfaces and known defective classes.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.class_names = ["normal", "defect"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, tensor: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """
        Runs inference on input tensor [1, C, H, W].
        Returns:
          predicted_class: class label string
          confidence: float probability of predicted class
          all_probabilities: 1D numpy array of probabilities across all classes
        """
        self.eval()
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.forward(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            class_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"

        return class_name, confidence, probs

    def save_checkpoint(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Saves weights, architecture config, and class mappings."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": "cnn_classifier",
            "backbone_name": self.backbone_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "state_dict": self.state_dict(),
            "metadata": metadata or {}
        }, path)

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: Optional[torch.device] = None) -> "CNNDefectClassifier":
        """Restores model from checkpoint."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        dev = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=dev)

        model = cls(
            backbone_name=checkpoint["backbone_name"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,
            device=dev
        )
        model.class_names = checkpoint.get("class_names", model.class_names)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
