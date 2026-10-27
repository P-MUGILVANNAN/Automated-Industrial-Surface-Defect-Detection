"""
Training Pipeline for Supervised CNN Defect Classifier.
Trains ResNet18 on labeled normal and defective samples.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from models.cnn_classifier import CNNDefectClassifier
from src.dataset_loader import MVTecDatasetManager
from src.preprocessing import PreprocessingPipeline
from src.utils import setup_logger, load_config, get_device, set_seed

logger = setup_logger("train_classifier")


class SupervisedDefectDataset(Dataset):
    """Dataset pairing images with binary labels (0: normal, 1: defect)."""

    def __init__(self, items: List[Tuple[Path, int]], image_size: Tuple[int, int] = (256, 256)):
        self.items = items
        self.preprocessor = PreprocessingPipeline(target_size=image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, label = self.items[idx]
        res = self.preprocessor.preprocess(path)
        return {
            "tensor": res["tensor"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path)
        }


def train_cnn_classifier(
    category: str = "bottle",
    config_path: Union[str, Path] = "config/config.yaml",
    dataset_root: Optional[Union[str, Path]] = None,
    output_checkpoint: Optional[Union[str, Path]] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Trains CNN classifier on available labeled data for the category.
    Uses normal training images and a subset of labeled defective images.
    """
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    set_seed(cfg.get("project", {}).get("random_seed", 42))

    dev = device or get_device(cfg.get("hardware", {}).get("device", "auto"))
    ds_root = Path(dataset_root or cfg["paths"]["dataset_root"])

    manager = MVTecDatasetManager(ds_root)
    files = manager.get_category_files(category)

    normal_imgs = files["train_images"]
    defective_items = files["test_defects"]

    if len(normal_imgs) == 0 or len(defective_items) == 0:
        raise ValueError(
            f"Supervised classifier requires both normal and labeled defect images for category '{category}'."
        )

    # Construct labeled dataset pairs: (path, label)
    # 0 = normal, 1 = defect
    normal_pairs = [(p, 0) for p in normal_imgs]
    defect_pairs = [(d["image_path"], 1) for d in defective_items]

    # Balance train/val split (80% train, 20% val)
    np.random.shuffle(normal_pairs)
    np.random.shuffle(defect_pairs)

    n_val_norm = max(1, int(len(normal_pairs) * 0.2))
    n_val_def = max(1, int(len(defect_pairs) * 0.2))

    val_pairs = normal_pairs[:n_val_norm] + defect_pairs[:n_val_def]
    train_pairs = normal_pairs[n_val_norm:] + defect_pairs[n_val_def:]

    img_size = tuple(cfg.get("dataset", {}).get("image_size", [256, 256]))
    train_ds = SupervisedDefectDataset(train_pairs, image_size=img_size)
    val_ds = SupervisedDefectDataset(val_pairs, image_size=img_size)

    cnn_cfg = cfg.get("models", {}).get("cnn_classifier", {})
    batch_size = cnn_cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_epochs = epochs if epochs is not None else cnn_cfg.get("epochs", 10)
    learning_rate = lr if lr is not None else cnn_cfg.get("learning_rate", 0.001)

    model = CNNDefectClassifier(
        backbone_name=cnn_cfg.get("backbone", "resnet18"),
        num_classes=2,
        pretrained=cnn_cfg.get("pretrained", True),
        device=dev
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    logger.info(f"Starting CNN training for {num_epochs} epochs on {len(train_pairs)} training samples...")

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            x = batch["tensor"].to(dev)
            y = batch["label"].to(dev)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += x.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["tensor"].to(dev)
                y = batch["label"].to(dev)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y).sum().item()
                total_val += x.size(0)

        epoch_val_loss = val_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.1f}%"
            )

    # Save checkpoint
    cp_dir = Path(cfg["paths"]["checkpoints_dir"])
    cp_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_checkpoint) if output_checkpoint else cp_dir / f"{category}_cnn.pt"

    if best_state is not None:
        model.load_state_dict(best_state)

    train_duration = time.perf_counter() - t0
    metadata = {
        "category": category,
        "best_val_acc": float(best_val_acc),
        "history": history,
        "epochs": num_epochs,
        "training_time_seconds": train_duration,
        "num_train_samples": len(train_pairs),
        "num_val_samples": len(val_pairs),
        "device": str(dev),
    }

    model.save_checkpoint(out_path, metadata=metadata)
    logger.info(f"CNN Classifier checkpoint saved to {out_path.resolve()}")

    return {
        "status": "success",
        "category": category,
        "checkpoint_path": str(out_path),
        "best_val_acc": float(best_val_acc),
        "history": history,
        "training_time_seconds": train_duration,
    }


if __name__ == "__main__":
    train_cnn_classifier()
