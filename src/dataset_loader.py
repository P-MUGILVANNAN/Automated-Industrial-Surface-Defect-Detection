"""
MVTec Anomaly Detection Dataset Loader and Annotation Utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.preprocessing import PreprocessingPipeline


class MVTecDatasetManager:
    """
    Manages discovery, indexing, and splitting of MVTec AD categories.
    Strictly follows MVTec AD directory layout:
      <root>/<category>/train/good/
      <root>/<category>/test/<defect_type>/
      <root>/<category>/ground_truth/<defect_type>/
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, dataset_root: Union[str, Path] = "data/mvtec_anomaly_detection"):
        self.root = Path(dataset_root)

    def is_available(self) -> bool:
        """Checks if dataset root directory exists."""
        return self.root.exists() and self.root.is_dir()

    def get_available_categories(self) -> List[str]:
        """Scans root directory and returns valid MVTec category folders."""
        if not self.is_available():
            return []
        categories = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and (p / "train" / "good").exists():
                categories.append(p.name)
        return categories

    def get_category_files(self, category: str) -> Dict[str, Any]:
        """
        Discovers all train, test normal, and test defective files for a category.
        Pairs defective images with ground-truth masks.
        """
        cat_dir = self.root / category
        if not cat_dir.exists():
            raise FileNotFoundError(f"Category '{category}' does not exist in {self.root}")

        train_good_dir = cat_dir / "train" / "good"
        train_images = [
            p for p in sorted(train_good_dir.iterdir())
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        test_good_images: List[Path] = []
        test_defects: List[Dict[str, Any]] = []

        if test_dir.exists():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                defect_name = sub.name
                images = [
                    p for p in sorted(sub.iterdir())
                    if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
                ]
                if defect_name == "good":
                    test_good_images.extend(images)
                else:
                    for img_path in images:
                        # MVTec mask naming convention: <stem>_mask.png
                        mask_path = gt_dir / defect_name / f"{img_path.stem}_mask.png"
                        if not mask_path.exists():
                            # Fallback if without _mask
                            alt_mask = gt_dir / defect_name / f"{img_path.name}"
                            mask_path = alt_mask if alt_mask.exists() else None

                        test_defects.append({
                            "image_path": img_path,
                            "defect_type": defect_name,
                            "mask_path": mask_path,
                            "is_defective": True
                        })

        return {
            "category": category,
            "train_images": train_images,
            "test_good_images": test_good_images,
            "test_defects": test_defects,
            "defect_types": sorted(list({d["defect_type"] for d in test_defects})),
        }

    @staticmethod
    def load_mask(mask_path: Optional[Union[str, Path]], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Loads ground-truth mask as binary uint8 array (0 or 255).
        If mask_path is None, returns all-zero mask of target_size.
        """
        if mask_path is None:
            size = target_size or (256, 256)
            return np.zeros((size[1], size[0]), dtype=np.uint8)

        path = Path(mask_path)
        if not path.exists():
            size = target_size or (256, 256)
            return np.zeros((size[1], size[0]), dtype=np.uint8)

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            size = target_size or (256, 256)
            return np.zeros((size[1], size[0]), dtype=np.uint8)

        # Threshold to binary 0 or 255
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if target_size is not None:
            target_w, target_h = target_size
            binary_mask = cv2.resize(binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return binary_mask

    @staticmethod
    def mask_to_bounding_boxes(
        mask: np.ndarray,
        min_area: int = 10,
        normalize_yolo: bool = False
    ) -> List[Tuple[float, ...]]:
        """
        Extracts connected component bounding boxes from a binary mask.
        Returns list of (xmin, ymin, xmax, ymax) or (x_center, y_center, w, h) if normalized.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = mask.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if normalize_yolo:
                xc = (x + bw / 2.0) / w
                yc = (y + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                boxes.append((xc, yc, nw, nh))
            else:
                boxes.append((x, y, x + bw, y + bh))

        return boxes


class MVTecTrainDataset(Dataset):
    """PyTorch Dataset for normal-only industrial training images."""

    def __init__(self, image_paths: List[Path], image_size: Tuple[int, int] = (256, 256)):
        self.image_paths = image_paths
        self.preprocessor = PreprocessingPipeline(target_size=image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        res = self.preprocessor.preprocess(path)
        return {
            "tensor": res["tensor"].squeeze(0),  # [C, H, W]
            "path": str(path),
            "label": 0  # 0: normal
        }


class MVTecEvaluationDataset(Dataset):
    """
    PyTorch Dataset for evaluation containing both normal and defective test images,
    along with corresponding ground-truth masks.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        samples: list of dicts with:
          image_path: Path
          label: 0 for normal, 1 for defect
          defect_type: str ("good", "scratch", etc.)
          mask_path: Optional[Path]
        """
        self.samples = samples
        self.image_size = image_size
        self.preprocessor = PreprocessingPipeline(target_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_res = self.preprocessor.preprocess(item["image_path"])
        mask = MVTecDatasetManager.load_mask(item.get("mask_path"), target_size=self.image_size)
        mask_binary = (mask > 127).astype(np.uint8)

        return {
            "tensor": img_res["tensor"].squeeze(0),
            "original_image": img_res["original_image"],
            "mask": mask_binary,
            "label": item["label"],
            "defect_type": item.get("defect_type", "good"),
            "path": str(item["image_path"])
        }
