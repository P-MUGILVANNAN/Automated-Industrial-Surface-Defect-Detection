"""
Industrial Image Preprocessing, Validation, and Quality Assessment Module.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from PIL import Image


class PreprocessingPipeline:
    """
    Standard preprocessing, validation, and enhancement pipeline for
    industrial surface inspection images.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        apply_clahe: bool = False,
        apply_denoise: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        blur_threshold: float = 40.0,
        min_brightness: float = 25.0,
        max_brightness: float = 240.0,
        min_contrast: float = 20.0,
    ):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.apply_denoise = apply_denoise
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

    @staticmethod
    def load_image(image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads an image from various input types and returns it as an RGB uint8 NumPy array.
        Raises ValueError if the image is corrupted or invalid.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Failed to read image (corrupt or unsupported format): {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb

        if isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Failed to decode image from bytes.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            if image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                if image_input.shape[2] == 3:
                    return image_input.copy()
            raise ValueError(f"Invalid image array shape: {image_input.shape}")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def assess_quality(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Computes image quality metrics: sharpness (Laplacian variance),
        mean brightness, and standard deviation contrast.
        Returns a dictionary with metrics and warning flags.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        warnings: List[str] = []
        if laplacian_var < self.blur_threshold:
            warnings.append(f"Image may be blurry (sharpness score {laplacian_var:.1f} < {self.blur_threshold})")
        if mean_brightness < self.min_brightness:
            warnings.append(f"Image may be underexposed (brightness {mean_brightness:.1f} < {self.min_brightness})")
        elif mean_brightness > self.max_brightness:
            warnings.append(f"Image may be overexposed (brightness {mean_brightness:.1f} > {self.max_brightness})")
        if contrast < self.min_contrast:
            warnings.append(f"Image has low contrast (std {contrast:.1f} < {self.min_contrast})")

        return {
            "sharpness": laplacian_var,
            "mean_brightness": mean_brightness,
            "contrast": contrast,
            "warnings": warnings,
            "is_acceptable": len(warnings) == 0,
        }

    def enhance(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Applies optional gentle contrast enhancement and denoising.
        Kept non-aggressive to preserve subtle defects.
        """
        output = image_rgb.copy()

        if self.apply_denoise:
            output = cv2.bilateralFilter(output, d=5, sigmaColor=30, sigmaSpace=30)

        if self.apply_clahe:
            # Apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            output = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        return output

    def preprocess(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        force_enhance: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Full preprocessing pipeline:
        1. Loads raw image.
        2. Assesses image quality.
        3. Optionally enhances.
        4. Resizes to target dimension.
        5. Normalizes to ImageNet float32 tensor [1, 3, H, W].
        Returns dictionary with raw, enhanced, resized, tensor, and quality info.
        """
        raw_rgb = self.load_image(image_input)
        original_h, original_w = raw_rgb.shape[:2]

        quality_report = self.assess_quality(raw_rgb)

        should_enhance = self.apply_clahe or self.apply_denoise
        if force_enhance is not None:
            should_enhance = force_enhance

        if should_enhance:
            enhanced_rgb = self.enhance(raw_rgb)
        else:
            enhanced_rgb = raw_rgb

        # Resize for model input (Width, Height for cv2)
        target_w, target_h = self.target_size
        resized_rgb = cv2.resize(
            enhanced_rgb,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA if (original_w > target_w) else cv2.INTER_LINEAR
        )

        # Normalize to [0, 1] then ImageNet standardize
        normalized = resized_rgb.astype(np.float32) / 255.0
        standardized = (normalized - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Convert to PyTorch Tensor [1, C, H, W]
        tensor = torch.from_numpy(standardized).permute(2, 0, 1).unsqueeze(0).float()

        return {
            "original_image": raw_rgb,
            "original_shape": (original_h, original_w),
            "enhanced_image": enhanced_rgb,
            "resized_image": resized_rgb,
            "tensor": tensor,
            "quality": quality_report,
        }

    @staticmethod
    def scale_mask_to_original(
        mask: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scales a binary mask or heatmap back to original image dimensions (H, W).
        """
        orig_h, orig_w = original_shape
        scaled = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return scaled
