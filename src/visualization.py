"""
Visualization, Heatmap Blending, and Annotation Utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """Utilities for rendering inspection results, anomaly heatmaps, and diagnostic plots."""

    @staticmethod
    def overlay_heatmap_on_image(
        image_rgb: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays a 2D float anomaly map [0, 1] onto an RGB uint8 image using a heatmap colormap.
        """
        # Ensure dimensions match
        h, w = image_rgb.shape[:2]
        if anomaly_map.shape[:2] != (h, w):
            anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Scale map to [0, 255] uint8
        norm_map = np.clip(anomaly_map, 0.0, 1.0)
        heatmap_gray = (norm_map * 255).astype(np.uint8)

        # Apply colormap (cv2 outputs BGR)
        heatmap_bgr = cv2.applyColorMap(heatmap_gray, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend
        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return blended

    @staticmethod
    def draw_boxes(
        image_rgb: np.ndarray,
        boxes: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (255, 50, 50),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draws defect bounding boxes and labels on an image.
        boxes: list of dicts with 'box' [x1, y1, x2, y2], 'confidence', 'class_name'
        """
        out = image_rgb.copy()
        for b in boxes:
            box = b["box"]
            x1, y1, x2, y2 = [int(v) for v in box]
            label = b.get("class_name", "defect")
            conf = b.get("confidence")

            text = f"{label} ({conf:.2f})" if conf is not None else label

            # Draw rectangle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Draw label banner
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
            cv2.putText(
                out,
                text,
                (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return out

    @staticmethod
    def create_comparison_grid(
        original: np.ndarray,
        heatmap_overlay: np.ndarray,
        binary_mask: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Creates a side-by-side inspection comparison panel.
        """
        h, w = original.shape[:2]

        # Convert 1-channel binary mask to 3-channel RGB
        if binary_mask.ndim == 2:
            mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = binary_mask

        if mask_rgb.shape[:2] != (h, w):
            mask_rgb = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

        panels = [original, heatmap_overlay, mask_rgb]

        if ground_truth is not None:
            if ground_truth.ndim == 2:
                gt_rgb = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2RGB)
            else:
                gt_rgb = ground_truth
            if gt_rgb.shape[:2] != (h, w):
                gt_rgb = cv2.resize(gt_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
            panels.append(gt_rgb)

        grid = np.hstack(panels)
        return grid

    @staticmethod
    def save_plot_figure(fig: plt.Figure, filepath: Union[str, Path]) -> None:
        """Saves a matplotlib figure cleanly to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), bbox_inches="tight", dpi=150)
        plt.close(fig)
