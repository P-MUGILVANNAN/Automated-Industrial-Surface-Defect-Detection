"""
False-Alarm Reduction, Morphological Filtering, and Decision Engine.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class FalseAlarmFilter:
    """
    Configurable postprocessing pipeline designed to suppress high-frequency noise spikes,
    spurious edge artifacts, and isolated false alarms while preserving genuine defect structures.
    """

    def __init__(
        self,
        min_component_area: int = 50,
        morphological_kernel: int = 3,
        gaussian_blur_kernel: int = 5,
        gaussian_blur_sigma: float = 1.5,
        use_smoothing: bool = True,
        use_morphology: bool = True,
        use_area_filter: bool = True,
        uncertainty_margin: float = 0.08,
    ):
        self.min_component_area = min_component_area
        self.morphological_kernel = morphological_kernel
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.use_smoothing = use_smoothing
        self.use_morphology = use_morphology
        self.use_area_filter = use_area_filter
        self.uncertainty_margin = uncertainty_margin

    def smooth_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Applies spatial Gaussian smoothing to suppress pixel-level outlier noise."""
        if not self.use_smoothing or self.gaussian_blur_kernel <= 0:
            return anomaly_map.copy()

        k = self.gaussian_blur_kernel
        if k % 2 == 0:
            k += 1  # Kernel dimension must be odd
        smoothed = cv2.GaussianBlur(anomaly_map, (k, k), self.gaussian_blur_sigma)
        return smoothed

    def apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological opening (removes small isolated noise spots)
        followed by morphological closing (bridges fragmented defect contours).
        """
        if not self.use_morphology or self.morphological_kernel <= 0:
            return binary_mask.copy()

        k = self.morphological_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def filter_components(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Filters out connected components smaller than `min_component_area`.
        Returns:
          cleaned_mask: uint8 array [0, 255]
          retained_components: list of dicts with bounding boxes and area
        """
        if not self.use_area_filter or self.min_component_area <= 0:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            comps = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                comps.append({"bbox": (x, y, x + w, y + h), "area": int(cv2.contourArea(cnt))})
            return binary_mask.copy(), comps

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        cleaned_mask = np.zeros_like(binary_mask)
        retained = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_component_area:
                cleaned_mask[labels == i] = 255
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                retained.append({
                    "bbox": (int(x), int(y), int(x + w), int(y + h)),
                    "area": int(area),
                    "centroid": (float(centroids[i][0]), float(centroids[i][1]))
                })

        return cleaned_mask, retained

    def process(
        self,
        raw_anomaly_map: np.ndarray,
        threshold: float,
        image_score: float,
        quality_report: Optional[Dict[str, Any]] = None,
        supervised_detected: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executes complete false-alarm reduction pipeline:
        1. Smooths anomaly map (optional).
        2. Applies thresholding to obtain raw binary defect mask.
        3. Applies morphological filtering (opening + closing).
        4. Filters connected components by minimum area.
        5. Computes inspection decision based on calibrated thresholds and quality checks.
        """
        # Step 1: Smoothing
        smoothed_map = self.smooth_anomaly_map(raw_anomaly_map)

        # Step 2: Thresholding
        raw_binary = (smoothed_map >= threshold).astype(np.uint8) * 255

        # Step 3: Morphology
        morph_binary = self.apply_morphology(raw_binary)

        # Step 4: Component Area Filtering
        filtered_binary, retained_components = self.filter_components(morph_binary)

        # Metrics before and after filtering
        raw_defect_pixels = int(np.sum(raw_binary > 0))
        filtered_defect_pixels = int(np.sum(filtered_binary > 0))
        suppressed_pixels = max(0, raw_defect_pixels - filtered_defect_pixels)

        # Step 5: Decision Derivation
        # Margin around threshold for manual review
        lower_bound = max(0.0, threshold - self.uncertainty_margin)
        upper_bound = threshold + self.uncertainty_margin

        has_defect_blobs = len(retained_components) > 0

        # Quality check impact
        is_bad_quality = quality_report and not quality_report.get("is_acceptable", True)

        if is_bad_quality:
            decision = "MANUAL REVIEW REQUIRED"
            rationale = "Degraded image quality (blur/lighting) impedes automated confidence."
        elif has_defect_blobs and image_score >= threshold:
            if supervised_detected is False:
                decision = "POSSIBLE UNKNOWN DEFECT"
                rationale = "Anomaly detected by feature distribution without matching known supervised class."
            else:
                decision = "DEFECT DETECTED"
                rationale = f"Anomaly score ({image_score:.3f}) exceeds calibrated threshold ({threshold:.3f}) with {len(retained_components)} validated component(s)."
        elif lower_bound <= image_score <= upper_bound:
            decision = "MANUAL REVIEW REQUIRED"
            rationale = f"Anomaly score ({image_score:.3f}) lies in borderline uncertainty band [{lower_bound:.3f}, {upper_bound:.3f}]."
        else:
            decision = "NORMAL"
            rationale = f"Surface satisfies normal inspection criteria (score {image_score:.3f} < threshold {threshold:.3f})."

        return {
            "smoothed_map": smoothed_map,
            "raw_binary_mask": raw_binary,
            "morph_binary_mask": morph_binary,
            "filtered_binary_mask": filtered_binary,
            "retained_components": retained_components,
            "raw_defect_pixels": raw_defect_pixels,
            "filtered_defect_pixels": filtered_defect_pixels,
            "suppressed_pixels": suppressed_pixels,
            "threshold": threshold,
            "image_score": image_score,
            "decision": decision,
            "rationale": rationale,
        }
