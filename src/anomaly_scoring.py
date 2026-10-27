"""
Anomaly Scoring and Metric Aggregation Module.
Extracts holistic image-level, statistical, and spatial indicators from anomaly maps.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class AnomalyScorer:
    """
    Computes statistical and structural anomaly indicators from raw and normalized anomaly maps.
    """

    @staticmethod
    def compute_metrics(
        anomaly_map: np.ndarray,
        threshold: float = 0.5,
        percentile: float = 99.0
    ) -> Dict[str, Any]:
        """
        Calculates:
          - image_anomaly_score: maximum anomaly score
          - percentile_score: high-percentile score (robust to single-pixel outliers)
          - mean_anomaly_score: average anomaly score across surface
          - anomalous_pixel_ratio: percentage of pixels >= threshold
          - largest_component_area: pixel count of largest continuous defect region
          - num_components: count of disconnected defect regions
        """
        # Statistical scores
        max_score = float(np.max(anomaly_map))
        perc_score = float(np.percentile(anomaly_map, percentile))
        mean_score = float(np.mean(anomaly_map))

        # Binary thresholding for structural analysis
        binary_mask = (anomaly_map >= threshold).astype(np.uint8) * 255
        total_pixels = anomaly_map.size
        anomalous_pixels = int(np.sum(binary_mask > 0))
        anomalous_ratio = float(anomalous_pixels / total_pixels)

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # Component 0 is background; subsequent components are defect blobs
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = int(np.max(areas))
            num_components = int(num_labels - 1)
        else:
            largest_area = 0
            num_components = 0

        return {
            "max_score": max_score,
            "percentile_score": perc_score,
            "mean_score": mean_score,
            "anomalous_pixel_ratio": anomalous_ratio,
            "anomalous_pixel_count": anomalous_pixels,
            "largest_component_area": largest_area,
            "num_components": num_components,
            "binary_mask": binary_mask,
        }
