"""
Threshold Calibration Module for Industrial Anomaly Detection.
Calibrates decision boundaries on normal validation data to control false-alarm rates.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ThresholdCalibrator:
    """
    Calibrates decision thresholds on validation images.
    Implements normal-quantile false-alarm rate targeting and F1-score optimization.
    """

    def __init__(self, target_fpr: float = 0.05, default_threshold: float = 0.5):
        self.target_fpr = target_fpr
        self.default_threshold = default_threshold
        self.calibrated_threshold: float = default_threshold
        self.calibration_metadata: Dict[str, Any] = {}

    def calibrate_on_normal_validation(
        self,
        validation_normal_scores: List[float],
        target_fpr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates threshold corresponding to the (1 - target_fpr) quantile of normal validation scores.
        For instance, with target_fpr=0.05, the threshold is placed at the 95th percentile of normal scores,
        theoretically limiting normal false alarms on the validation distribution to 5%.
        """
        if not validation_normal_scores:
            self.calibrated_threshold = self.default_threshold
            return {
                "threshold": self.default_threshold,
                "method": "fallback_default",
                "target_fpr": self.target_fpr,
                "notes": "No validation normal scores provided; using default threshold."
            }

        scores = np.array(validation_normal_scores, dtype=np.float32)
        target = target_fpr if target_fpr is not None else self.target_fpr
        target = max(0.001, min(0.50, target))  # Bound target FPR between 0.1% and 50%

        quantile_perc = (1.0 - target) * 100.0
        calibrated_thresh = float(np.percentile(scores, quantile_perc))

        # Empirical validation FPR achieved on the validation cohort
        empirical_val_fpr = float(np.mean(scores > calibrated_thresh))

        self.calibrated_threshold = calibrated_thresh
        self.calibration_metadata = {
            "threshold": calibrated_thresh,
            "method": "normal_quantile",
            "target_fpr": target,
            "quantile_percentile": quantile_perc,
            "empirical_val_fpr": empirical_val_fpr,
            "num_val_normal_samples": len(scores),
            "normal_score_mean": float(np.mean(scores)),
            "normal_score_std": float(np.std(scores)),
            "normal_score_max": float(np.max(scores)),
            "disclaimer": (
                "Validation FPR is an empirical calibration target derived from validation "
                "normals. Generalization to unseen operational test distributions may vary."
            )
        }
        return self.calibration_metadata

    def calibrate_optimal_f1(
        self,
        val_scores: List[float],
        val_labels: List[int],
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Determines the threshold that maximizes the F1-score across validation data.
        val_labels: 0 for normal, 1 for defect.
        """
        if len(val_scores) == 0 or len(val_labels) == 0:
            return {"threshold": self.default_threshold, "method": "fallback_default"}

        scores = np.array(val_scores)
        labels = np.array(val_labels)

        min_s, max_s = np.min(scores), np.max(scores)
        if min_s == max_s:
            return {"threshold": float(min_s), "f1": 1.0, "method": "uniform_scores"}

        candidates = np.linspace(min_s, max_s, num_steps)
        best_f1 = -1.0
        best_thresh = self.default_threshold

        for t in candidates:
            preds = (scores >= t).astype(int)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        self.calibrated_threshold = best_thresh
        self.calibration_metadata = {
            "threshold": best_thresh,
            "best_f1": best_f1,
            "method": "f1_optimal",
            "num_samples": len(scores)
        }
        return self.calibration_metadata
