"""
Comprehensive Evaluation Metrics Module for Industrial Surface Inspection.
Computes Image-level, Pixel-level, Region-overlap (PRO), and Operational benchmarks.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import cv2


class EvaluationMetrics:
    """
    Standard computer vision evaluation suite for industrial defect detection.
    """

    @staticmethod
    def compute_image_metrics(
        y_true: List[int],
        y_scores: List[float],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Computes image-level classification and anomaly metrics.
        y_true: 0 for normal, 1 for defective.
        """
        if len(y_true) == 0:
            return {}

        y_true_arr = np.array(y_true, dtype=int)
        y_scores_arr = np.array(y_scores, dtype=np.float32)
        y_pred = (y_scores_arr >= threshold).astype(int)

        # ROC-AUC & AP require both classes to be present
        has_both_classes = len(np.unique(y_true_arr)) > 1
        if has_both_classes:
            roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
            ap = float(average_precision_score(y_true_arr, y_scores_arr))
        else:
            roc_auc = 1.0 if np.all(y_true_arr == y_pred) else 0.5
            ap = 1.0 if np.all(y_true_arr == 1) else 0.0

        p, r, f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred, average="binary", zero_division=0
        )

        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return {
            "roc_auc": roc_auc,
            "average_precision": ap,
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "confusion_matrix": {
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            },
            "num_samples": len(y_true_arr),
            "threshold": threshold,
        }

    @staticmethod
    def compute_pixel_metrics(
        masks_true: List[np.ndarray],
        maps_pred: List[np.ndarray],
        threshold: float = 0.5,
        subsample_step: int = 4
    ) -> Dict[str, Any]:
        """
        Computes pixel-level localization metrics:
        Pixel ROC-AUC, Pixel AP, Mean IoU, Dice Coefficient, and Per-Region Overlap (PRO).
        Subsample step prevents out-of-memory when aggregating millions of pixels.
        """
        if len(masks_true) == 0:
            return {}

        flat_true = []
        flat_pred = []
        ious = []
        dices = []
        region_overlaps = []

        for gt_mask, pred_map in zip(masks_true, maps_pred):
            # Normalize gt mask to binary [0, 1]
            gt_bin = (gt_mask > 0).astype(np.uint8)
            pred_bin = (pred_map >= threshold).astype(np.uint8)

            # Subsampled pixels for global ROC-AUC and AP
            flat_true.append(gt_bin[::subsample_step, ::subsample_step].ravel())
            flat_pred.append(pred_map[::subsample_step, ::subsample_step].ravel())

            # IoU and Dice per image
            intersection = np.logical_and(gt_bin, pred_bin).sum()
            union = np.logical_or(gt_bin, pred_bin).sum()
            iou = (intersection / union) if union > 0 else (1.0 if gt_bin.sum() == 0 else 0.0)
            ious.append(iou)

            total_pos = gt_bin.sum() + pred_bin.sum()
            dice = (2.0 * intersection / total_pos) if total_pos > 0 else (1.0 if gt_bin.sum() == 0 else 0.0)
            dices.append(dice)

            # Per-Region Overlap (PRO) calculation on defective images
            if gt_bin.sum() > 0:
                num_labels, labels = cv2.connectedComponents(gt_bin)
                for lbl in range(1, num_labels):
                    region = (labels == lbl)
                    region_size = region.sum()
                    if region_size > 0:
                        overlap = np.logical_and(region, pred_bin).sum() / region_size
                        region_overlaps.append(overlap)

        y_true_all = np.concatenate(flat_true)
        y_pred_all = np.concatenate(flat_pred)

        has_both = len(np.unique(y_true_all)) > 1
        if has_both:
            pix_roc_auc = float(roc_auc_score(y_true_all, y_pred_all))
            pix_ap = float(average_precision_score(y_true_all, y_pred_all))
        else:
            pix_roc_auc = 1.0 if np.all(y_true_all == (y_pred_all >= threshold)) else 0.5
            pix_ap = 1.0 if np.all(y_true_all == 1) else 0.0

        mean_pro = float(np.mean(region_overlaps)) if region_overlaps else float(np.mean(ious))

        return {
            "pixel_roc_auc": pix_roc_auc,
            "pixel_ap": pix_ap,
            "mean_iou": float(np.mean(ious)),
            "mean_dice": float(np.mean(dices)),
            "per_region_overlap_pro": mean_pro,
        }

    @staticmethod
    def measure_operational_latency(
        infer_fn,
        sample_tensor,
        warmup_runs: int = 3,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Measures mean inference latency and throughput in images per second.
        """
        # Warmup
        for _ in range(warmup_runs):
            _ = infer_fn(sample_tensor)

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = infer_fn(sample_tensor)
            times.append(time.perf_counter() - t0)

        mean_sec = float(np.mean(times))
        std_sec = float(np.std(times))
        latency_ms = mean_sec * 1000.0
        fps = (1.0 / mean_sec) if mean_sec > 0 else 0.0

        return {
            "mean_latency_ms": latency_ms,
            "std_latency_ms": std_sec * 1000.0,
            "throughput_fps": fps,
        }
