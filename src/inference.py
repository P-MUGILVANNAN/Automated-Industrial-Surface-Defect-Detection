"""
Unified Inspection and Inference Engine.
Connects image validation, preprocessing, model inference, anomaly scoring,
false-alarm filtering, and decision synthesis.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.preprocessing import PreprocessingPipeline
from src.anomaly_scoring import AnomalyScorer
from src.postprocessing import FalseAlarmFilter
from src.threshold_calibration import ThresholdCalibrator
from src.visualization import Visualizer
from models.patchcore_detector import PatchCoreDetector
from models.cnn_classifier import CNNDefectClassifier
from models.yolo_detector import YOLODefectDetector
from src.utils import setup_logger, load_config, get_device

logger = setup_logger("inference_engine")


class InspectionEngine:
    """
    Production-grade inference and decision engine for industrial surface defect inspection.
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "patchcore",
        config: Optional[Dict[str, Any]] = None,
        custom_threshold: Optional[float] = None
    ):
        self.model = model
        self.model_type = model_type.lower()
        self.config = config or load_config()

        # Image Preprocessing Pipeline
        img_size = tuple(self.config.get("dataset", {}).get("image_size", [256, 256]))
        self.preprocessor = PreprocessingPipeline(
            target_size=img_size,
            blur_threshold=self.config.get("false_alarm_reduction", {}).get("quality_checks", {}).get("blur_threshold", 40.0),
            min_brightness=self.config.get("false_alarm_reduction", {}).get("quality_checks", {}).get("min_brightness", 25.0),
            max_brightness=self.config.get("false_alarm_reduction", {}).get("quality_checks", {}).get("max_brightness", 240.0),
            min_contrast=self.config.get("false_alarm_reduction", {}).get("quality_checks", {}).get("min_contrast", 20.0),
        )

        # Threshold setup
        if custom_threshold is not None:
            self.threshold = float(custom_threshold)
        elif hasattr(self.model, "calibrated_threshold"):
            self.threshold = float(self.model.calibrated_threshold)
        else:
            self.threshold = float(self.config.get("calibration", {}).get("default_fixed_threshold", 0.5))

        # False-Alarm Reduction Filter
        far_cfg = self.config.get("false_alarm_reduction", {})
        self.filter = FalseAlarmFilter(
            min_component_area=far_cfg.get("min_component_area", 50),
            morphological_kernel=far_cfg.get("morphological_kernel", 3),
            gaussian_blur_kernel=far_cfg.get("gaussian_blur_kernel", 5),
            gaussian_blur_sigma=far_cfg.get("gaussian_blur_sigma", 1.5),
            use_smoothing=far_cfg.get("enabled", True),
            use_morphology=far_cfg.get("enabled", True),
            use_area_filter=far_cfg.get("enabled", True),
            uncertainty_margin=self.config.get("inspection_decisions", {}).get("uncertainty_margin", 0.08)
        )

    def inspect_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        override_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs end-to-end industrial surface inspection on an image.
        """
        t0 = time.perf_counter()
        active_thresh = override_threshold if override_threshold is not None else self.threshold

        # Step 1: Preprocessing & Quality Validation
        prep_res = self.preprocessor.preprocess(image_input)
        orig_img = prep_res["original_image"]
        tensor = prep_res["tensor"]
        quality_report = prep_res["quality"]
        orig_shape = prep_res["original_shape"]

        # Step 2: Deep Learning Inference
        anomaly_map = None
        bounding_boxes = []
        class_name = None
        confidence = 0.0
        supervised_detected = None

        if self.model_type == "patchcore":
            anomaly_map, image_score = self.model.predict_anomaly_map(tensor)
            # Postprocessing & False-Alarm Reduction
            post_res = self.filter.process(
                raw_anomaly_map=anomaly_map,
                threshold=active_thresh,
                image_score=image_score,
                quality_report=quality_report,
                supervised_detected=supervised_detected
            )
            filtered_mask = post_res["filtered_binary_mask"]
            decision = post_res["decision"]
            rationale = post_res["rationale"]
            raw_mask = post_res["raw_binary_mask"]

            # Retained defect boxes
            for comp in post_res["retained_components"]:
                bounding_boxes.append({
                    "box": comp["bbox"],
                    "confidence": image_score,
                    "class_name": "anomaly",
                    "area": comp["area"]
                })

            scoring_metrics = AnomalyScorer.compute_metrics(
                anomaly_map=post_res["smoothed_map"],
                threshold=active_thresh
            )

        elif self.model_type in ("cnn", "cnn_classifier"):
            class_name, confidence, probs = self.model.predict(tensor)
            is_defect = (class_name.lower() != "normal") and (confidence >= active_thresh)
            image_score = confidence if is_defect else (1.0 - confidence)

            # Uniform pseudo-anomaly map for visualization
            anomaly_map = np.ones((tensor.shape[2], tensor.shape[3]), dtype=np.float32) * float(image_score)
            filtered_mask = (anomaly_map >= active_thresh).astype(np.uint8) * 255
            raw_mask = filtered_mask.copy()

            if is_defect:
                decision = "DEFECT DETECTED"
                rationale = f"CNN classified as '{class_name}' with {confidence*100:.1f}% confidence."
            else:
                decision = "NORMAL"
                rationale = f"CNN classified as normal with {probs[0]*100:.1f}% confidence."

            scoring_metrics = {
                "max_score": image_score,
                "percentile_score": image_score,
                "mean_score": image_score,
                "anomalous_pixel_ratio": 1.0 if is_defect else 0.0,
                "largest_component_area": int(filtered_mask.size) if is_defect else 0,
                "num_components": 1 if is_defect else 0,
            }

        elif self.model_type in ("yolo", "yolo_detector"):
            yolo_res = self.model.predict(orig_img, conf=active_thresh)
            dets = yolo_res["detections"]
            num_dets = len(dets)
            image_score = float(max([d["confidence"] for d in dets])) if num_dets > 0 else 0.0
            bounding_boxes = dets

            # Construct binary mask from bounding boxes
            h, w = orig_shape
            filtered_mask = np.zeros((h, w), dtype=np.uint8)
            for d in dets:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                filtered_mask[y1:y2, x1:x2] = 255
            raw_mask = filtered_mask.copy()
            anomaly_map = filtered_mask.astype(np.float32) / 255.0

            if num_dets > 0:
                decision = "DEFECT DETECTED"
                rationale = f"YOLO identified {num_dets} defect instance(s) of type: {', '.join([d['class_name'] for d in dets])}."
            else:
                decision = "NORMAL"
                rationale = "No defect bounding boxes detected above confidence threshold."

            scoring_metrics = {
                "max_score": image_score,
                "percentile_score": image_score,
                "mean_score": float(np.mean(anomaly_map)),
                "anomalous_pixel_ratio": float(np.sum(filtered_mask > 0) / filtered_mask.size),
                "largest_component_area": 0,
                "num_components": num_dets,
            }

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Scale visualizations back to original image shape if needed
        scaled_mask = PreprocessingPipeline.scale_mask_to_original(filtered_mask, orig_shape)
        scaled_heatmap = PreprocessingPipeline.scale_mask_to_original(
            (anomaly_map * 255).astype(np.uint8), orig_shape
        ).astype(np.float32) / 255.0

        # Create overlay visualization
        overlay_image = Visualizer.overlay_heatmap_on_image(orig_img, scaled_heatmap, alpha=0.5)
        if bounding_boxes:
            overlay_image = Visualizer.draw_boxes(overlay_image, bounding_boxes)

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "decision": decision,
            "rationale": rationale,
            "image_score": float(image_score),
            "threshold": float(active_thresh),
            "latency_ms": float(duration_ms),
            "original_image": orig_img,
            "overlay_image": overlay_image,
            "anomaly_map": scaled_heatmap,
            "binary_mask": scaled_mask,
            "raw_binary_mask": raw_mask,
            "bounding_boxes": bounding_boxes,
            "quality_report": quality_report,
            "scoring_metrics": scoring_metrics,
            "model_type": self.model_type,
        }
