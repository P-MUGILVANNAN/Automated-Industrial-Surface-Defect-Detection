"""
Ultralytics YOLO Defect Detection Wrapper Module.
Provides known defect localization, bounding-box parsing, and confidence scoring.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO


class YOLODefectDetector:
    """
    Supervised YOLO Object Detection module for localizing known surface defects.
    Produces bounding boxes, category labels, and detection confidence scores.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(self.model_path)

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs YOLO object detection on a single image.
        Returns:
          detections: list of dicts with box [x1, y1, x2, y2], conf, class_id, class_name
          annotated_image: numpy array with drawn boxes
          num_defects: int count of detected defects
        """
        conf_val = conf if conf is not None else self.conf_threshold
        iou_val = iou if iou is not None else self.iou_threshold

        results = self.model.predict(
            source=image_input,
            conf=conf_val,
            iou=iou_val,
            device=self.device,
            verbose=False
        )

        detections = []
        annotated_image = None

        if len(results) > 0:
            res = results[0]
            annotated_image = res.plot()  # BGR plotted image
            if annotated_image is not None and annotated_image.ndim == 3:
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                names = res.names

                for box, score, cls_id in zip(xyxy, confs, classes):
                    name = names.get(cls_id, f"defect_{cls_id}")
                    detections.append({
                        "box": [float(b) for b in box],  # [x1, y1, x2, y2]
                        "confidence": float(score),
                        "class_id": int(cls_id),
                        "class_name": name,
                    })

        return {
            "detections": detections,
            "num_defects": len(detections),
            "annotated_image": annotated_image,
        }

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Runs batch detection over a list of images."""
        return [self.predict(img, conf=conf) for img in image_inputs]
