"""
Experiment Runner and Ablation Study Framework.
Executes reproducible comparative evaluations, ablations, and held-out defect studies.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.dataset_loader import MVTecDatasetManager, MVTecEvaluationDataset
from src.evaluation import EvaluationMetrics
from src.postprocessing import FalseAlarmFilter
from src.threshold_calibration import ThresholdCalibrator
from models.patchcore_detector import PatchCoreDetector
from src.train_anomaly_detector import train_patchcore
from src.utils import setup_logger, load_config, get_device

logger = setup_logger("experiment_runner")


class ExperimentRunner:
    """
    Orchestrates comparative benchmarks and ablation studies.
    """

    def __init__(self, config_path: Union[str, Path] = "config/config.yaml"):
        self.config = load_config(config_path)
        self.device = get_device(self.config.get("hardware", {}).get("device", "auto"))
        self.dataset_root = Path(self.config["paths"]["dataset_root"])
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def run_ablation_study(
        self,
        category: str = "bottle",
        patchcore_checkpoint: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Conducts systematic ablation of false-alarm reduction components:
        1. Baseline (Raw score, fixed 0.5 threshold, no spatial filtering)
        2. + Calibrated Threshold (normal validation quantile)
        3. + Connected Component Area Filtering (min_area=50)
        4. + Gaussian Spatial Smoothing
        5. Full Pipeline (Calibrated + Area Filtered + Smoothed)
        """
        logger.info(f"Starting False-Alarm Reduction Ablation Study for category '{category}'...")

        # 1. Load or train PatchCore
        cp_path = patchcore_checkpoint or (Path(self.config["paths"]["checkpoints_dir"]) / f"{category}_patchcore.pkl")
        if not Path(cp_path).exists():
            logger.info(f"Checkpoint {cp_path} not found. Training PatchCore first...")
            train_patchcore(category=category, config_path="config/config.yaml")

        model = PatchCoreDetector.load_checkpoint(cp_path, device=self.device)

        # 2. Prepare test evaluation samples
        manager = MVTecDatasetManager(self.dataset_root)
        files = manager.get_category_files(category)

        test_samples = []
        for p in files["test_good_images"]:
            test_samples.append({
                "image_path": p,
                "label": 0,
                "defect_type": "good",
                "mask_path": None
            })
        for d in files["test_defects"]:
            test_samples.append({
                "image_path": d["image_path"],
                "label": 1,
                "defect_type": d["defect_type"],
                "mask_path": d.get("mask_path")
            })

        eval_dataset = MVTecEvaluationDataset(
            test_samples,
            image_size=tuple(self.config.get("dataset", {}).get("image_size", [256, 256]))
        )

        # 3. Infer raw maps and scores
        raw_maps = []
        raw_scores = []
        gt_masks = []
        gt_labels = []

        logger.info(f"Extracting test anomaly maps across {len(eval_dataset)} test images...")
        with torch.no_grad():
            for item in eval_dataset:
                tensor = item["tensor"].unsqueeze(0).to(self.device)
                m, s = model.predict_anomaly_map(tensor)
                raw_maps.append(m)
                raw_scores.append(s)
                gt_masks.append(item["mask"])
                gt_labels.append(item["label"])

        calibrated_thresh = float(model.calibrated_threshold)
        fixed_thresh = 0.5

        # Define ablation configurations
        configurations = {
            "1_Raw_Baseline": {
                "thresh": fixed_thresh,
                "smooth": False,
                "morph": False,
                "area_filter": False,
            },
            "2_Threshold_Calibrated": {
                "thresh": calibrated_thresh,
                "smooth": False,
                "morph": False,
                "area_filter": False,
            },
            "3_Component_Area_Filtered": {
                "thresh": fixed_thresh,
                "smooth": False,
                "morph": True,
                "area_filter": True,
            },
            "4_Gaussian_Smoothed": {
                "thresh": fixed_thresh,
                "smooth": True,
                "morph": False,
                "area_filter": False,
            },
            "5_Full_False_Alarm_Pipeline": {
                "thresh": calibrated_thresh,
                "smooth": True,
                "morph": True,
                "area_filter": True,
            },
        }

        records = []
        for name, cfg_item in configurations.items():
            filter_obj = FalseAlarmFilter(
                min_component_area=50 if cfg_item["area_filter"] else 0,
                morphological_kernel=3 if cfg_item["morph"] else 0,
                gaussian_blur_kernel=5 if cfg_item["smooth"] else 0,
                use_smoothing=cfg_item["smooth"],
                use_morphology=cfg_item["morph"],
                use_area_filter=cfg_item["area_filter"]
            )

            # Process predictions under this config
            processed_scores = []
            processed_masks = []

            for r_map, r_score in zip(raw_maps, raw_scores):
                res = filter_obj.process(r_map, threshold=cfg_item["thresh"], image_score=r_score)
                # If area filter removed all components, suppress score
                if cfg_item["area_filter"] and len(res["retained_components"]) == 0 and r_score < 0.8:
                    p_score = min(r_score, cfg_item["thresh"] * 0.9)
                else:
                    p_score = r_score

                processed_scores.append(p_score)
                processed_masks.append(res["filtered_binary_mask"])

            # Compute image metrics
            img_metrics = EvaluationMetrics.compute_image_metrics(
                y_true=gt_labels,
                y_scores=processed_scores,
                threshold=cfg_item["thresh"]
            )

            # Compute pixel metrics
            pix_metrics = EvaluationMetrics.compute_pixel_metrics(
                masks_true=gt_masks,
                maps_pred=raw_maps if not cfg_item["smooth"] else [
                    filter_obj.smooth_anomaly_map(m) for m in raw_maps
                ],
                threshold=cfg_item["thresh"]
            )

            rec = {
                "Configuration": name,
                "Threshold": cfg_item["thresh"],
                "Image_ROC_AUC": img_metrics.get("roc_auc", 0.0),
                "Image_F1": img_metrics.get("f1_score", 0.0),
                "Image_FPR": img_metrics.get("false_positive_rate", 0.0),
                "Image_FNR": img_metrics.get("false_negative_rate", 0.0),
                "Pixel_ROC_AUC": pix_metrics.get("pixel_roc_auc", 0.0),
                "Pixel_IoU": pix_metrics.get("mean_iou", 0.0),
                "Pixel_PRO": pix_metrics.get("per_region_overlap_pro", 0.0),
            }
            records.append(rec)

        df_ablation = pd.DataFrame(records)

        # Save to disk
        csv_path = self.metrics_dir / f"ablation_{category}.csv"
        json_path = self.metrics_dir / f"ablation_{category}.json"
        df_ablation.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)

        logger.info(f"Ablation study saved to {csv_path.resolve()}")
        return df_ablation

    def evaluate_held_out_unseen_defects(
        self,
        category: str = "bottle",
        held_out_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model performance on known vs completely held-out unseen defect types.
        """
        manager = MVTecDatasetManager(self.dataset_root)
        files = manager.get_category_files(category)
        defect_types = files["defect_types"]

        if len(defect_types) < 2:
            held_out = defect_types[0] if defect_types else "unseen"
            known_types = []
        else:
            held_out = held_out_type or defect_types[-1]
            known_types = [dt for dt in defect_types if dt != held_out]

        logger.info(f"Evaluating Held-Out Unseen Defect Protocol: Held-out = '{held_out}', Known = {known_types}")

        # Load PatchCore
        cp_path = Path(self.config["paths"]["checkpoints_dir"]) / f"{category}_patchcore.pkl"
        if not cp_path.exists():
            train_patchcore(category=category, config_path="config/config.yaml")
        model = PatchCoreDetector.load_checkpoint(cp_path, device=self.device)

        # Evaluate Normal images
        from src.preprocessing import PreprocessingPipeline
        preproc = PreprocessingPipeline(target_size=tuple(self.config.get("dataset", {}).get("image_size", [256, 256])))
        normal_scores = []
        for p in files["test_good_images"]:
            t = preproc.preprocess(p)["tensor"].to(self.device)
            _, score = model.predict_anomaly_map(t)
            normal_scores.append(score)

        # Evaluate Known Defect images
        known_scores = []
        for d in files["test_defects"]:
            if d["defect_type"] in known_types:
                t = preproc.preprocess(d["image_path"])["tensor"].to(self.device)
                _, score = model.predict_anomaly_map(t)
                known_scores.append(score)

        # Evaluate Held-Out Defect images
        held_out_scores = []
        for d in files["test_defects"]:
            if d["defect_type"] == held_out:
                t = preproc.preprocess(d["image_path"])["tensor"].to(self.device)
                _, score = model.predict_anomaly_map(t)
                held_out_scores.append(score)

        thresh = float(model.calibrated_threshold)

        # Calculate detection recall on known vs held-out
        known_recall = float(np.mean([s >= thresh for s in known_scores])) if known_scores else 1.0
        held_out_recall = float(np.mean([s >= thresh for s in held_out_scores])) if held_out_scores else 1.0
        normal_fpr = float(np.mean([s >= thresh for s in normal_scores])) if normal_scores else 0.0

        results = {
            "category": category,
            "held_out_defect_type": held_out,
            "known_defect_types": known_types,
            "calibrated_threshold": thresh,
            "normal_test_fpr": normal_fpr,
            "known_defects_recall": known_recall,
            "held_out_unseen_recall": held_out_recall,
            "num_normal_test": len(normal_scores),
            "num_known_defect_samples": len(known_scores),
            "num_held_out_samples": len(held_out_scores),
            "conclusion": (
                f"Normal-only PatchCore achieved {held_out_recall*100:.1f}% detection recall on unseen defect '{held_out}' "
                f"with a test false-alarm rate of {normal_fpr*100:.1f}% on normal products."
            )
        }

        # Save results
        out_json = self.metrics_dir / f"held_out_evaluation_{category}.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Held-out evaluation completed: {results['conclusion']}")
        return results
