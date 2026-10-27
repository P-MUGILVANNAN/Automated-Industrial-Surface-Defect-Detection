"""
End-to-End Demo Script for Industrial Surface Inspection.
Inspects a sample image, derives defect localization, performs false-alarm reduction,
logs to SQLite database, and creates an official PDF inspection report.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from models.model_loader import ModelRegistry
from src.inference import InspectionEngine
from src.database import InspectionDatabase
from src.reporting import ReportGenerator
from src.utils import setup_logger, load_config

logger = setup_logger("run_demo")


def main():
    parser = argparse.ArgumentParser(description="Run Automated Inspection Demo")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--image", type=str, default=None, help="Path to image to inspect")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cp_dir = Path(cfg["paths"]["checkpoints_dir"])
    cp_path = cp_dir / f"{args.category}_patchcore.pkl"

    if not cp_path.exists():
        logger.error(f"Checkpoint {cp_path} not found. Please run: python -m src.train_anomaly_detector --category {args.category}")
        return

    # Select test image
    if args.image:
        test_img_path = Path(args.image)
    else:
        test_dir = Path(cfg["paths"]["dataset_root"]) / args.category / "test"
        defects = list(test_dir.glob("*/*.png"))
        if not defects:
            logger.error(f"No test images found in {test_dir}")
            return
        # Pick a scratch or defective image
        test_img_path = defects[0]
        for d in defects:
            if "good" not in str(d):
                test_img_path = d
                break

    logger.info(f"Loading PatchCore model from {cp_path}...")
    model = ModelRegistry.load_model("patchcore", checkpoint_path=cp_path, config=cfg)

    logger.info(f"Initializing Inspection Engine...")
    engine = InspectionEngine(model=model, model_type="patchcore", config=cfg)

    logger.info(f"Inspecting sample: {test_img_path}...")
    result = engine.inspect_image(test_img_path)

    print("\n" + "="*50)
    print("       INDUSTRIAL INSPECTION RESULT")
    print("="*50)
    print(f"File:             {test_img_path.name}")
    print(f"Decision:         {result['decision']}")
    print(f"Anomaly Score:    {result['image_score']:.4f}")
    print(f"Threshold:        {result['threshold']:.4f}")
    print(f"Inference Time:   {result['latency_ms']:.2f} ms")
    print(f"Defect Blobs:     {len(result['bounding_boxes'])}")
    print(f"Rationale:        {result['rationale']}")
    print("="*50 + "\n")

    # Save overlay image
    preds_dir = Path(cfg["paths"]["predictions_dir"])
    preds_dir.mkdir(parents=True, exist_ok=True)
    overlay_out = preds_dir / f"demo_overlay_{test_img_path.name}"
    cv2.imwrite(str(overlay_out), cv2.cvtColor(result["overlay_image"], cv2.COLOR_RGB2BGR))

    # Log to SQLite
    db = InspectionDatabase()
    rec_id = db.insert_inspection(
        image_name=test_img_path.name,
        model_type="patchcore",
        anomaly_score=result["image_score"],
        threshold=result["threshold"],
        decision=result["decision"],
        latency_ms=result["latency_ms"],
        overlay_path=str(overlay_out),
        notes="Automated Demo Run"
    )
    logger.info(f"Inspection recorded in SQLite history with ID: {rec_id}")

    # Generate PDF report
    report_gen = ReportGenerator()
    pdf_path = report_gen.generate_pdf_report(
        inspection_result=result,
        image_path=test_img_path,
        overlay_path=overlay_out
    )
    logger.info(f"Official PDF inspection report generated: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
