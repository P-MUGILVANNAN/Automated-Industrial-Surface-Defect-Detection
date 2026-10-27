"""
Full Model Evaluation and Research Benchmark Script.
Evaluates PatchCore baseline, false-alarm reduction pipeline, and held-out unseen defect detection.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_runner import ExperimentRunner
from src.utils import setup_logger

logger = setup_logger("evaluate_all")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection and False-Alarm Reduction")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec Category")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    args = parser.parse_args()

    runner = ExperimentRunner(config_path=args.config)

    logger.info(f"=== RUNNING ABLATION STUDY FOR CATEGORY: {args.category} ===")
    df_ablation = runner.run_ablation_study(category=args.category)
    print("\n--- ABLATION RESULTS ---")
    print(df_ablation.to_string(index=False))

    logger.info(f"\n=== RUNNING HELD-OUT UNSEEN DEFECT EVALUATION ===")
    held_out_res = runner.evaluate_held_out_unseen_defects(category=args.category)
    print("\n--- HELD-OUT UNSEEN DEFECT RESULTS ---")
    for k, v in held_out_res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
