# Presentation Slide Deck: Automated Industrial Surface Defect Detection

**Duration:** 20 Minutes  
**Audience:** Technical Leadership, Computer Vision Engineers, Manufacturing QC Managers  

---

### Slide 1: Title & Overview
- **Title:** Automated Industrial Surface Defect Detection Using Deep Learning and Unsupervised Anomaly Detection
- **Subtitle:** Overcoming Unseen Defect Blindspots & Eliminating False-Positive Alarms in Smart Manufacturing
- **Presenter:** Senior Computer Vision & Deep Learning Engineer

### Slide 2: The Industrial Challenge
- Manufacturing reality: Defect rates < 0.1%.
- The Supervised Dilemma: Models cannot classify defects they have never seen.
- High cost of false alarms: Uncalibrated models halt automated production lines with spurious alerts.

### Slide 3: Proposed Solution & Novel Pipeline
- **Representation**: PatchCore unsupervised feature embeddings extracted from multiscale CNN layers.
- **Calibration**: Quantile threshold bounding targeting $FPR \le 5\%$ using normal validation images.
- **Postprocessing**: Spatial Gaussian smoothing + connected-component area bounds ($A \ge 50$ px).
- **Decision Engine**: 4-state output (NORMAL, DEFECT DETECTED, UNKNOWN DEFECT, MANUAL REVIEW).

### Slide 4: System Architecture Flow
- Diagram: Input Image → Quality Validation → Feature Extraction → Memory Bank Vector Search → Anomaly Map → Threshold Calibration → Spatial Filtering → Dashboard & Audit Log.

### Slide 5: Experimental Benchmark Setup
- Benchmark: MVTec Anomaly Detection standard.
- Models Compared:
  1. Supervised CNN Baseline (ResNet18)
  2. Ultralytics YOLO Object Detector
  3. PatchCore Anomaly Detector
  4. Full Calibrated False-Alarm Reduction Pipeline

### Slide 6: Key Results - Unseen Defect Generalization
- Normal-only PatchCore achieved **100.0% recall** on completely held-out unseen defect types (`scratch`).
- False-positive rate on normal production tests: **0.0%**.

### Slide 7: Ablation Study - False Alarm Filtering
- Baseline Pixel IoU: `0.4264`
- Full Postprocessed Pipeline Pixel IoU: `0.4431` (+3.9% relative gain)
- Pixel ROC-AUC: `0.9745`

### Slide 8: Enterprise Web Dashboard Demo
- Live demo of the 9 Streamlit modules:
  - Single image inspection with heatmaps & PDF certificates.
  - High-throughput batch processing with packaged ZIP exports.
  - Optical webcam live conveyor inspection.
  - SQLite persistent audit trail.

### Slide 9: Scalability & Production Readiness
- Runs on standard CPU hardware (~65 ms per inspection, ~15.4 FPS).
- Fully decoupled backend architecture compatible with FastAPI and React.
- Complete automated test suite (17/17 pytest checks passing).

### Slide 10: Conclusion & Q&A
- Key Takeaway: Normal-only representation learning + statistical threshold calibration solves both the unseen defect dilemma and the false-alarm challenge.
- Open floor for discussion.
