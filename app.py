"""
Automated Industrial Surface Defect Detection Dashboard.
Main application landing page and system architecture overview.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.database import InspectionDatabase
from src.dataset_loader import MVTecDatasetManager
from models.model_loader import ModelRegistry

# Streamlit Page Config
st.set_page_config(
    page_title="Industrial Defect Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Industrial CSS Styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pipeline-step {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 10px 16px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .badge-normal {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-defect {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    cfg = load_config()
    db = InspectionDatabase()
    stats = db.get_summary_stats()
    manager = MVTecDatasetManager(cfg["paths"]["dataset_root"])
    categories = manager.get_available_categories()
    checkpoints = ModelRegistry.get_available_checkpoints(cfg["paths"]["checkpoints_dir"])

    st.title("🏭 Automated Industrial Surface Defect Detection")
    st.markdown("#### Unsupervised Anomaly Detection, Localization & False-Alarm Reduction")

    st.markdown("---")

    # High-level System KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Inspected</div>
            <div class="metric-val">{stats['total_inspections']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Normal Accepted</div>
            <div class="metric-val" style="color: #34d399;">{stats['normal_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Defects Localized</div>
            <div class="metric-val" style="color: #f87171;">{stats['defect_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Manual Reviews / Borderline</div>
            <div class="metric-val" style="color: #fbbf24;">{stats['manual_review_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Architecture and System Capabilities
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("System Architecture & Inspection Pipeline")
        st.markdown("""
        ```mermaid
        graph TD
            A[Industrial Camera / Image] --> B[Image Quality & Preprocessing]
            B --> C{Model Architecture}
            C -->|Unsupervised| D[PatchCore Anomaly Detector]
            C -->|Supervised| E[ResNet18 / YOLO Defect Localizer]
            D --> F[Spatial Distance Memory Bank]
            F --> G[Pixel Anomaly Heatmap]
            G --> H[Validation Quantile Calibration]
            H --> I[Spatial & Component Filtering]
            I --> J{Inspection Decision}
            J -->|Score < Thresh| K[NORMAL ACCEPTED]
            J -->|Score > Thresh| L[DEFECT DETECTED]
            J -->|Borderline Score| M[MANUAL REVIEW]
            J -->|Novel Signature| N[UNKNOWN DEFECT ALERT]
        ```
        """)

        st.markdown(r"""
        ### Core Research Innovation
        1. **Normal-Only Representation Learning**: Learns nominal surface features using PatchCore without requiring defect samples.
        2. **Quantile False-Alarm Calibration**: Calibrates decision thresholds on validation normals to bound test false alarms ($FPR \le 5\%$).
        3. **Spatial Postprocessing Filter**: Eliminates high-frequency noise spikes via morphological cleaning and connected-component area bounds ($A \ge 50$ px).
        4. **Unseen Defect Generalization**: Successfully detects completely novel defect subtypes absent during model configuration.
        """)

    with right_col:
        st.subheader("System Status & Environment")
        st.info(f"**Default Category:** `{cfg['dataset']['default_category']}`")
        st.info(f"**Discovered Categories:** `{', '.join(categories) if categories else 'None (Generate demo in Dataset Explorer)'}`")
        st.info(f"**Active Checkpoints:** `{len(checkpoints['patchcore'])} PatchCore, {len(checkpoints['cnn_classifier'])} CNN, {len(checkpoints['yolo'])} YOLO`")
        st.info(f"**Target FPR Calibration:** `{cfg['calibration']['target_fpr']*100:.1f}%`")
        st.info(f"**Device Mode:** `{cfg['hardware']['device']}`")

        st.markdown("### Navigation Guide")
        st.markdown("""
        - **1 Overview:** Analytics, outcomes, and research comparison charts.
        - **2 Image Inspection:** Upload & inspect single images with interactive heatmaps.
        - **3 Batch Inspection:** High-throughput batch processing with CSV/ZIP export.
        - **4 Real-Time Inspection:** Webcam-based live defect monitoring.
        - **5 Dataset Explorer:** Browse MVTec AD samples, masks, and category distributions.
        - **6 Model Training:** Train PatchCore, CNN, or YOLO models directly from the UI.
        - **7 Model Evaluation:** Run benchmark metrics, ROC-AUC, PRO, and ablation tests.
        - **8 Inspection History:** SQLite audit log with search, filters, and review status.
        - **9 Reports:** Export formal PDF inspection certificates and JSON audit dossiers.
        """)


if __name__ == "__main__":
    main()
