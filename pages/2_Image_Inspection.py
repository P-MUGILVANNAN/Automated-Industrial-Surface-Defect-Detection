"""
Page 2: Single Image Surface Inspection and Diagnostic Visualization.
"""

from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.utils import load_config
from src.inference import InspectionEngine
from src.database import InspectionDatabase
from src.reporting import ReportGenerator
from models.model_loader import ModelRegistry

st.set_page_config(page_title="Image Inspection", page_icon="🔬", layout="wide")


def main():
    st.title("🔬 Single Image Surface Defect Inspection")
    st.caption("Inspect industrial parts, localize defective regions, and detect unseen defects.")

    cfg = load_config()
    db = InspectionDatabase()
    checkpoints = ModelRegistry.get_available_checkpoints(cfg["paths"]["checkpoints_dir"])

    # Sidebar: Model and Threshold Controls
    st.sidebar.header("Inspection Configuration")

    model_type = st.sidebar.selectbox(
        "Select Model Type",
        options=["patchcore", "cnn", "yolo"],
        index=0,
        help="PatchCore is recommended for unseen defect detection without defect training data."
    )

    # Checkpoint selection
    available_cps = checkpoints.get(
        "patchcore" if model_type == "patchcore" else ("cnn_classifier" if model_type == "cnn" else "yolo"), []
    )
    if available_cps:
        selected_cp = st.sidebar.selectbox("Checkpoint", options=available_cps, format_func=lambda x: x.name)
    else:
        selected_cp = None
        st.sidebar.warning(f"No checkpoint found for {model_type}. Train a model first in Page 6 or use fresh model.")

    # Threshold & Postprocessing Sliders
    st.sidebar.subheader("Threshold & Filtering")
    custom_thresh = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=2.0,
        value=0.50,
        step=0.05,
        help="Higher values reduce false alarms; lower values increase defect sensitivity."
    )

    use_far = st.sidebar.checkbox("Enable False-Alarm Filter", value=True)
    min_comp_area = st.sidebar.slider("Min Component Area (px)", 10, 500, 50, 10) if use_far else 0
    smooth_k = st.sidebar.slider("Gaussian Blur Kernel", 1, 9, 5, 2) if use_far else 1

    # Image Upload
    uploaded_file = st.file_uploader(
        "Upload Industrial Part Image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supports high-resolution PNG, JPG, and industrial camera image formats."
    )

    # Sample selector option
    sample_dir = Path(cfg["paths"]["dataset_root"])
    sample_images = list(sample_dir.glob("*/*/*/*.png")) if sample_dir.exists() else []

    selected_sample = None
    if sample_images and uploaded_file is None:
        st.write("Or choose a sample from dataset:")
        sample_names = {f"{p.parent.name}/{p.name}": p for p in sample_images[:20]}
        chosen = st.selectbox("Sample Image", options=list(sample_names.keys()))
        selected_sample = sample_names[chosen]

    image_source = uploaded_file or selected_sample

    if image_source is not None:
        # Load image bytes/array
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            image_name = uploaded_file.name
        else:
            image_bytes = selected_sample
            image_name = selected_sample.name

        if st.button("🚀 Run Surface Inspection", type="primary"):
            with st.spinner("Analyzing surface morphology and computing anomaly embeddings..."):
                try:
                    # Instantiate model
                    model = ModelRegistry.load_model(
                        model_type=model_type,
                        checkpoint_path=selected_cp,
                        config=cfg
                    )

                    engine = InspectionEngine(
                        model=model,
                        model_type=model_type,
                        config=cfg,
                        custom_threshold=custom_thresh
                    )

                    # Override filter settings
                    engine.filter.use_smoothing = use_far
                    engine.filter.use_morphology = use_far
                    engine.filter.use_area_filter = use_far
                    engine.filter.min_component_area = min_comp_area
                    engine.filter.gaussian_blur_kernel = smooth_k

                    res = engine.inspect_image(image_bytes, override_threshold=custom_thresh)

                    # Display Decision Badge
                    decision = res["decision"]
                    score = res["image_score"]
                    thresh = res["threshold"]

                    st.markdown("---")

                    col_dec1, col_dec2, col_dec3 = st.columns([2, 1, 1])

                    with col_dec1:
                        if decision == "NORMAL":
                            st.success(f"### Outcome: {decision}")
                        elif "UNKNOWN" in decision:
                            st.warning(f"### Outcome: {decision}")
                        elif "MANUAL" in decision:
                            st.info(f"### Outcome: {decision}")
                        else:
                            st.error(f"### Outcome: {decision}")
                        st.write(f"**Rationale:** {res['rationale']}")

                    with col_dec2:
                        st.metric("Anomaly Score", f"{score:.4f}")
                        st.caption(f"Threshold: {thresh:.4f}")

                    with col_dec3:
                        st.metric("Inference Latency", f"{res['latency_ms']:.1f} ms")
                        st.caption(f"Detected Blobs: {len(res['bounding_boxes'])}")

                    # Visual Panels
                    t1, t2, t3, t4 = st.tabs(["Overlay Detection", "Heatmap Map", "Defect Mask", "Original Image"])

                    with t1:
                        st.image(res["overlay_image"], caption="Annotated Anomaly Overlay with Bounding Boxes", width="stretch")
                    with t2:
                        # Colormap visualization
                        st.image(res["anomaly_map"], caption="Spatial Anomaly Distribution", clamp=True, width="stretch")
                    with t3:
                        st.image(res["binary_mask"], caption="Filtered Binary Defect Mask", width="stretch")
                    with t4:
                        st.image(res["original_image"], caption="Raw Input Surface", width="stretch")

                    # Quality Assessment Report
                    qual = res.get("quality_report", {})
                    if qual.get("warnings"):
                        st.warning("⚠️ **Image Quality Advisory:** " + " | ".join(qual["warnings"]))

                    # Save and Persist
                    preds_dir = Path(cfg["paths"]["predictions_dir"])
                    preds_dir.mkdir(parents=True, exist_ok=True)
                    overlay_path = preds_dir / f"inspected_{image_name}"
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(res["overlay_image"], cv2.COLOR_RGB2BGR))

                    rec_id = db.insert_inspection(
                        image_name=image_name,
                        model_type=model_type,
                        anomaly_score=score,
                        threshold=thresh,
                        decision=decision,
                        latency_ms=res["latency_ms"],
                        overlay_path=str(overlay_path),
                        notes="UI Single Image Inspection"
                    )

                    # Export & Download Buttons
                    st.markdown("### Export Inspection Artifacts")
                    e_col1, e_col2 = st.columns(2)

                    report_gen = ReportGenerator()
                    pdf_path = report_gen.generate_pdf_report(
                        inspection_result=res,
                        image_path=overlay_path,
                        overlay_path=overlay_path
                    )

                    with open(pdf_path, "rb") as pdf_file:
                        e_col1.download_button(
                            label="📄 Download Official PDF Certificate",
                            data=pdf_file.read(),
                            file_name=f"certificate_{image_name.split('.')[0]}.pdf",
                            mime="application/pdf"
                        )

                    json_path = report_gen.generate_json_report(
                        {k: (str(v) if not isinstance(v, (int, float, bool, list, dict)) else v)
                         for k, v in res.items() if k not in ("original_image", "overlay_image", "anomaly_map", "binary_mask", "raw_binary_mask")}
                    )

                    with open(json_path, "r") as j_file:
                        e_col2.download_button(
                            label="📋 Download JSON Audit Report",
                            data=j_file.read(),
                            file_name=f"report_{image_name.split('.')[0]}.json",
                            mime="application/json"
                        )

                except Exception as e:
                    st.error(f"Inspection failed: {str(e)}")
                    st.exception(e)
    else:
        st.info("Upload an image or generate demo data to begin inspection.")


if __name__ == "__main__":
    main()
