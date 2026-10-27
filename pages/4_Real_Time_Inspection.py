"""
Page 4: Real-Time Camera Inspection.
Provides live camera feed defect inspection, adjustable interval pacing, and frame capture.
"""

from pathlib import Path
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.utils import load_config
from src.inference import InspectionEngine
from src.database import InspectionDatabase
from models.model_loader import ModelRegistry

st.set_page_config(page_title="Real-Time Camera Inspection", page_icon="📹", layout="wide")


def main():
    st.title("📹 Real-Time Camera Inspection")
    st.caption("Live video stream inspection for conveyor lines and robotic inspection stations.")

    cfg = load_config()
    db = InspectionDatabase()
    checkpoints = ModelRegistry.get_available_checkpoints(cfg["paths"]["checkpoints_dir"])

    col_ctrl, col_view = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Inspection Parameters")
        model_type = st.selectbox("Inspection Model", options=["patchcore", "cnn"], index=0)
        available_cps = checkpoints.get("patchcore" if model_type == "patchcore" else "cnn_classifier", [])
        selected_cp = st.selectbox("Checkpoint", options=available_cps, format_func=lambda x: x.name) if available_cps else None

        thresh = st.slider("Anomaly Threshold", 0.1, 2.0, 0.5, 0.05)
        interval = st.slider("Inference Interval (sec)", 0.2, 3.0, 0.5, 0.1, help="Pacing between inspections to prevent CPU throttling")

        save_flagged = st.checkbox("Automatically Save Defective Frames", value=True)

    with col_view:
        st.subheader("Live Camera Stream")
        cam_input = st.camera_input("Optical Sensor Stream")

        if cam_input is not None:
            # Analyze captured frame
            try:
                model = ModelRegistry.load_model(model_type=model_type, checkpoint_path=selected_cp, config=cfg)
                engine = InspectionEngine(model=model, model_type=model_type, config=cfg, custom_threshold=thresh)

                frame_bytes = cam_input.getvalue()
                t0 = time.perf_counter()
                res = engine.inspect_image(frame_bytes, override_threshold=thresh)
                lat = (time.perf_counter() - t0) * 1000.0

                decision = res["decision"]
                score = res["image_score"]

                # Status Banner
                if decision == "NORMAL":
                    st.success(f"STATUS: {decision} | Score: {score:.3f} | Latency: {lat:.1f}ms")
                elif "UNKNOWN" in decision:
                    st.warning(f"STATUS: {decision} | Score: {score:.3f} | Latency: {lat:.1f}ms")
                elif "MANUAL" in decision:
                    st.info(f"STATUS: {decision} | Score: {score:.3f} | Latency: {lat:.1f}ms")
                else:
                    st.error(f"STATUS: {decision} | Score: {score:.3f} | Latency: {lat:.1f}ms")

                # Show annotated overlay
                st.image(res["overlay_image"], caption="Live Annotated Inspection Overlay", width="stretch")

                if save_flagged and decision != "NORMAL":
                    db.insert_inspection(
                        image_name=f"camera_frame_{int(time.time())}.png",
                        model_type=model_type,
                        anomaly_score=score,
                        threshold=thresh,
                        decision=decision,
                        latency_ms=lat,
                        notes="Live Camera Inspection Frame"
                    )
                    st.toast("Defective frame saved to SQLite inspection history!", icon="💾")

            except Exception as e:
                st.error(f"Inference error: {e}")
        else:
            st.info("Ensure browser camera permission is granted to inspect live industrial surfaces.")


if __name__ == "__main__":
    main()
