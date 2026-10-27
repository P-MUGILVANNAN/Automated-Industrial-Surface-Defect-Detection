"""
Page 3: High-Throughput Batch Surface Inspection.
Processes multi-image batches, exports aggregated CSVs and packaged ZIP archives.
"""

import io
import time
import zipfile
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st

from src.utils import load_config
from src.inference import InspectionEngine
from src.database import InspectionDatabase
from src.reporting import ReportGenerator
from models.model_loader import ModelRegistry

st.set_page_config(page_title="Batch Inspection", page_icon="📦", layout="wide")


def main():
    st.title("📦 High-Throughput Batch Inspection")
    st.caption("Inspect production batches, export defect metrics, and package annotated ZIP archives.")

    cfg = load_config()
    db = InspectionDatabase()
    checkpoints = ModelRegistry.get_available_checkpoints(cfg["paths"]["checkpoints_dir"])

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Batch Images",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            help="Upload multiple industrial surface images for concurrent quality screening."
        )

    with col2:
        model_type = st.selectbox("Inspection Model", options=["patchcore", "cnn"], index=0)
        available_cps = checkpoints.get("patchcore" if model_type == "patchcore" else "cnn_classifier", [])
        selected_cp = st.selectbox("Checkpoint", options=available_cps, format_func=lambda x: x.name) if available_cps else None
        batch_thresh = st.slider("Anomaly Threshold", 0.1, 2.0, 0.5, 0.05)

    if uploaded_files:
        st.info(f"Loaded {len(uploaded_files)} image(s) ready for inspection.")

        if st.button("⚡ Execute Batch Quality Screening", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                model = ModelRegistry.load_model(model_type=model_type, checkpoint_path=selected_cp, config=cfg)
                engine = InspectionEngine(model=model, model_type=model_type, config=cfg, custom_threshold=batch_thresh)
            except Exception as e:
                st.error(f"Failed to initialize model: {e}")
                return

            records = []
            zip_buffer = io.BytesIO()

            t_start = time.perf_counter()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {file.name}")
                    try:
                        img_bytes = file.read()
                        res = engine.inspect_image(img_bytes, override_threshold=batch_thresh)

                        # Save to database
                        db.insert_inspection(
                            image_name=file.name,
                            model_type=model_type,
                            anomaly_score=res["image_score"],
                            threshold=res["threshold"],
                            decision=res["decision"],
                            latency_ms=res["latency_ms"],
                            notes="Batch Inspection"
                        )

                        # Encode annotated overlay for ZIP
                        overlay_bgr = cv2.cvtColor(res["overlay_image"], cv2.COLOR_RGB2BGR)
                        _, buf = cv2.imencode(".png", overlay_bgr)
                        zip_file.writestr(f"inspected_{file.name}", buf.tobytes())

                        records.append({
                            "Filename": file.name,
                            "Decision": res["decision"],
                            "Anomaly_Score": round(res["image_score"], 4),
                            "Threshold": round(res["threshold"], 4),
                            "Defect_Blobs": len(res["bounding_boxes"]),
                            "Latency_ms": round(res["latency_ms"], 1),
                            "Status": "Success"
                        })
                    except Exception as err:
                        records.append({
                            "Filename": file.name,
                            "Decision": "ERROR",
                            "Anomaly_Score": 0.0,
                            "Threshold": batch_thresh,
                            "Defect_Blobs": 0,
                            "Latency_ms": 0.0,
                            "Status": f"Failed: {str(err)}"
                        })

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                total_time = time.perf_counter() - t_start

                # Include CSV report inside the ZIP
                df_results = pd.DataFrame(records)
                csv_str = df_results.to_csv(index=False)
                zip_file.writestr("batch_inspection_summary.csv", csv_str)

            status_text.success(f"Batch completed in {total_time:.2f}s ({len(uploaded_files)/max(total_time, 0.001):.1f} FPS)!")

            # Results Display
            st.markdown("### Batch Results Summary")
            st.dataframe(df_results, width="stretch")

            # Download Buttons
            st.markdown("### Export Artifacts")
            d_col1, d_col2 = st.columns(2)

            d_col1.download_button(
                label="📥 Download CSV Inspection Report",
                data=csv_str,
                file_name="batch_report.csv",
                mime="text/csv"
            )

            zip_buffer.seek(0)
            d_col2.download_button(
                label="📦 Download Complete Package (Images + Report ZIP)",
                data=zip_buffer.getvalue(),
                file_name="inspected_batch_package.zip",
                mime="application/zip"
            )


if __name__ == "__main__":
    main()
