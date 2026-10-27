"""
Page 6: Model Training Studio.
Trains PatchCore, CNN Classifier, and YOLO models with live training metrics.
"""

from pathlib import Path
import time
import torch
import streamlit as st
import plotly.express as px
import pandas as pd

from src.utils import load_config, get_device
from src.dataset_loader import MVTecDatasetManager
from src.train_anomaly_detector import train_patchcore
from src.train_classifier import train_cnn_classifier

st.set_page_config(page_title="Model Training", page_icon="⚙️", layout="wide")


def main():
    st.title("⚙️ Model Training Studio")
    st.caption("Train unsupervised anomaly representations or supervised defect classifiers.")

    cfg = load_config()
    manager = MVTecDatasetManager(cfg["paths"]["dataset_root"])
    categories = manager.get_available_categories()

    if not categories:
        st.warning("No dataset categories found. Please visit Page 5 (Dataset Explorer) to generate or download a dataset.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Training Parameters")
        model_type = st.selectbox("Model Architecture", ["PatchCore (Unsupervised Anomaly)", "CNN Classifier (Supervised)"])
        selected_cat = st.selectbox("Target Category", options=categories)
        device_opt = st.selectbox("Compute Device", ["auto", "cpu", "cuda"])

        if "PatchCore" in model_type:
            coreset_ratio = st.slider("Coreset Subsampling Ratio", 0.01, 0.50, 0.10, 0.01, help="Higher ratio increases memory bank capacity at the cost of inference latency.")
            backbone = st.selectbox("Feature Extractor Backbone", ["resnet18", "wide_resnet50_2"])
        else:
            epochs = st.slider("Training Epochs", 5, 50, 15, 5)
            lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)

    with col2:
        st.subheader("Device & Resource Check")
        dev = get_device(device_opt)
        st.info(f"Active Hardware Accelerator: **{dev.type.upper()}**")
        st.info(f"Target Category: **{selected_cat}**")
        if "PatchCore" in model_type:
            st.info("Training Strategy: **Normal-only Feature Memory Bank + Quantile Calibration**")
        else:
            st.info("Training Strategy: **Transfer Learning with CrossEntropy Defect Loss**")

        st.write("")
        if st.button("🚀 Start Model Training", type="primary"):
            status_box = st.empty()
            status_box.info("Training in progress... This may take a minute.")

            try:
                t0 = time.perf_counter()
                if "PatchCore" in model_type:
                    res = train_patchcore(
                        category=selected_cat,
                        coreset_ratio=coreset_ratio,
                        device=dev
                    )
                    dur = time.perf_counter() - t0
                    status_box.success(f"PatchCore training completed in {dur:.1f}s!")
                    st.metric("Calibrated Threshold", f"{res['calibrated_threshold']:.4f}")
                    st.metric("Memory Bank Size", f"{res['metadata']['coreset_size']} features")
                    st.info(f"Checkpoint saved to: `{res['checkpoint_path']}`")
                else:
                    res = train_cnn_classifier(
                        category=selected_cat,
                        epochs=epochs,
                        lr=lr,
                        device=dev
                    )
                    dur = time.perf_counter() - t0
                    status_box.success(f"CNN Classifier training completed in {dur:.1f}s!")
                    st.metric("Best Validation Accuracy", f"{res['best_val_acc']*100:.1f}%")
                    st.info(f"Checkpoint saved to: `{res['checkpoint_path']}`")

                    # Plot training history
                    hist = res["history"]
                    df_hist = pd.DataFrame({
                        "Epoch": range(1, len(hist["train_loss"]) + 1),
                        "Train Loss": hist["train_loss"],
                        "Val Loss": hist["val_loss"],
                        "Val Accuracy": hist["val_acc"]
                    })
                    fig_loss = px.line(df_hist, x="Epoch", y=["Train Loss", "Val Loss"], title="Loss Curves")
                    st.plotly_chart(fig_loss, width="stretch")

            except Exception as e:
                status_box.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()
