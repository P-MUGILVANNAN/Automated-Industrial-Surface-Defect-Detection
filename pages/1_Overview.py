"""
Page 1: Overview Dashboard and Operational Analytics.
"""

from pathlib import Path
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.utils import load_config
from src.database import InspectionDatabase
from models.model_loader import ModelRegistry

st.set_page_config(page_title="Inspection Overview", page_icon="📊", layout="wide")


def main():
    st.title("📊 Inspection Overview & Analytics")
    st.caption("Live Operational Status, Quality Metrics & Research Model Comparisons")

    cfg = load_config()
    db = InspectionDatabase()
    stats = db.get_summary_stats()
    checkpoints = ModelRegistry.get_available_checkpoints(cfg["paths"]["checkpoints_dir"])

    # Load evaluated metrics if available
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"ablation_{cfg['dataset']['default_category']}.json"
    held_out_path = Path(cfg["paths"]["metrics_dir"]) / f"held_out_evaluation_{cfg['dataset']['default_category']}.json"

    eval_data = None
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            eval_data = json.load(f)

    held_out_data = None
    if held_out_path.exists():
        with open(held_out_path, "r") as f:
            held_out_data = json.load(f)

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Inspected", stats["total_inspections"])
    c2.metric("Normal Approved", stats["normal_count"])
    c3.metric("Defects Flagged", stats["defect_count"])
    c4.metric("Manual Review", stats["manual_review_count"])

    benchmark_fpr = (held_out_data.get("normal_test_fpr", 0.0) * 100) if held_out_data else 0.0
    c5.metric("Evaluated Test FPR", f"{benchmark_fpr:.1f}%", help="Measured on held-out normal test set")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Inspection Decisions Distribution")
        decisions_dict = stats["decision_counts"]
        if decisions_dict and sum(decisions_dict.values()) > 0:
            df_dec = pd.DataFrame([
                {"Decision": k, "Count": v} for k, v in decisions_dict.items()
            ])
            fig_pie = px.pie(
                df_dec,
                names="Decision",
                values="Count",
                hole=0.45,
                color="Decision",
                color_discrete_map={
                    "NORMAL": "#10b981",
                    "DEFECT DETECTED": "#ef4444",
                    "POSSIBLE UNKNOWN DEFECT": "#f59e0b",
                    "MANUAL REVIEW REQUIRED": "#6366f1",
                }
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("No operational inspection records logged yet. Run an inspection in Page 2 or run demo.")

    with col_right:
        st.subheader("Model Comparison & Ablation Performance")
        if eval_data:
            df_ablation = pd.DataFrame(eval_data)
            fig_bar = px.bar(
                df_ablation,
                x="Configuration",
                y=["Pixel_IoU", "Pixel_ROC_AUC"],
                barmode="group",
                title="Pixel-Level Localization Accuracy across Pipeline Stages",
                color_discrete_sequence=["#38bdf8", "#818cf8"]
            )
            fig_bar.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("Benchmark evaluation data pending. Run evaluation in Page 7 to visualize ablation gains.")

    st.markdown("---")

    # Recent Inspection Table and Score Distribution
    b_col1, b_col2 = st.columns([3, 2])

    with b_col1:
        st.subheader("Recent Production Inspection Log")
        recent = db.get_recent_inspections(limit=10)
        if recent:
            df_recent = pd.DataFrame(recent)[
                ["id", "timestamp", "image_name", "model_type", "anomaly_score", "decision", "latency_ms"]
            ]
            st.dataframe(df_recent, width="stretch", height=260)
        else:
            st.info("Inspection table is empty.")

    with b_col2:
        st.subheader("Anomaly Score Distribution")
        if recent:
            scores = [r["anomaly_score"] for r in recent]
            fig_hist = px.histogram(
                scores,
                nbins=10,
                labels={"value": "Anomaly Score"},
                title="Recent Anomaly Scores",
                color_discrete_sequence=["#f43f5e"]
            )
            fig_hist.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=260, showlegend=False)
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info("Score distribution will appear once inspections are logged.")


if __name__ == "__main__":
    main()
