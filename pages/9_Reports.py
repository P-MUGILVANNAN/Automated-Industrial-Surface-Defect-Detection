"""
Page 9: Reports and Research Dossier Generator.
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st

from src.utils import load_config
from src.database import InspectionDatabase
from src.reporting import ReportGenerator

st.set_page_config(page_title="Inspection Reports", page_icon="📑", layout="wide")


def main():
    st.title("📑 Inspection Reports & Research Dossier")
    st.caption("Generate compliance certificates, export structured JSON dossiers, and review research formulations.")

    cfg = load_config()
    db = InspectionDatabase()
    stats = db.get_summary_stats()
    report_gen = ReportGenerator()

    # Load evaluated metrics if present
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / f"ablation_{cfg['dataset']['default_category']}.json"
    eval_metrics = []
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            eval_metrics = json.load(f)

    st.subheader("Generate Global Audit Dossier")
    st.write("Exports the complete system audit file combining operational inspection summaries, configuration parameters, and research metrics.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📦 Generate Complete JSON Audit Dossier", type="primary"):
            dossier_data = {
                "project": cfg["project"],
                "dataset_config": cfg["dataset"],
                "hardware_device": cfg["hardware"]["device"],
                "summary_stats": stats,
                "ablation_evaluation": eval_metrics,
                "recent_inspections": db.get_recent_inspections(limit=50)
            }
            json_path = report_gen.generate_json_report(dossier_data, filename="system_audit_dossier.json")

            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Download JSON Dossier",
                    data=f.read(),
                    file_name="system_audit_dossier.json",
                    mime="application/json"
                )
            st.success(f"Dossier prepared at: `{json_path.name}`")

    with col2:
        df_all = db.export_to_dataframe()
        csv_bytes = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Inspection Log (CSV)",
            data=csv_bytes,
            file_name="inspection_log.csv",
            mime="text/csv"
        )

    st.markdown("---")

    st.subheader("Research Formulation & Mathematical Summary")
    st.markdown(r"""
    ### 1. Patch-Level Feature Extraction
    Given an industrial image $x \in \mathbb{R}^{C \times H \times W}$, multiscale intermediate feature maps from ResNet layers $\ell \in \{2, 3\}$ are extracted, aggregated via $3 \times 3$ average pooling, and aligned:
    $$f(x, p) = \bigoplus_{\ell \in \mathcal{L}} \phi^{(\ell)}(x, p) \in \mathbb{R}^{D}$$

    ### 2. Normal Memory Bank Subsampling
    From nominal training images, the representative memory bank $\mathcal{M}^*$ is selected using greedy k-Center coreset selection:
    $$\mathcal{M}^* = \arg \min_{\mathcal{M} \subset \mathcal{X}, |\mathcal{M}|=K} \max_{x \in \mathcal{X}} \min_{m \in \mathcal{M}} \|x - m\|_2$$

    ### 3. Anomaly Scoring & Spatial Heatmap
    For each patch $p$ in query image $x_q$, the pixel anomaly score is determined by the Euclidean distance to its nearest neighbor in $\mathcal{M}^*$:
    $$s(x_q, p) = \min_{m \in \mathcal{M}^*} \|f(x_q, p) - m\|_2$$
    The image-level anomaly score is formulated as the maximum patch distance:
    $$S^*(x_q) = \max_p s(x_q, p)$$

    ### 4. Validation-Calibrated Decision Threshold
    To bound false-positive alarms on normal industrial surfaces, the threshold $\tau_\alpha$ is calibrated to the $(1 - \alpha)$ empirical quantile of validation normals:
    $$\tau_\alpha = \text{Quantile}_{1 - \alpha} \left( \{ S^*(x_{val, i}) \}_{i=1}^{N_{val}} \right), \quad \alpha = 0.05$$

    ### 5. Spatial False-Alarm Reduction
    The continuous anomaly map is smoothed via Gaussian convolution $G_\sigma * s$, thresholded at $\tau_\alpha$, and filtered to remove connected components with pixel area $A < A_{min} = 50$:
    $$\text{Mask}_{clean}(p) = \begin{cases} 1 & \text{if } (G_\sigma * s)(p) \ge \tau_\alpha \text{ and } \text{Area}(\mathcal{C}_p) \ge 50 \\ 0 & \text{otherwise} \end{cases}$$
    """)


if __name__ == "__main__":
    main()
