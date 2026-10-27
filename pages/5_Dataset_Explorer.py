"""
Page 5: MVTec Dataset Explorer and Data Distribution Analytics.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from src.utils import load_config
from src.dataset_loader import MVTecDatasetManager
from src.dataset_validation import DatasetValidator
from scripts.prepare_dataset import generate_synthetic_industrial_category

st.set_page_config(page_title="Dataset Explorer", page_icon="📁", layout="wide")


def main():
    st.title("📁 Industrial Dataset Explorer (MVTec AD)")
    st.caption("Inspect industrial product categories, ground-truth masks, and defect distributions.")

    cfg = load_config()
    ds_root = Path(cfg["paths"]["dataset_root"])
    validator = DatasetValidator(ds_root)
    val_report = validator.validate_dataset()

    # Generator utility
    with st.expander("🛠️ Synthetic Category Generator (Instant Testing)"):
        st.write("Generate a procedural industrial category formatted strictly to MVTec AD specifications:")
        g_c1, g_c2, g_c3 = st.columns([2, 1, 1])
        new_cat = g_c1.selectbox("Category to Generate", ["bottle", "metal_nut", "tile", "leather", "wood"])
        num_tr = g_c2.number_input("Train Normal", 10, 100, 25)
        num_te = g_c3.number_input("Test Defects", 5, 50, 15)

        if st.button("Generate Industrial Category", type="secondary"):
            with st.spinner(f"Synthesizing {new_cat}..."):
                generate_synthetic_industrial_category(
                    output_root=ds_root,
                    category=new_cat,
                    num_train=num_tr,
                    num_test_defective=num_te
                )
            st.success(f"Category '{new_cat}' generated successfully!")
            st.rerun()

    if not val_report["is_available"] or not val_report["categories"]:
        st.warning(val_report["message"])
        return

    categories = val_report["categories"]
    selected_category = st.selectbox("Select Industrial Category", options=categories, index=0)

    # Category Details
    cat_details = val_report["category_details"][selected_category]
    stats = cat_details.get("statistics", {})

    st.markdown("---")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Normals", stats.get("num_train_good", 0))
    c2.metric("Testing Normals", stats.get("num_test_good", 0))
    c3.metric("Defective Samples", stats.get("num_test_defective", 0))
    c4.metric("Defect Subtypes", len(stats.get("defect_types", [])))

    if cat_details.get("warnings"):
        st.info("ℹ️ " + " | ".join(cat_details["warnings"]))

    st.markdown("---")

    # Defect Distribution Chart
    col_chart, col_gallery = st.columns([1, 2])

    with col_chart:
        st.subheader("Defect Subtypes Distribution")
        dist = stats.get("defect_distribution", {})
        if dist:
            df_dist = pd.DataFrame([{"Defect Subtype": k, "Count": v} for k, v in dist.items()])
            fig = px.bar(
                df_dist,
                x="Defect Subtype",
                y="Count",
                color="Defect Subtype",
                title="Defect Classes Breakdown"
            )
            fig.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig, width="stretch")
        else:
            st.write("No defect subtypes present.")

    with col_gallery:
        st.subheader("Sample Gallery: Surface & Ground Truth")
        manager = MVTecDatasetManager(ds_root)
        files = manager.get_category_files(selected_category)

        tab_norm, tab_def = st.tabs(["Normal Samples", "Defects & Masks"])

        with tab_norm:
            norm_imgs = files["train_images"][:4]
            if norm_imgs:
                cols = st.columns(len(norm_imgs))
                for c, p in zip(cols, norm_imgs):
                    c.image(str(p), caption=p.name, width="stretch")
            else:
                st.write("No normal images.")

        with tab_def:
            defects = files["test_defects"][:4]
            if defects:
                for d in defects:
                    c_img, c_mask = st.columns(2)
                    c_img.image(str(d["image_path"]), caption=f"{d['defect_type']}: {d['image_path'].name}", width="stretch")
                    if d.get("mask_path") and Path(d["mask_path"]).exists():
                        c_mask.image(str(d["mask_path"]), caption="Ground Truth Mask", width="stretch")
                    else:
                        c_mask.info("No mask file.")


if __name__ == "__main__":
    main()
