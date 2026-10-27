"""
Page 8: Persistent Inspection History and Quality Audit Log.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.database import InspectionDatabase
from src.utils import load_config

st.set_page_config(page_title="Inspection History", page_icon="📜", layout="wide")


def main():
    st.title("📜 Inspection Audit History")
    st.caption("Permanent SQLite ledger of surface inspections, reviews, and operator annotations.")

    db = InspectionDatabase()

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])

    decision_filter = f_col1.selectbox(
        "Filter by Decision",
        ["ALL", "NORMAL", "DEFECT DETECTED", "POSSIBLE UNKNOWN DEFECT", "MANUAL REVIEW REQUIRED"]
    )

    model_filter = f_col2.selectbox("Filter by Model", ["ALL", "patchcore", "cnn", "yolo"])
    search_text = f_col3.text_input("Search (Filename or Notes)", "")

    records = db.search_inspections(
        decision=decision_filter,
        model_type=model_filter,
        search_query=search_text,
        limit=200
    )

    st.markdown("---")

    if records:
        df = pd.DataFrame(records)
        st.dataframe(
            df[["id", "timestamp", "image_name", "model_type", "anomaly_score", "threshold", "decision", "latency_ms", "review_status", "notes"]],
            width="stretch",
            height=350
        )

        st.markdown("### Operator Review & Annotation")
        r_c1, r_c2, r_c3, r_c4 = st.columns([1, 1, 2, 1])
        selected_id = r_c1.selectbox("Select Record ID", options=[r["id"] for r in records])
        new_status = r_c2.selectbox("Update Status", ["approved", "rejected", "re-inspect"])
        note_text = r_c3.text_input("Reviewer Notes", "Confirmed defect")

        if r_c4.button("Save Review", type="secondary"):
            db.update_review_status(selected_id, new_status, notes=note_text)
            st.success(f"Record #{selected_id} updated to '{new_status}'!")
            st.rerun()

        # CSV Export
        st.markdown("---")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Full History as CSV",
            data=csv_data,
            file_name="inspection_history_export.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching inspection records found in the audit database.")


if __name__ == "__main__":
    main()
