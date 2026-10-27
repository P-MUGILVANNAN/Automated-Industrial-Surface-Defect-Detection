"""
Reporting and Export Module.
Generates comprehensive inspection reports in CSV, JSON, and PDF formats using ReportLab.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

from src.utils import setup_logger, load_config

logger = setup_logger("reporting")


class ReportGenerator:
    """
    Generates industrial inspection reports and research summary dossiers.
    """

    def __init__(self, output_dir: Union[str, Path] = "reports/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(self, data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Exports inspection or evaluation summary as a structured JSON file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = filename or f"inspection_report_{ts}.json"
        out_p = self.output_dir / name
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON report generated at {out_p.resolve()}")
        return out_p

    def generate_csv_report(self, records: List[Dict[str, Any]], filename: Optional[str] = None) -> Path:
        """Exports tabular inspection batches as a CSV file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = filename or f"batch_report_{ts}.csv"
        out_p = self.output_dir / name
        df = pd.DataFrame(records)
        df.to_csv(out_p, index=False)
        logger.info(f"CSV report generated at {out_p.resolve()}")
        return out_p

    def generate_pdf_report(
        self,
        inspection_result: Dict[str, Any],
        image_path: Optional[Union[str, Path]] = None,
        overlay_path: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Generates a professional single-inspection PDF report with metadata,
        decision outcome, metrics table, and image figures.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = filename or f"inspection_certificate_{ts}.pdf"
        out_p = self.output_dir / name

        doc = SimpleDocTemplate(str(out_p), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=10
        )
        elements.append(Paragraph("Industrial Surface Defect Inspection Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        elements.append(Spacer(1, 15))

        # Outcome Banner Table
        decision = inspection_result.get("decision", "NORMAL")
        if decision == "NORMAL":
            banner_bg = colors.HexColor("#10b981")  # Emerald
        elif "UNKNOWN" in decision:
            banner_bg = colors.HexColor("#f59e0b")  # Amber
        elif "MANUAL" in decision:
            banner_bg = colors.HexColor("#6366f1")  # Indigo
        else:
            banner_bg = colors.HexColor("#ef4444")  # Red

        banner_text = Paragraph(
            f"<font color='white' size='14'><b>INSPECTION DECISION: {decision}</b></font>",
            styles["Normal"]
        )
        banner_table = Table([[banner_text]], colWidths=[540])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(banner_table)
        elements.append(Spacer(1, 15))

        # Metadata Table
        score = inspection_result.get("image_score", 0.0)
        thresh = inspection_result.get("threshold", 0.5)
        latency = inspection_result.get("latency_ms", 0.0)
        m_type = inspection_result.get("model_type", "PatchCore")

        meta_data = [
            ["Model Type:", str(m_type).upper(), "Anomaly Score:", f"{score:.4f}"],
            ["Decision Threshold:", f"{thresh:.4f}", "Inference Latency:", f"{latency:.2f} ms"],
            ["Quality Assessment:", "Acceptable" if inspection_result.get("quality_report", {}).get("is_acceptable", True) else "Warning", "Defect Regions:", str(len(inspection_result.get("bounding_boxes", [])))],
        ]

        t_meta = Table(meta_data, colWidths=[135, 135, 135, 135])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor("#f8fafc")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#334155")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTNAME', (3, 0), (3, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_meta)
        elements.append(Spacer(1, 15))

        # Decision Rationale
        rationale = inspection_result.get("rationale", "Automated inspection completed.")
        elements.append(Paragraph(f"<b>Decision Rationale:</b> {rationale}", styles["Normal"]))
        elements.append(Spacer(1, 15))

        # Image Visualizations
        img_cells = []
        if image_path and Path(image_path).exists():
            img_cells.append(RLImage(str(image_path), width=240, height=240))
        if overlay_path and Path(overlay_path).exists():
            img_cells.append(RLImage(str(overlay_path), width=240, height=240))

        if len(img_cells) == 2:
            img_table = Table([[Paragraph("<b>Original Surface</b>", styles["Normal"]), Paragraph("<b>Defect Anomaly Overlay</b>", styles["Normal"])], img_cells], colWidths=[270, 270])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(img_table)
        elif len(img_cells) == 1:
            elements.append(img_cells[0])

        doc.build(elements)
        logger.info(f"PDF report generated at {out_p.resolve()}")
        return out_p
