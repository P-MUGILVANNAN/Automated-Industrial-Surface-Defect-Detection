"""
Unit tests for Report Generation (CSV, JSON, PDF).
"""

from pathlib import Path
import tempfile
import pytest

from src.reporting import ReportGenerator


def test_json_and_csv_reports():
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)

        # Test JSON
        sample_data = {"test_run": "unit_test", "metric": 0.99}
        json_path = generator.generate_json_report(sample_data, filename="test.json")
        assert json_path.exists()
        assert json_path.stat().st_size > 0

        # Test CSV
        sample_records = [{"id": 1, "score": 0.2}, {"id": 2, "score": 0.8}]
        csv_path = generator.generate_csv_report(sample_records, filename="test.csv")
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


def test_pdf_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)

        inspection_result = {
            "decision": "DEFECT DETECTED",
            "image_score": 0.875,
            "threshold": 0.50,
            "latency_ms": 42.1,
            "model_type": "patchcore",
            "rationale": "High anomaly score detected on surface.",
            "bounding_boxes": [{"box": [10, 10, 50, 50], "class_name": "scratch"}],
            "quality_report": {"is_acceptable": True}
        }

        pdf_path = generator.generate_pdf_report(inspection_result, filename="test_cert.pdf")
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 500  # Valid non-empty PDF
