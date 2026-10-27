"""
Unit tests for SQLite Inspection History Database.
"""

from pathlib import Path
import tempfile
import pytest

from src.database import InspectionDatabase


def test_database_insert_and_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_history.db"
        db = InspectionDatabase(db_path=db_path)

        # Insert inspection record
        record_id = db.insert_inspection(
            image_name="test_part_01.png",
            model_type="patchcore",
            anomaly_score=0.82,
            threshold=0.50,
            decision="DEFECT DETECTED",
            latency_ms=45.2,
            overlay_path="/tmp/overlay.png",
            notes="Scratch defect"
        )
        assert record_id == 1

        # Query recent
        recent = db.get_recent_inspections(limit=10)
        assert len(recent) == 1
        assert recent[0]["decision"] == "DEFECT DETECTED"
        assert recent[0]["anomaly_score"] == 0.82

        # Summary stats
        stats = db.get_summary_stats()
        assert stats["total_inspections"] == 1
        assert stats["defect_count"] == 1


def test_database_search_and_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_history2.db"
        db = InspectionDatabase(db_path=db_path)

        id1 = db.insert_inspection("part_a.png", "patchcore", 0.12, 0.5, "NORMAL", 30.0)
        id2 = db.insert_inspection("part_b.png", "patchcore", 0.95, 0.5, "DEFECT DETECTED", 35.0)

        # Filter by decision
        normals = db.search_inspections(decision="NORMAL")
        assert len(normals) == 1
        assert normals[0]["image_name"] == "part_a.png"

        # Update review status
        updated = db.update_review_status(id2, "approved", notes="Confirmed scratch")
        assert updated is True

        recent = db.get_recent_inspections(limit=5)
        b_rec = [r for r in recent if r["id"] == id2][0]
        assert b_rec["review_status"] == "approved"
        assert b_rec["notes"] == "Confirmed scratch"
