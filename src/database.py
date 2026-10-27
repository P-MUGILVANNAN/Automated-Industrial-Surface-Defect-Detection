"""
Inspection Database and SQLite Persistence Module.
Manages persistent inspection records, operational search, filtering, and export.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.utils import setup_logger, load_config

logger = setup_logger("database")


from contextlib import contextmanager

class InspectionDatabase:
    """
    Thread-safe SQLite database manager for industrial surface inspection history.
    Uses parameterized queries to prevent SQL injection and ensure reliable persistence.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            cfg = load_config()
            self.db_path = Path(cfg.get("paths", {}).get("database_path", "artifacts/inspection_history.db"))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yields SQLite connection with dict row access and ensures proper closure."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Creates inspections table and indexes if not already existing."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            threshold REAL NOT NULL,
            decision TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            overlay_path TEXT,
            notes TEXT,
            review_status TEXT DEFAULT 'pending'
        );
        """
        create_idx_sql = """
        CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_decision ON inspections(decision);
        CREATE INDEX IF NOT EXISTS idx_model ON inspections(model_type);
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.executescript(create_idx_sql)
            conn.commit()

    def insert_inspection(
        self,
        image_name: str,
        model_type: str,
        anomaly_score: float,
        threshold: float,
        decision: str,
        latency_ms: float,
        overlay_path: Optional[str] = None,
        notes: str = "",
        review_status: str = "pending"
    ) -> int:
        """
        Inserts an inspection record and returns the assigned ID.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        insert_sql = """
        INSERT INTO inspections (
            timestamp, image_name, model_type, anomaly_score, threshold,
            decision, latency_ms, overlay_path, notes, review_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                insert_sql,
                (
                    timestamp,
                    str(image_name),
                    str(model_type),
                    float(anomaly_score),
                    float(threshold),
                    str(decision),
                    float(latency_ms),
                    str(overlay_path) if overlay_path else None,
                    str(notes),
                    str(review_status),
                )
            )
            conn.commit()
            record_id = cursor.lastrowid

        return record_id

    def get_recent_inspections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent inspections sorted descending by ID/timestamp."""
        query = "SELECT * FROM inspections ORDER BY id DESC LIMIT ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_inspections(
        self,
        decision: Optional[str] = None,
        model_type: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Filters inspections based on criteria."""
        conditions = []
        params = []

        if decision and decision != "ALL":
            conditions.append("decision = ?")
            params.append(decision)

        if model_type and model_type != "ALL":
            conditions.append("model_type = ?")
            params.append(model_type)

        if search_query:
            conditions.append("(image_name LIKE ? OR notes LIKE ?)")
            wildcard = f"%{search_query}%"
            params.extend([wildcard, wildcard])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM inspections {where_clause} ORDER BY id DESC LIMIT ?;"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_review_status(self, record_id: int, new_status: str, notes: Optional[str] = None) -> bool:
        """Updates inspection review status ('approved', 'rejected', etc.)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if notes is not None:
                cursor.execute(
                    "UPDATE inspections SET review_status = ?, notes = ? WHERE id = ?;",
                    (new_status, notes, record_id)
                )
            else:
                cursor.execute(
                    "UPDATE inspections SET review_status = ? WHERE id = ?;",
                    (new_status, record_id)
                )
            conn.commit()
            return cursor.rowcount > 0

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates live operational counts and decision distributions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inspections;")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT decision, COUNT(*) FROM inspections GROUP BY decision;")
            decision_counts = dict(cursor.fetchall())

            cursor.execute("SELECT AVG(latency_ms) FROM inspections;")
            avg_lat = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT AVG(anomaly_score) FROM inspections;")
            avg_score = cursor.fetchone()[0] or 0.0

        return {
            "total_inspections": total,
            "decision_counts": decision_counts,
            "average_latency_ms": float(avg_lat),
            "average_anomaly_score": float(avg_score),
            "normal_count": decision_counts.get("NORMAL", 0),
            "defect_count": decision_counts.get("DEFECT DETECTED", 0),
            "unknown_defect_count": decision_counts.get("POSSIBLE UNKNOWN DEFECT", 0),
            "manual_review_count": decision_counts.get("MANUAL REVIEW REQUIRED", 0),
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Exports entire inspection history to a pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC;", conn)
        return df

    def export_to_csv(self, filepath: Union[str, Path]) -> Path:
        """Exports database records to a CSV file."""
        df = self.export_to_dataframe()
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        return out_p
