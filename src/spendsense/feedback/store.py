"""SQLite local storage for user corrections and category feedback.

Ensures sensitive financial amounts are kept OUT of the learning store (Privacy Principle #8).
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Generator, List, Optional
from pydantic import BaseModel


class UserCorrection(BaseModel):
    """Schema for a persisted category correction."""
    transaction_id: str
    normalized_description: str
    original_category: str
    corrected_category: str
    model_version: str = "1.0.0"
    created_at: Optional[str] = None


class FeedbackStore:
    """Local SQLite repository for human-in-the-loop corrections."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = "data/spendsense.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides an SQLite connection and guarantees handle closure to prevent Windows file locks."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Creates the corrections table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS category_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL,
                    normalized_description TEXT NOT NULL,
                    original_category TEXT NOT NULL,
                    corrected_category TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(transaction_id, corrected_category)
                );
            """)
            conn.commit()

    def save_correction(
        self,
        transaction_id: str,
        normalized_description: str,
        original_category: str,
        corrected_category: str,
        model_version: str = "1.0.0",
    ) -> bool:
        """Persists a human correction locally with deduplication."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO category_corrections (
                        transaction_id, normalized_description, original_category,
                        corrected_category, model_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    transaction_id,
                    normalized_description.strip(),
                    original_category.strip(),
                    corrected_category.strip(),
                    model_version,
                    now_iso,
                ))
                conn.commit()
            return True
        except Exception:
            return False

    def get_all_corrections(self) -> List[UserCorrection]:
        """Retrieves all historical human corrections."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT transaction_id, normalized_description, original_category,
                       corrected_category, model_version, created_at
                FROM category_corrections
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        return [
            UserCorrection(
                transaction_id=row["transaction_id"],
                normalized_description=row["normalized_description"],
                original_category=row["original_category"],
                corrected_category=row["corrected_category"],
                model_version=row["model_version"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_corrections_count(self) -> int:
        """Returns total count of stored user corrections."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM category_corrections")
            return int(cursor.fetchone()[0])
