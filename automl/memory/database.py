"""
MicroAutoML-Agent — Database Layer
Handles SQLite connections, schema initialization, and transaction management.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages the SQLite database connection and schema."""

    def __init__(self, db_path: str | Path = "microautoml.db") -> None:
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initializes the SQLite schema using WAL mode for concurrency safety."""
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrency and crash safety
            conn.execute("PRAGMA journal_mode=WAL;")
            
            # Datasets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    fingerprint_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Experiments (Specs & Results)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Failures
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Checkpoints (SearchState)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Lessons (Meta-Prior)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_hash TEXT NOT NULL,
                    hypothesis_hash TEXT NOT NULL,
                    is_improvement INTEGER NOT NULL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_run_id ON experiments(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lessons_hashes ON lessons(dataset_hash, hypothesis_hash);")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for yielding a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Enable foreign keys and set row factory
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
