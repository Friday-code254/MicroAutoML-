"""
MicroAutoML-Agent — Experiment Store
DAO for interacting with SQLite to save experiments, failures, and search states.
"""

from __future__ import annotations

import logging
from typing import Any

from automl.core.schemas import DatasetFingerprint, ExperimentResult, ExperimentSpec, FailureReport, SearchState
from automl.memory.database import DatabaseManager

logger = logging.getLogger(__name__)


class ExperimentStore:
    """Handles CRUD operations for experiments, failures, and search state."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def save_dataset_fingerprint(self, dataset_id: str, fingerprint: DatasetFingerprint) -> None:
        """Saves or updates a dataset fingerprint."""
        query = """
            INSERT INTO datasets (dataset_id, fingerprint_json)
            VALUES (?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                fingerprint_json=excluded.fingerprint_json
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (dataset_id, fingerprint.model_dump_json()))

    def save_experiment_spec(self, run_id: str, dataset_id: str, spec: ExperimentSpec) -> None:
        """Saves a newly planned experiment spec."""
        query = """
            INSERT INTO experiments (experiment_id, run_id, dataset_id, spec_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                spec_json=excluded.spec_json
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (spec.experiment_id, run_id, dataset_id, spec.model_dump_json()))

    def save_experiment_result(self, result: ExperimentResult) -> None:
        """Updates an existing experiment with its result."""
        query = """
            UPDATE experiments
            SET result_json = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE experiment_id = ?
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (result.model_dump_json(), result.experiment_id))

    def save_failure_report(self, run_id: str, report: FailureReport) -> None:
        """Saves a failure report."""
        query = """
            INSERT INTO failures (experiment_id, run_id, failure_type, report_json)
            VALUES (?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (
                report.experiment_id, 
                run_id, 
                report.failure_type.value, 
                report.model_dump_json()
            ))

    def save_checkpoint(self, dataset_id: str, state: SearchState) -> None:
        """Saves the complete SearchState to allow resume functionality."""
        query = """
            INSERT INTO checkpoints (run_id, dataset_id, state_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                state_json=excluded.state_json,
                updated_at=CURRENT_TIMESTAMP
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (state.run_id, dataset_id, state.model_dump_json()))

    def load_checkpoint(self, run_id: str) -> SearchState | None:
        """Loads a SearchState from the database to resume a run."""
        query = "SELECT state_json FROM checkpoints WHERE run_id = ?"
        with self.db.get_connection() as conn:
            row = conn.execute(query, (run_id,)).fetchone()
            if row:
                return SearchState.model_validate_json(row["state_json"])
        return None

    def get_spec(self, experiment_id: str) -> ExperimentSpec | None:
        """Retrieves a single experiment spec by its ID."""
        query = "SELECT spec_json FROM experiments WHERE experiment_id = ?"
        with self.db.get_connection() as conn:
            row = conn.execute(query, (experiment_id,)).fetchone()
            if row:
                return ExperimentSpec.model_validate_json(row["spec_json"])
        return None

    def get_all_specs(self, run_id: str) -> list[ExperimentSpec]:
        """Retrieves all experiment specs for a given run."""
        query = "SELECT spec_json FROM experiments WHERE run_id = ?"
        with self.db.get_connection() as conn:
            rows = conn.execute(query, (run_id,)).fetchall()
            return [ExperimentSpec.model_validate_json(row["spec_json"]) for row in rows]

    def get_all_results(self, run_id: str) -> list[ExperimentResult]:
        """Retrieves all non-null experiment results for a given run."""
        query = "SELECT result_json FROM experiments WHERE run_id = ? AND result_json IS NOT NULL"
        with self.db.get_connection() as conn:
            rows = conn.execute(query, (run_id,)).fetchall()
            return [ExperimentResult.model_validate_json(row["result_json"]) for row in rows]

