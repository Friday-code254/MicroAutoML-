"""
MicroAutoML-Agent — Lifecycle Manager

Handles the creation, persistence, and resumption of run state directories.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from automl.core.config import RunConfig
from automl.core.schemas import SearchState, EvidencePack, Event, EventType
from automl.memory.database import DatabaseManager

logger = logging.getLogger(__name__)


class RunLifecycle:
    """Manages the on-disk artifacts for a specific run."""

    def __init__(self, run_config: RunConfig):
        self.config = run_config
        
        # Ensure run_id is set
        if not self.config.run_id:
            from uuid import uuid4
            dataset_name = self.config.dataset_path.stem if self.config.dataset_path else "AutoML"
            self.config.run_id = f"RUN_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:4].upper()}"
            
        self.run_dir = self.config.output_dir / self.config.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.model_dir = self.run_dir / "model"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        
        self.db_path = self.run_dir / "database.sqlite"
        self.events_path = self.run_dir / "events.jsonl"
        self.config_path = self.run_dir / "config.yaml"
        self.evidence_path = self.run_dir / "evidence.json"
        
        self.db_manager: DatabaseManager | None = None

    def initialize_new_run(self) -> None:
        """Sets up a completely new run directory structure."""
        if self.run_dir.exists():
            logger.warning(f"Run directory {self.run_dir} already exists. Overwriting.")
            shutil.rmtree(self.run_dir)
            
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Save config
        self.config.to_yaml(self.config_path)
        
        # Initialize SQLite database
        self.db_manager = DatabaseManager(self.db_path)
        
        # Start event log
        self.log_event(Event(
            event_type=EventType.RUN_STARTED,
            run_id=self.config.run_id,
            message="Run initialized."
        ))

    def load_existing_run(self) -> tuple[RunConfig, SearchState, EvidencePack]:
        """Loads state from an existing run directory."""
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory {self.run_dir} does not exist.")
            
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database {self.db_path} not found.")
            
        # Initialize SQLite database
        self.db_manager = DatabaseManager(self.db_path)
        
        # Load evidence
        if not self.evidence_path.exists():
            raise FileNotFoundError(f"Evidence file {self.evidence_path} not found.")
            
        with open(self.evidence_path, "r") as f:
            evidence = EvidencePack.model_validate_json(f.read())
            
        # Load state from database
        from automl.memory.experiments import ExperimentStore
        store = ExperimentStore(self.db_manager)
        state = store.load_checkpoint(self.config.run_id)
        
        if state is None:
            raise RuntimeError(f"No checkpoint found in database for run {self.config.run_id}.")
            
        self.log_event(Event(
            event_type=EventType.RUN_STARTED,
            run_id=self.config.run_id,
            message="Run resumed from checkpoint."
        ))
            
        return self.config, state, evidence

    def save_evidence(self, evidence: EvidencePack) -> None:
        """Persists the evidence pack to disk."""
        with open(self.evidence_path, "w", encoding="utf-8") as f:
            f.write(evidence.model_dump_json(indent=2))

    def log_event(self, event: Event) -> None:
        """Appends an event to the JSONL log."""
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
            
    def get_db_manager(self) -> DatabaseManager:
        """Returns the initialized database manager."""
        if self.db_manager is None:
            self.db_manager = DatabaseManager(self.db_path)
        return self.db_manager

