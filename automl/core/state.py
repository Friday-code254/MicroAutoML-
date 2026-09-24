"""
MicroAutoML-Agent — Run State

`RunState` is the mutable singleton for a single AutoML run.
It holds all live objects that span multiple subsystems.

Design rules:
- Only one RunState per process.
- All subsystems receive RunState at construction; never create it inside a subsystem.
- FinalMetrics can be set exactly once (enforced here).
- SearchState is checkpointed to SQLite by the memory layer; RunState holds
  the in-memory copy.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from automl.core.config import RunConfig
from automl.core.exceptions import FinalTestAlreadyUsedError, SearchLockedError
from automl.core.schemas import (
    EvidencePack,
    ExperimentResult,
    ExperimentSpec,
    FinalMetrics,
    FinalModel,
    SearchState,
)

if TYPE_CHECKING:
    pass


class RunState:
    """
    Mutable container for all live run-level objects.

    Lifecycle:
        1. Created by the CLI / entry point.
        2. Passed to each subsystem constructor.
        3. Checkpointed to SQLite after every meaningful event.
        4. Destroyed when the run completes.
    """

    def __init__(self, config: RunConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.started_at: datetime = datetime.now()

        # Run output directory
        self.run_dir: Path = config.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Phase A output
        self.evidence: EvidencePack | None = None

        # Search state (synced to SQLite)
        self.search_state: SearchState = SearchState(
            run_id=run_id,
            budget_total_seconds=config.search.time_budget_seconds,
        )

        # Experiment registry (in-memory cache, authoritative copy in SQLite)
        self._specs: dict[str, ExperimentSpec] = {}
        self._results: dict[str, ExperimentResult] = {}

        # Final model (set exactly once)
        self._final_model: FinalModel | None = None
        self._final_metrics: FinalMetrics | None = None
        self._final_metrics_set: bool = False
        self._search_locked: bool = False

        # Hardware snapshot (for reproducibility artifacts)
        self.hardware_info: dict[str, object] = self._capture_hardware()

    # ------------------------------------------------------------------
    # Experiment registry
    # ------------------------------------------------------------------

    def register_spec(self, spec: ExperimentSpec) -> None:
        self._specs[spec.experiment_id] = spec

    def register_result(self, result: ExperimentResult, metric_direction: "MetricDirection | None" = None) -> None:
        from automl.core.enums import MetricDirection
        self._results[result.experiment_id] = result
        # Update search state bookkeeping
        if result.status.value in ("success", "early_stopped", "repaired"):
            eid = result.experiment_id
            if eid not in self.search_state.completed_experiments:
                self.search_state.completed_experiments.append(eid)
            # Update incumbent — direction-aware
            if result.cv_score_mean is not None:
                incumbent = self.search_state.incumbent_score
                # Resolve direction: prefer explicit arg, then evidence on state, then default higher-is-better
                direction = metric_direction
                if direction is None and self.evidence is not None and self.evidence.metric_direction is not None:
                    direction = self.evidence.metric_direction
                
                if incumbent is None:
                    is_better = True
                elif direction == MetricDirection.LOWER_IS_BETTER:
                    is_better = result.cv_score_mean < incumbent
                else:  # HIGHER_IS_BETTER or unknown
                    is_better = result.cv_score_mean > incumbent
                    
                if is_better:
                    self.search_state.incumbent_score = result.cv_score_mean
                    self.search_state.incumbent_score_std = result.cv_score_std
                    self.search_state.incumbent_experiment_id = result.experiment_id
        elif result.status == "failed":
            eid = result.experiment_id
            if eid not in self.search_state.failed_experiments:
                self.search_state.failed_experiments.append(eid)
        self.search_state.last_updated_at = datetime.now()

    def get_spec(self, experiment_id: str) -> ExperimentSpec | None:
        return self._specs.get(experiment_id)

    def get_result(self, experiment_id: str) -> ExperimentResult | None:
        return self._results.get(experiment_id)

    def all_results(self) -> list[ExperimentResult]:
        return list(self._results.values())

    def successful_results(self) -> list[ExperimentResult]:
        return [
            r for r in self._results.values()
            if r.cv_score_mean is not None
        ]

    # ------------------------------------------------------------------
    # Search lock
    # ------------------------------------------------------------------

    def lock_search(self) -> None:
        """Lock the search — no new experiments can be started after this."""
        self._search_locked = True

    @property
    def search_locked(self) -> bool:
        return self._search_locked

    def assert_search_open(self) -> None:
        if self._search_locked:
            raise SearchLockedError(
                "Search is locked. Cannot start new experiments."
            )

    # ------------------------------------------------------------------
    # Final model / metrics
    # ------------------------------------------------------------------

    def set_final_model(self, model: FinalModel) -> None:
        if self._final_model is not None:
            raise RuntimeError("Final model has already been set for this run.")
        self._final_model = model

    @property
    def final_model(self) -> FinalModel | None:
        return self._final_model

    def set_final_metrics(self, metrics: FinalMetrics) -> None:
        """Set final test-set metrics. Can only be called once."""
        if self._final_metrics_set:
            raise FinalTestAlreadyUsedError()
        self._final_metrics = metrics
        self._final_metrics_set = True
        if self._final_model is not None:
            self._final_model.final_metrics = metrics

    @property
    def final_metrics(self) -> FinalMetrics | None:
        return self._final_metrics

    @property
    def final_metrics_set(self) -> bool:
        return self._final_metrics_set

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    def update_elapsed(self, elapsed_seconds: float) -> None:
        self.search_state.budget_elapsed_seconds = elapsed_seconds

    @property
    def budget_remaining_seconds(self) -> float:
        return self.search_state.budget_remaining_seconds

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "phase": self.search_state.phase.value,
            "experiments_completed": len(self.search_state.completed_experiments),
            "experiments_failed": len(self.search_state.failed_experiments),
            "incumbent_score": self.search_state.incumbent_score,
            "budget_remaining_seconds": self.budget_remaining_seconds,
            "search_locked": self._search_locked,
            "final_metrics_set": self._final_metrics_set,
        }

    # ------------------------------------------------------------------
    # Hardware info
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_hardware() -> dict[str, object]:
        mem = psutil.virtual_memory()
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_gb": round(mem.total / 1e9, 2),
            "available_ram_gb": round(mem.available / 1e9, 2),
        }
