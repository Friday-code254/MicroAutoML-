"""
MicroAutoML-Agent — Resource Governor

Estimates memory and compute requirements for an experiment and enforces
hard caps, returning a ResourceDecision (e.g. ALLOWED_DEGRADED).
"""

from __future__ import annotations

import os
import psutil
from dataclasses import dataclass

from automl.core.enums import ResourceDecision
from automl.core.schemas import ExperimentSpec, ResourceBudget


@dataclass
class ResourceProfile:
    """Estimated resource needs for an experiment."""
    estimated_memory_mb: float
    recommended_threads: int
    decision: ResourceDecision


class ResourceGovernor:
    """
    Evaluates whether an experiment can be run safely within the machine's constraints.
    """
    
    def __init__(self, max_memory_mb: float | None = None, max_threads: int | None = None) -> None:
        # If not provided, we use 80% of available system memory
        if max_memory_mb is None:
            self.max_memory_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.8
        else:
            self.max_memory_mb = max_memory_mb
            
        if max_threads is None:
            # Leave 1 core free for OS/Agent if possible
            total_cores = os.cpu_count() or 4
            self.max_threads = max(1, total_cores - 1)
        else:
            self.max_threads = max_threads

    def evaluate(self, spec: ExperimentSpec, num_rows: int, num_cols: int, dataset_mb: float | None = None) -> ResourceProfile:
        """
        Estimates the memory and threads for the pipeline.
        
        If dataset_mb is provided, it is used as the exact base footprint.
        Otherwise, it falls back to a naive heuristic (rows * cols * 8 bytes).
        Tree ensembles typically need 5-10x dataset memory during training.
        """
        base_dataset_mb = dataset_mb if dataset_mb is not None else (num_rows * num_cols * 8) / (1024 * 1024)
        
        # Memory multiplier based on model family
        # (This would ideally be looked up from ComponentRegistry, but we use heuristics)
        multiplier = 3.0
        if spec.model_name in ("RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier"):
            multiplier = 8.0  # Sklearn RF memory footprint is notoriously high
        elif spec.model_name in ("CatBoostClassifier", "CatBoostRegressor", "LGBMClassifier", "XGBClassifier"):
            multiplier = 5.0
            
        estimated_memory_mb = base_dataset_mb * multiplier
        
        # Decide
        if estimated_memory_mb > self.max_memory_mb:
            return ResourceProfile(
                estimated_memory_mb=estimated_memory_mb,
                recommended_threads=1,
                decision=ResourceDecision.DENIED_MEMORY
            )
            
        # Determine threads
        recommended_threads = self.max_threads
        decision = ResourceDecision.ALLOWED
        
        # Degrade threads if memory is tight but not exceeding max
        if estimated_memory_mb > self.max_memory_mb * 0.7:
            recommended_threads = max(1, self.max_threads // 2)
            decision = ResourceDecision.ALLOWED_DEGRADED
            
        return ResourceProfile(
            estimated_memory_mb=estimated_memory_mb,
            recommended_threads=recommended_threads,
            decision=decision
        )
