"""
MicroAutoML-Agent — Run Context

A single container that holds all mutable run state. Every subsystem receives
a reference to this, ensuring no duplicate trees or states exist.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from automl.core.config import RunConfig
    from automl.core.schemas import EvidencePack, SearchState
    from automl.search.tree import ExperimentTree
    from automl.memory.experiments import ExperimentStore

logger = logging.getLogger(__name__)

class RunContext:
    """
    Central repository for all mutable state in a MicroAutoML run.
    Passed to subsystems to prevent state desynchronization.
    """
    
    def __init__(
        self,
        config: "RunConfig",
        store: "ExperimentStore",
        evidence: "EvidencePack",
        state: "SearchState",
        tree: "ExperimentTree",
        dataset_id: str,
    ) -> None:
        self.config = config
        self.store = store
        self.evidence = evidence
        self.state = state
        self.tree = tree
        self.dataset_id = dataset_id
