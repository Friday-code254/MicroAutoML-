"""
MicroAutoML-Agent — Attribution Engine
Isolates which specific change in a lineage actually yielded performance gains.
"""

from __future__ import annotations

from typing import Any

from automl.core.schemas import MetricDirection
from automl.search.tree import ExperimentTree, SearchNode


class AttributionEngine:
    """Computes edge-by-edge deltas for an experiment branch."""

    def __init__(self, metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> None:
        self.metric_direction = metric_direction

    def attribute_branch(self, leaf_id: str, tree: ExperimentTree) -> list[dict[str, Any]]:
        """
        Traces a node's lineage back to the root (baseline).
        Returns a list of attribution records (deltas).
        """
        history = tree.get_branch_history(leaf_id)
        
        attributions = []
        
        for i in range(1, len(history)):
            parent = history[i - 1]
            child = history[i]
            
            # Deltas
            score_delta = 0.0
            if parent.score is not None and child.score is not None:
                score_delta = child.score - parent.score
                
            runtime_delta = child.runtime_seconds - parent.runtime_seconds
            ram_delta = child.peak_ram_gb - parent.peak_ram_gb
            
            # Is improvement?
            is_improvement = False
            if self.metric_direction == MetricDirection.HIGHER_IS_BETTER:
                is_improvement = score_delta > 0
            else:
                is_improvement = score_delta < 0
                
            attr = {
                "parent_id": parent.experiment_id,
                "child_id": child.experiment_id,
                "operator": child.operator,
                "score_delta": score_delta,
                "runtime_delta": runtime_delta,
                "ram_delta": ram_delta,
                "is_improvement": is_improvement,
                "hypothesis": child.hypothesis_summary
            }
            attributions.append(attr)
            
        return attributions
