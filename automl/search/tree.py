"""
MicroAutoML-Agent — Search Tree
Manages the DAG of SearchNodes for tracking experiment lineage, pruning, and promotion.
"""

from __future__ import annotations

import json
from typing import Any

from automl.core.enums import BranchStatus, Fidelity, MetricDirection
from automl.core.schemas import ExperimentResult, ExperimentSpec, SearchNode
from automl.application.metric_utils import is_better


class ExperimentTree:
    """Manages the directed acyclic graph (DAG) of SearchNodes."""

    def __init__(self) -> None:
        self.nodes: dict[str, SearchNode] = {}
        self.root_nodes: list[str] = []

    def insert_node(self, spec: ExperimentSpec) -> SearchNode:
        """Create and insert a SearchNode based on a planned ExperimentSpec."""
        # Calculate depth
        depth = 0
        if spec.parent_id and spec.parent_id in self.nodes:
            parent = self.nodes[spec.parent_id]
            depth = parent.depth + 1
            
        node = SearchNode(
            experiment_id=spec.experiment_id,
            parent_id=spec.parent_id,
            operator=spec.operator,
            fidelity=spec.fidelity,
            status=BranchStatus.ACTIVE,
            hypothesis_summary=spec.hypothesis,
            depth=depth
        )
        
        self.nodes[node.experiment_id] = node
        
        if node.parent_id:
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(node.experiment_id)
        else:
            self.root_nodes.append(node.experiment_id)
            
        return node

    def update_node_result(self, result: ExperimentResult) -> None:
        """Update a node's scores and status based on its execution result."""
        if result.experiment_id not in self.nodes:
            return
            
        node = self.nodes[result.experiment_id]
        node.score = result.cv_score_mean
        node.score_std = result.cv_score_std
        node.runtime_seconds = result.resource_usage.runtime_seconds if result.resource_usage else 0.0
        node.peak_ram_gb = result.resource_usage.peak_ram_gb if result.resource_usage else 0.0
        
        if result.status == "failed" or result.status == "timeout":
            node.status = BranchStatus.FAILED
        elif result.status == "success":
            node.status = BranchStatus.COMPLETED
            # Calculate delta vs parent
            if node.parent_id and node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                if parent.score is not None and node.score is not None:
                    node.score_delta = node.score - parent.score

    def get_incumbent(self, direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> SearchNode | None:
        """Returns the completed node with the highest score at FULL fidelity."""
        candidates = [
            node for node in self.nodes.values() 
            if node.status in (BranchStatus.COMPLETED, BranchStatus.PROMISING) 
            and node.fidelity == Fidelity.FULL
            and node.score is not None
        ]
        if not candidates:
            # Fallback to lower fidelities if no full fidelity exists yet
            candidates = [
                node for node in self.nodes.values() 
                if node.status in (BranchStatus.COMPLETED, BranchStatus.PROMISING) 
                and node.score is not None
            ]
            if not candidates:
                return None
        
        # Sort candidates using the is_better function to find the max
        best = candidates[0]
        for candidate in candidates[1:]:
            if is_better(candidate.score, best.score, direction):
                best = candidate
        return best

    def get_promising_candidates(self, fidelity: Fidelity) -> list[SearchNode]:
        """Returns nodes currently at `fidelity` that are promising."""
        return [
            node for node in self.nodes.values()
            if node.fidelity == fidelity 
            and node.status == BranchStatus.PROMISING
        ]

    def get_promising_branches(self) -> list[SearchNode]:
        """Returns all promising branches sorted by score."""
        promising = [
            node for node in self.nodes.values()
            if node.status == BranchStatus.PROMISING and node.score is not None
        ]
        # Basic sort, in controller we only use it if it exists
        return sorted(promising, key=lambda x: x.score, reverse=True)

    def get_active_branches(self) -> list[SearchNode]:
        """Returns all currently active branches."""
        return [
            node for node in self.nodes.values()
            if node.status == BranchStatus.ACTIVE
        ]

    def get_branch_history(self, node_id: str) -> list[SearchNode]:
        """Returns the lineage of a node back to its root."""
        history = []
        current_id = node_id
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            history.append(node)
            current_id = node.parent_id
        return history[::-1]  # Root to leaf

    def to_json(self) -> str:
        """Serialize tree to JSON."""
        return json.dumps({
            k: v.model_dump(mode="json") 
            for k, v in self.nodes.items()
        })

    def load_json(self, data: str) -> None:
        """Deserialize tree from JSON."""
        raw = json.loads(data)
        self.nodes = {
            k: SearchNode.model_validate(v)
            for k, v in raw.items()
        }
        self.root_nodes = [
            k for k, v in self.nodes.items() 
            if v.parent_id is None
        ]
