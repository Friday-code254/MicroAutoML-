"""
MicroAutoML-Agent — Search Strategies
Handles branch pruning and fidelity promotion logic for the search tree.
"""

from __future__ import annotations

from automl.core.enums import BranchStatus, Fidelity
from automl.core.schemas import MetricDirection
from automl.search.tree import ExperimentTree, SearchNode


class SearchStrategist:
    """Evaluates the search tree to prune dead ends and promote promising candidates."""

    def __init__(
        self,
        metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
        prune_threshold_pct: float = 0.05,
        promotion_threshold_pct: float = 0.02
    ) -> None:
        self.metric_direction = metric_direction
        self.prune_threshold_pct = prune_threshold_pct
        self.promotion_threshold_pct = promotion_threshold_pct

    def evaluate_tree(self, tree: ExperimentTree) -> None:
        """
        Evaluate the entire tree. Marks nodes as PRUNED or PROMISING.
        Should be called after every experiment completes.
        """
        incumbent = tree.get_incumbent(self.metric_direction)
        if not incumbent or incumbent.score is None:
            return  # Can't prune or promote without an incumbent

        # Evaluate all ACTIVE or COMPLETED nodes
        for node in tree.nodes.values():
            if node.status in (BranchStatus.ACTIVE, BranchStatus.FAILED, BranchStatus.PRUNED):
                continue
                
            if node.score is None:
                continue

            # 1. Check for Pruning
            if self._should_prune(node, incumbent):
                node.status = BranchStatus.PRUNED
                self._prune_children(tree, node)
                continue

            # 2. Check for Promotion
            if self._should_promote(node, incumbent):
                node.status = BranchStatus.PROMISING

    def _should_prune(self, node: SearchNode, incumbent: SearchNode) -> bool:
        """Determines if a node is hopelessly behind the incumbent."""
        if node.fidelity == Fidelity.FULL:
            return False  # Never prune a full fidelity run, it's a final candidate
            
        if node.score is None or incumbent.score is None:
            return False

        # Calculate percentage difference
        # Example: incumbent is 0.90. If prune_threshold is 0.05 (5%), threshold is 0.855
        if self.metric_direction == MetricDirection.HIGHER_IS_BETTER:
            threshold = incumbent.score * (1.0 - self.prune_threshold_pct)
            if node.score < threshold:
                return True
        else: # LOWER_IS_BETTER
            threshold = incumbent.score * (1.0 + self.prune_threshold_pct)
            if node.score > threshold:
                return True

        # Additionally, if this node is worse than its own parent by a large margin, prune it
        if node.score_delta is not None:
            if self.metric_direction == MetricDirection.HIGHER_IS_BETTER and node.score_delta < -0.05:
                return True
            if self.metric_direction == MetricDirection.LOWER_IS_BETTER and node.score_delta > 0.05:
                return True

        return False

    def _prune_children(self, tree: ExperimentTree, parent_node: SearchNode) -> None:
        """Recursively mark all descendants as PRUNED."""
        for child_id in parent_node.children:
            if child_id in tree.nodes:
                child = tree.nodes[child_id]
                child.status = BranchStatus.PRUNED
                self._prune_children(tree, child)

    def _should_promote(self, node: SearchNode, incumbent: SearchNode) -> bool:
        """Determines if a node is promising enough to advance to the next fidelity."""
        if node.status == BranchStatus.PROMISING:
            return False # Already promising
            
        if node.fidelity == Fidelity.FULL:
            return False # Can't promote beyond FULL
            
        if node.score is None or incumbent.score is None:
            return False

        # To be promoted, a lower-fidelity run must be close to the incumbent
        if self.metric_direction == MetricDirection.HIGHER_IS_BETTER:
            threshold = incumbent.score * (1.0 - self.promotion_threshold_pct)
            if node.score >= threshold:
                return True
        else:
            threshold = incumbent.score * (1.0 + self.promotion_threshold_pct)
            if node.score <= threshold:
                return True

        return False
