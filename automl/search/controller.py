"""
MicroAutoML-Agent — Search Controller
The main autonomous search loop that connects the planner, selector, and runner.
"""

from __future__ import annotations

import logging
import time
import random
from typing import Any, TYPE_CHECKING

import pandas as pd

from automl.core.enums import SaturationState, FailureType, BranchStatus, Fidelity, OperatorType
from automl.core.schemas import EvidencePack, SearchState
from automl.execution.resource import ResourceGovernor
from automl.execution.runner import ExperimentRunner
from automl.planner.hypothesis import HypothesisGenerator
from automl.planner.selector import HypothesisSelector
from automl.search.saturation import SaturationDetector
from automl.search.strategies import SearchStrategist
from automl.search.tree import ExperimentTree
from automl.critic.critic import NumericalCritic
from automl.repair.classifier import FailureClassifier
from automl.repair.policies import RepairPolicies

if TYPE_CHECKING:
    from automl.application.run_context import RunContext

logger = logging.getLogger(__name__)


class SearchController:
    """The central autonomous loop orchestrating the search process."""

    def __init__(
        self,
        planner: HypothesisGenerator,
        selector: HypothesisSelector,
        runner: ExperimentRunner,
        governor: ResourceGovernor,
        detector: SaturationDetector,
        strategist: SearchStrategist,
    ) -> None:
        self.planner = planner
        self.selector = selector
        self.runner = runner
        self.governor = governor
        self.detector = detector
        self.strategist = strategist
        
        self.critic = NumericalCritic()
        self.failure_classifier = FailureClassifier()
        self.repair_policies = RepairPolicies()

    def search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        evidence: EvidencePack,
        base_model_name: str,
        base_hyperparameters: dict[str, Any],
        context: RunContext,
    ) -> None:
        """
        Executes the autonomous search loop until saturation or budget exhaustion.
        Assumes Phase 1 (Data Intelligence) and Phase 2 (Baselines) are already complete.
        """
        state = context.state
        tree = context.tree
        store = context.store
        config = context.config
        
        logger.info(f"Starting autonomous search loop for run {state.run_id}")
        
        # Start the clock for this session (if resumed, we pick up where we left off)
        session_start_time = time.time()
        initial_elapsed = state.budget_elapsed_seconds

        while state.saturation_state not in (SaturationState.SATURATED, SaturationState.BUDGET_EXHAUSTED):
            # Update budget clock
            state.budget_elapsed_seconds = initial_elapsed + (time.time() - session_start_time)
            
            # 1. Check Saturation
            state.saturation_state = self.detector.check_saturation(state, tree)
            if state.saturation_state in (SaturationState.SATURATED, SaturationState.BUDGET_EXHAUSTED):
                logger.info("Search saturated/locked. Stopping search loop.")
                break

            # 2. Adapt Exploration Ratio
            self._adapt_explore_ratio(state)

            # 2.5 Multi-Fidelity Promotion (Wave 6.2)
            promoted_specs = []
            promising_nodes = tree.get_promising_branches()
            for p_node in promising_nodes:
                if p_node.status == BranchStatus.PROMISING and p_node.fidelity != Fidelity.FULL:
                    old_spec = store.get_spec(p_node.experiment_id)
                    if old_spec:
                        new_spec = old_spec.model_copy(deep=True)
                        from automl.core.schemas import _new_id
                        new_spec.experiment_id = _new_id("EXP")
                        new_spec.parent_id = p_node.experiment_id
                        
                        if old_spec.fidelity == Fidelity.SMOKE:
                            new_spec.fidelity = Fidelity.CHEAP
                        elif old_spec.fidelity == Fidelity.CHEAP:
                            new_spec.fidelity = Fidelity.MEDIUM
                        elif old_spec.fidelity == Fidelity.MEDIUM:
                            new_spec.fidelity = Fidelity.FULL
                            
                        new_spec.operator = OperatorType.RETEST
                        new_spec.hypothesis = f"Promoting {old_spec.experiment_id} to {new_spec.fidelity.value} fidelity."
                        promoted_specs.append(new_spec)
                        # Mark old node as completed so we don't promote it again
                        p_node.status = BranchStatus.COMPLETED

            specs = promoted_specs

            # 3. Plan new hypotheses if no promotions
            if not specs:
                # Diversify Parent Selection
                # 60% incumbent, 20% second-best, 20% random active
                parent_id = "BASELINE"
                incumbent = tree.get_incumbent(evidence.metric_direction)
                
                if incumbent:
                    roll = random.random()
                    if roll < 0.6:
                        parent_id = incumbent.experiment_id
                    elif roll < 0.8:
                        promising = tree.get_promising_branches()
                        if len(promising) > 1:
                            parent_id = promising[1].experiment_id # Second best
                        else:
                            parent_id = incumbent.experiment_id
                    else:
                        active = tree.get_active_branches()
                        if active:
                            parent_id = random.choice(active).experiment_id
                        else:
                            parent_id = incumbent.experiment_id
                
                logger.info(f"Planning round {state.planning_round} (parent: {parent_id})")
                hyp_set = self.planner.generate(evidence, round_num=state.planning_round)
                hyp_set.planning_round = state.planning_round
                
                if not hyp_set.hypotheses:
                    logger.warning("Planner returned no hypotheses. Forcing saturation.")
                    state.saturation_state = SaturationState.SATURATED
                    break

                # 2.3 Pass Real History to Selector
                history_specs = [store.get_spec(eid) for eid in state.completed_experiments[-10:]]
                history = [s.model_dump() for s in history_specs if s]
                
                # 2.4 Select Multiple Candidates
                top_k = config.search.cheap_survivors
                
                # 4. Select
                specs = self.selector.select(
                    hypothesis_set=hyp_set,
                    top_k=top_k,
                    parent_id=parent_id,
                    base_model_name=base_model_name,
                    base_hyperparameters=base_hyperparameters,
                    history=history,
                    explore_ratio=state.explore_ratio
                )
                
                if not specs:
                    logger.warning("Selector returned no valid specs. Forcing saturation.")
                    state.saturation_state = SaturationState.SATURATED
                    break
                    
            for spec in specs:
                # Execution with Self-Healing loop
                max_repairs = config.search.max_repairs_per_experiment
                
                while True: # Repair loop
                    # Record Node in Tree
                    tree.insert_node(spec)
                    store.save_experiment_spec(state.run_id, context.dataset_id, spec)
                    
                    # 6. Execute
                    logger.info(f"Running experiment {spec.experiment_id}: {spec.hypothesis[:50]}...")
                    result = self.runner.run(spec, X_train, y_train, evidence)
                    
                    # 2.1 Wire Critic into Search Loop
                    critic_report = self.critic.analyze(result)
                    result.is_stable = critic_report.is_stable
                    result.is_overfitting = critic_report.is_overfitting
                    result.critic_flags = critic_report.critic_flags
                    
                    # 7. Update Tree and State
                    tree.update_node_result(result)
                    store.save_experiment_result(result)
                    self.strategist.evaluate_tree(tree)
                    
                    self._update_state(state, result, evidence)
                    
                    # 2.2 Self-Healing Loop
                    repair_count = getattr(spec, "repair_count", 0)
                    if result.status.value == "failed" and repair_count < max_repairs:
                        err_type = "Exception"
                        if result.error_traceback and "Exception:" in result.error_traceback:
                            err_type = result.error_traceback.split("Exception:")[0].split("\n")[-1].strip()
                            
                        failure_type = self.failure_classifier.classify(err_type, result.error_message or "", result.error_traceback or "")
                        
                        logger.warning(f"Experiment {spec.experiment_id} failed ({failure_type.value}). Attempting repair {repair_count+1}/{max_repairs}")
                        repaired_spec, action = self.repair_policies.apply_policy(spec, failure_type)
                        repaired_spec.repair_count = repair_count + 1
                        repaired_spec.parent_id = spec.experiment_id
                        
                        # Generate new ID for the repaired spec to avoid primary key conflicts
                        from automl.core.schemas import _new_id
                        repaired_spec.experiment_id = _new_id("EXP")
                        spec = repaired_spec # Loop again
                    else:
                        break # Success or out of repairs
                
            # Loop bookkeeping
            state.planning_round += 1
            
        logger.info(f"Search loop finished. Total experiments: {len(state.completed_experiments)}")

    def _adapt_explore_ratio(self, state: SearchState) -> None:
        """Adapts exploration vs exploitation based on search depth."""
        if state.planning_round < 5:
            state.explore_ratio = 0.5
        elif state.planning_round < 15:
            state.explore_ratio = 0.25
        else:
            state.explore_ratio = 0.1

    def _update_state(self, state: SearchState, result: Any, evidence: EvidencePack) -> None:
        """Update RunState with the latest result."""
        if result.status.value == "success":
            state.completed_experiments.append(result.experiment_id)
            
            if result.cv_score_mean is not None:
                # Update incumbent if better
                from automl.application.metric_utils import is_better
                
                is_better_score = False
                if state.incumbent_score is None:
                    is_better_score = True
                else:
                    is_better_score = is_better(
                        result.cv_score_mean, 
                        state.incumbent_score, 
                        evidence.metric_direction
                    )
                    
                if is_better_score:
                    state.incumbent_score = result.cv_score_mean
                    state.incumbent_score_std = result.cv_score_std
                    state.incumbent_experiment_id = result.experiment_id
                    state.consecutive_no_gain = 0
                else:
                    state.consecutive_no_gain += 1
        else:
            state.failed_experiments.append(result.experiment_id)
            state.consecutive_no_gain += 1

