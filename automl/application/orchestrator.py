"""
MicroAutoML-Agent — Application Orchestrator

The single canonical lifecycle owner. Executes the full pipeline from Data Intelligence
to final Model Packaging, enforcing all invariants.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from automl.core.config import RunConfig
from automl.core.schemas import SearchState, EvidencePack, SearchPhase, SaturationState, Event, EventType
from automl.core.enums import OperatorType
from automl.application.lifecycle import RunLifecycle
from automl.application.run_context import RunContext
from automl.application.invariants import assert_metric_locked, assert_test_not_used
from automl.data.pipeline import DataIntelligencePipeline
from automl.metrics.selector import MetricSelector
from automl.models.baseline_hub import BaselineHub
from automl.execution.runner import ExperimentRunner
from automl.memory.experiments import ExperimentStore
from automl.search.tree import ExperimentTree
from automl.planner.hypothesis import HypothesisGenerator
from automl.planner.client import SLMClient
from automl.planner.selector import HypothesisSelector
from automl.execution.resource import ResourceGovernor
from automl.search.saturation import SaturationDetector
from automl.search.strategies import SearchStrategist
from automl.search.controller import SearchController
from automl.execution.final_eval import FinalHoldoutEvaluator
from automl.compiler.compiler import PipelineCompiler

logger = logging.getLogger(__name__)


class RunOrchestrator:
    """Owns the complete canonical lifecycle."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.lifecycle = RunLifecycle(config)

    def execute(self) -> float | None:
        """Executes the full pipeline for a new run and returns the final test score."""
        self.lifecycle.initialize_new_run()
        
        store = ExperimentStore(self.lifecycle.get_db_manager())
        
        state = SearchState(
            run_id=self.config.run_id,
            budget_total_seconds=self.config.search.time_budget_seconds,
            phase=SearchPhase.PROFILING
        )
        
        try:
            # 1. Data Intelligence
            evidence, df = self._run_data_intelligence()
            
            # 2. Metric Selection & LOCK
            self._lock_metric(evidence)
            self.lifecycle.save_evidence(evidence)
            
            X = df.drop(columns=[self.config.target_column])
            y = df[self.config.target_column]
            
            tree = ExperimentTree()
            dataset_id = "default_dataset_id"
            if getattr(evidence, "dataset_fingerprint", None):
                dataset_id = evidence.dataset_fingerprint.dataset_hash
                
            context = RunContext(
                config=self.config,
                store=store,
                evidence=evidence,
                state=state,
                tree=tree,
                dataset_id=dataset_id
            )
            
            # 3. Baselines
            state.phase = SearchPhase.BASELINING
            self._run_baselines(X, y, context)
            
            # 4. Search Loop
            state.phase = SearchPhase.SEARCHING
            self._search_loop(X, y, context)
            
            # 5. Finalization
            state.phase = SearchPhase.FINALIZING
            final_score = self._finalize(X, y, context)
            
            state.phase = SearchPhase.DONE
            logger.info("Orchestrator run completed successfully.")
            return final_score
            
        except Exception as e:
            logger.exception(f"Orchestrator run failed: {e}")
            state.phase = SearchPhase.FAILED
            raise
        finally:
            # Final state save
            self._save_checkpoint(dataset_id if 'dataset_id' in locals() else "default", state, store)

    def resume(self, run_id: str) -> None:
        """Resumes a run from a checkpoint."""
        self.config.run_id = run_id
        _, state, evidence = self.lifecycle.load_existing_run()
        
        store = ExperimentStore(self.lifecycle.get_db_manager())
        
        # Load dataset
        if self.config.dataset_path:
            from automl.data.loader import load_dataset
            df, _ = load_dataset(self.config.dataset_path, self.config.target_column)
            X = df.drop(columns=[self.config.target_column])
            y = df[self.config.target_column]
        else:
            raise ValueError("Dataset path is required to resume.")
            
        tree = self._rebuild_tree(state, store)
        fp = getattr(evidence, "dataset_fingerprint", None)
        dataset_id = fp.dataset_hash if fp else "default_dataset_id"
        
        context = RunContext(
            config=self.config,
            store=store,
            evidence=evidence,
            state=state,
            tree=tree,
            dataset_id=dataset_id
        )
        
        logger.info(f"Resuming run at phase {state.phase.value}")
        
        if state.phase.value in ("initializing", "profiling", "baselining"):
            # Too early to resume cleanly, just restart search
            self._search_loop(X, y, context)
            self._finalize(X, y, context)
        elif state.phase == SearchPhase.SEARCHING:
            self._search_loop(X, y, context)
            self._finalize(X, y, context)
        elif state.phase == SearchPhase.FINALIZING:
            self._finalize(X, y, context)
            
    # ------------------------------------------------------------------
    # Pipeline Stages
    # ------------------------------------------------------------------

    def _run_data_intelligence(self) -> tuple[EvidencePack, pd.DataFrame]:
        logger.info("Executing Phase 1: Data Intelligence")
        pipeline = DataIntelligencePipeline(self.config)
        
        if not self.config.dataset_path or not self.config.target_column:
            raise ValueError("Dataset path and target column must be provided.")
            
        evidence, df = pipeline.run(self.config.dataset_path, self.config.target_column)
        return evidence, df

    def _lock_metric(self, evidence: EvidencePack) -> None:
        logger.info("Selecting primary metric...")
        selector = MetricSelector()
        metric, direction = selector.select(evidence.task, evidence.fast_profile)
        
        evidence.lock_metric(metric, direction)
        assert_metric_locked(evidence) # INVARIANT
        
        logger.info(f"LOCKED Metric: {metric.value} (Direction: {direction.value})")
        self.lifecycle.log_event(Event(
            event_type=EventType.METRIC_LOCKED,
            run_id=self.config.run_id,
            message=f"Locked metric: {metric.value}",
            payload={"metric": metric.value, "direction": direction.value}
        ))

    def _run_baselines(
        self, X: pd.DataFrame, y: pd.Series, context: RunContext
    ) -> None:
        logger.info("Executing Phase 2: Baselines")
        assert_metric_locked(context.evidence) # INVARIANT
        
        hub = BaselineHub()
        specs = hub.generate_baselines(context.evidence.task)
        
        runner = ExperimentRunner(context.config)
        
        for spec in specs:
            result = runner.run(spec, X, y, context.evidence)
            context.tree.insert_node(spec)
            context.tree.update_node_result(result)
            
            # Save results
            context.store.save_experiment_spec(context.state.run_id, context.dataset_id, spec)
            context.store.save_experiment_result(result)
            
            # Update state manually for baselines
            if result.status.value == "success":
                context.state.completed_experiments.append(result.experiment_id)
                
                # Check incumbent
                from automl.application.metric_utils import is_better
                if context.state.incumbent_score is None or (
                    result.cv_score_mean is not None and 
                    is_better(result.cv_score_mean, context.state.incumbent_score, context.evidence.metric_direction)
                ):
                    context.state.incumbent_score = result.cv_score_mean
                    context.state.incumbent_score_std = result.cv_score_std
                    context.state.incumbent_experiment_id = result.experiment_id
            else:
                context.state.failed_experiments.append(result.experiment_id)
                
        self._save_checkpoint(context.dataset_id, context.state, context.store)

    def _search_loop(
        self, X: pd.DataFrame, y: pd.Series, context: RunContext
    ) -> None:
        logger.info("Executing Phase 7: Autonomous Search Loop")
        
        # Build dependencies
        slm_client = SLMClient(context.config.slm)
        planner = HypothesisGenerator(context.config.slm, slm_invoke_func=slm_client.generate)
        selector = HypothesisSelector()
        runner = ExperimentRunner(context.config)
        
        rc = context.config.resource
        governor = ResourceGovernor(max_memory_mb=rc.available_ram_gb * 1024, max_threads=rc.max_threads)
        
        sc = context.config.search
        detector = SaturationDetector(
            plateau_patience=sc.plateau_window, 
            min_gain_threshold=sc.min_gain_threshold,
            metric_direction=context.evidence.metric_direction
        )
        strategist = SearchStrategist(metric_direction=context.evidence.metric_direction)
        
        controller = SearchController(
            planner=planner,
            selector=selector,
            runner=runner,
            governor=governor,
            detector=detector,
            strategist=strategist,
        )
        
        # The controller will execute the loop and update tree/state
        # Base model parameters extracted from incumbent
        incumbent_node = context.tree.get_incumbent(context.evidence.metric_direction)
        base_model_name = "HistGradientBoostingClassifier" # fallback
        base_hp = {}
        if incumbent_node:
            spec = context.store.get_spec(incumbent_node.experiment_id)
            if spec:
                base_model_name = spec.model_name
                base_hp = spec.hyperparameters
                
        controller.search(X, y, context.evidence, base_model_name, base_hp, context)
        self._save_checkpoint(context.dataset_id, context.state, context.store)

    def _finalize(
        self, X: pd.DataFrame, y: pd.Series, context: RunContext
    ) -> float | None:
        logger.info("Executing Phase 10-11: Finalization & Packaging")
        
        incumbent = context.tree.get_incumbent(context.evidence.metric_direction)
        if not incumbent:
            logger.warning("No incumbent found for finalization. Aborting.")
            return

        spec = context.store.get_spec(incumbent.experiment_id)
        if not spec:
            raise RuntimeError("Incumbent spec not found in store.")

        if not context.evidence.split or not context.evidence.split.test_indices:
            raise RuntimeError("No test indices found for final evaluation.")

        # 4.2 Select Final Ensemble/Model (using Single Incumbent)
        tv_idx = context.evidence.split.train_val_indices
        test_idx = context.evidence.split.test_indices
        
        X_train_val = X.iloc[tv_idx]
        y_train_val = y.iloc[tv_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Fit final pipeline
        compiler = PipelineCompiler()
        final_pipeline = compiler.compile(spec)
        final_pipeline.fit(X_train_val, y_train_val)

        final_model_selection = {
            "selection_id": "final_" + incumbent.experiment_id,
            "type": "single",
            "weights": {incumbent.experiment_id: 1.0}
        }
        fitted_pipelines = {incumbent.experiment_id: final_pipeline}

        # 4.3 Trigger FinalHoldoutEvaluator
        evaluator = FinalHoldoutEvaluator()
        final_score = evaluator.evaluate(
            state=context.state,
            final_model_selection=final_model_selection,
            X_test=X_test,
            y_test=y_test,
            fitted_pipelines=fitted_pipelines,
            evidence=context.evidence
        )
        
        # 4.4 Save Final Model & Report
        import joblib
        model_path = self.lifecycle.model_dir / "final_pipeline.pkl"
        joblib.dump(final_pipeline, model_path)
        logger.info(f"Final pipeline saved to: {model_path}")
        
        # For now, we persist the test score into the state or lifecycle
        logger.info(f"Final pipeline packaged successfully. Test score: {final_score}")
        return final_score

    def _save_checkpoint(self, dataset_id: str, state: SearchState, store: ExperimentStore) -> None:
        store.save_checkpoint(dataset_id, state)
        
    def _rebuild_tree(self, state: SearchState, store: ExperimentStore) -> ExperimentTree:
        tree = ExperimentTree()
        specs = store.get_all_specs(state.run_id)
        results = store.get_all_results(state.run_id)
        
        for spec in specs:
            tree.insert_node(spec)
        for result in results:
            tree.update_node_result(result)
            
        return tree

