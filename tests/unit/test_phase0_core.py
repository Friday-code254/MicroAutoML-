"""
Phase 0 Tests — Core Schemas, Enums, Config, State

Tests:
- All Enum values are unique and JSON-serializable
- All Pydantic schemas serialize and deserialize cleanly (round-trip)
- RunConfig validation catches bad inputs
- ExperimentSpec enforces parent_id invariant
- SplitProfile.mark_test_used() raises on second call
- EvidencePack.lock_metric() raises on second call
- RunState.set_final_metrics() raises on second call
- DatasetFingerprint.to_lookup_key() is stable
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from automl.core.config import RunConfig, ResourceConfig, SearchConfig, SLMConfig
from automl.core.enums import (
    BranchStatus,
    DatasetStructure,
    ExperimentStatus,
    FailureType,
    FeatureDtype,
    Fidelity,
    LeakageSeverity,
    MetricDirection,
    MetricName,
    ModelFamily,
    OperatorType,
    ResourceDecision,
    SaturationState,
    SearchPhase,
    SplitType,
    TaskType,
    TransformType,
)
from automl.core.exceptions import (
    FinalTestAlreadyUsedError,
    MetricLockedError,
    PreSplitFitError,
    SearchLockedError,
)
from automl.core.schemas import (
    AuditReport,
    DatasetFingerprint,
    DeepAnalysis,
    EvidencePack,
    ExperimentResult,
    ExperimentSpec,
    FailureReport,
    FastProfile,
    FeatureInfo,
    FinalMetrics,
    FinalModel,
    Hypothesis,
    HypothesisSet,
    LeakageAudit,
    LeakageSignal,
    ResourceBudget,
    ResourceUsageRecord,
    SearchNode,
    SearchState,
    SplitProfile,
    TaskProfile,
)
from automl.core.state import RunState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_minimal_spec(operator: OperatorType = OperatorType.BASELINE) -> ExperimentSpec:
    """Create a minimal valid ExperimentSpec."""
    parent = "EXP_ROOT" if operator != OperatorType.BASELINE else None
    return ExperimentSpec(
        model_name="dummy",
        operator=operator,
        parent_id=parent,
        fidelity=Fidelity.SMOKE,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_all_task_types_unique(self) -> None:
        values = [t.value for t in TaskType]
        assert len(values) == len(set(values))

    def test_all_enums_json_serializable(self) -> None:
        """Every enum value should be directly JSON-serializable as a string."""
        enum_classes = [
            TaskType, DatasetStructure, SplitType, Fidelity, ExperimentStatus,
            OperatorType, FailureType, BranchStatus, SearchPhase, FeatureDtype,
            TransformType, LeakageSeverity, ModelFamily, ResourceDecision,
            MetricName, MetricDirection, SaturationState,
        ]
        for enum_cls in enum_classes:
            for member in enum_cls:
                # Should not raise
                result = json.dumps(member.value)
                assert isinstance(result, str)

    def test_str_enum_equality(self) -> None:
        assert TaskType.BINARY_CLASSIFICATION == "binary_classification"
        assert Fidelity.SMOKE == "smoke"
        assert MetricName.ROC_AUC == "roc_auc"


# ---------------------------------------------------------------------------
# Schema round-trip tests
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    def _assert_roundtrip(self, obj: object) -> None:
        """Serialize to JSON dict and back, assert equality."""
        data = obj.model_dump(mode="json")
        reconstructed = obj.__class__.model_validate(data)
        assert reconstructed.model_dump(mode="json") == data

    def test_audit_report_roundtrip(self) -> None:
        report = AuditReport(
            rows=1000, columns=20, memory_mb=5.0,
            suspect_id_columns=["id", "user_id"],
            missing_per_column={"age": 0.05, "income": 0.12},
        )
        self._assert_roundtrip(report)

    def test_feature_info_roundtrip(self) -> None:
        fi = FeatureInfo(
            feature="income",
            dtype=FeatureDtype.NUMERICAL,
            missing_ratio=0.03,
            unique_ratio=0.88,
            pearson=0.12,
            spearman=0.29,
            mutual_information=0.41,
            skew=2.4,
            candidate_actions=["log1p", "tree_model"],
        )
        self._assert_roundtrip(fi)

    def test_dataset_fingerprint_roundtrip(self) -> None:
        fp = DatasetFingerprint(
            sample_count=48921,
            feature_count=32,
            numeric_fraction=0.6,
            categorical_fraction=0.4,
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            size_label="medium",
        )
        self._assert_roundtrip(fp)

    def test_task_profile_roundtrip(self) -> None:
        tp = TaskProfile(
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_structure=DatasetStructure.IID,
            n_classes=2,
            class_names=["0", "1"],
            target_column="churn",
        )
        self._assert_roundtrip(tp)

    def test_split_profile_roundtrip(self) -> None:
        sp = SplitProfile(
            split_type=SplitType.STRATIFIED_KFOLD,
            n_folds=5,
            test_fraction=0.2,
            stratify=True,
        )
        self._assert_roundtrip(sp)

    def test_leakage_audit_roundtrip(self) -> None:
        signal = LeakageSignal(
            signal_type="target_leakage",
            affected_columns=["target_copy"],
            severity=LeakageSeverity.CRITICAL,
            description="Column is a copy of the target.",
        )
        audit = LeakageAudit(
            signals=[signal],
            overall_severity=LeakageSeverity.CRITICAL,
            pipeline_safe=False,
        )
        self._assert_roundtrip(audit)

    def test_evidence_pack_roundtrip(self) -> None:
        ep = EvidencePack(
            audit=AuditReport(rows=500),
            fast_profile=FastProfile(global_missing_fraction=0.05),
        )
        self._assert_roundtrip(ep)

    def test_experiment_spec_baseline_roundtrip(self) -> None:
        spec = ExperimentSpec(
            model_name="catboost",
            operator=OperatorType.BASELINE,
            fidelity=Fidelity.CHEAP,
            hyperparameters={"iterations": 50},
        )
        self._assert_roundtrip(spec)

    def test_experiment_spec_non_baseline_roundtrip(self) -> None:
        spec = ExperimentSpec(
            model_name="catboost",
            operator=OperatorType.FEATURE_ENGINEERING,
            parent_id="EXP_ROOT00",
            fidelity=Fidelity.MEDIUM,
            features=["log1p_income"],
        )
        self._assert_roundtrip(spec)

    def test_experiment_result_roundtrip(self) -> None:
        result = ExperimentResult(
            experiment_id="EXP_001",
            status=ExperimentStatus.SUCCESS,
            cv_score_mean=0.873,
            cv_score_std=0.006,
            fold_scores=[0.87, 0.88, 0.86],
        )
        self._assert_roundtrip(result)

    def test_failure_report_roundtrip(self) -> None:
        fr = FailureReport(
            experiment_id="EXP_002",
            failure_type=FailureType.MEMORY,
            error_message="OOM",
        )
        self._assert_roundtrip(fr)

    def test_hypothesis_roundtrip(self) -> None:
        h = Hypothesis(
            hypothesis="CatBoost with native categoricals should outperform OHE on this dataset.",
            evidence=["categorical_fraction=0.6", "mutual_information max=0.41"],
            experiment_type=OperatorType.MODEL_CHANGE,
            changes=["switch to CatBoost", "enable native categoricals"],
            expected_gain=0.02,
            expected_cost=0.3,
            success_condition="cv_score > incumbent + 0.005",
        )
        self._assert_roundtrip(h)

    def test_search_state_roundtrip(self) -> None:
        state = SearchState(
            run_id="RUN_TEST01",
            phase=SearchPhase.SEARCHING,
            incumbent_score=0.87,
            budget_total_seconds=3600.0,
            budget_elapsed_seconds=300.0,
        )
        self._assert_roundtrip(state)

    def test_final_metrics_roundtrip(self) -> None:
        fm = FinalMetrics(
            test_score=0.871,
            test_metric=MetricName.ROC_AUC,
            cv_score_mean=0.876,
            cv_score_std=0.006,
            n_test_samples=4892,
        )
        self._assert_roundtrip(fm)


# ---------------------------------------------------------------------------
# ExperimentSpec invariant tests
# ---------------------------------------------------------------------------


class TestExperimentSpecInvariants:
    def test_baseline_no_parent_ok(self) -> None:
        spec = ExperimentSpec(
            model_name="catboost",
            operator=OperatorType.BASELINE,
        )
        assert spec.parent_id is None

    def test_non_baseline_requires_parent(self) -> None:
        with pytest.raises(Exception):  # Pydantic ValidationError
            ExperimentSpec(
                model_name="catboost",
                operator=OperatorType.FEATURE_ENGINEERING,
                parent_id=None,  # Missing — should raise
            )

    def test_config_hash_is_stable(self) -> None:
        spec1 = ExperimentSpec(
            model_name="lightgbm",
            operator=OperatorType.BASELINE,
            hyperparameters={"n_estimators": 100, "learning_rate": 0.05},
            features=["log1p_income", "freq_enc_city"],
        )
        spec2 = ExperimentSpec(
            model_name="lightgbm",
            operator=OperatorType.BASELINE,
            hyperparameters={"n_estimators": 100, "learning_rate": 0.05},
            features=["freq_enc_city", "log1p_income"],  # different order
        )
        assert spec1.config_hash() == spec2.config_hash()

    def test_config_hash_differs_on_model_change(self) -> None:
        spec1 = ExperimentSpec(model_name="lightgbm", operator=OperatorType.BASELINE)
        spec2 = ExperimentSpec(model_name="catboost", operator=OperatorType.BASELINE)
        assert spec1.config_hash() != spec2.config_hash()


# ---------------------------------------------------------------------------
# SplitProfile single-use invariant
# ---------------------------------------------------------------------------


class TestSplitProfileInvariant:
    def test_mark_test_used_first_call_ok(self) -> None:
        sp = SplitProfile(split_type=SplitType.STRATIFIED_KFOLD)
        sp.mark_test_used()
        assert sp.test_set_used is True

    def test_mark_test_used_second_call_raises(self) -> None:
        sp = SplitProfile(split_type=SplitType.STRATIFIED_KFOLD)
        sp.mark_test_used()
        with pytest.raises(FinalTestAlreadyUsedError):
            sp.mark_test_used()


# ---------------------------------------------------------------------------
# EvidencePack metric lock invariant
# ---------------------------------------------------------------------------


class TestEvidencePackMetricLock:
    def test_lock_metric_once_ok(self) -> None:
        ep = EvidencePack()
        ep.lock_metric(MetricName.ROC_AUC, MetricDirection.HIGHER_IS_BETTER)
        assert ep.metric_locked is True
        assert ep.primary_metric == MetricName.ROC_AUC

    def test_lock_metric_twice_raises(self) -> None:
        ep = EvidencePack()
        ep.lock_metric(MetricName.ROC_AUC, MetricDirection.HIGHER_IS_BETTER)
        with pytest.raises(MetricLockedError):
            ep.lock_metric(MetricName.LOG_LOSS, MetricDirection.LOWER_IS_BETTER)


# ---------------------------------------------------------------------------
# RunConfig validation
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_cpu_default_is_valid(self) -> None:
        cfg = RunConfig.cpu_default()
        assert cfg.resource.available_ram_gb > 0

    def test_development_is_valid(self) -> None:
        cfg = RunConfig.development()
        assert cfg.slm.enabled is False
        assert cfg.search.time_budget_seconds == 120.0

    def test_negative_available_ram_raises(self) -> None:
        with pytest.raises(Exception):
            RunConfig(
                resource=ResourceConfig(
                    total_ram_gb=1.0,
                    os_reserved_gb=2.0,  # More than total — should fail
                )
            )

    def test_yaml_roundtrip(self, tmp_path) -> None:
        cfg = RunConfig.cpu_default()
        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)
        loaded = RunConfig.from_yaml(yaml_path)
        assert loaded.resource.total_ram_gb == cfg.resource.total_ram_gb
        assert loaded.random_seed == cfg.random_seed


# ---------------------------------------------------------------------------
# RunState invariants
# ---------------------------------------------------------------------------


class TestRunState:
    def test_set_final_metrics_once_ok(self) -> None:
        cfg = RunConfig.development()
        state = RunState(cfg, "RUN_TEST01")
        metrics = FinalMetrics(
            test_score=0.87,
            test_metric=MetricName.ROC_AUC,
            cv_score_mean=0.876,
            cv_score_std=0.006,
        )
        state.set_final_metrics(metrics)
        assert state.final_metrics_set is True

    def test_set_final_metrics_twice_raises(self) -> None:
        cfg = RunConfig.development()
        state = RunState(cfg, "RUN_TEST02")
        metrics = FinalMetrics(
            test_score=0.87,
            test_metric=MetricName.ROC_AUC,
            cv_score_mean=0.876,
            cv_score_std=0.006,
        )
        state.set_final_metrics(metrics)
        with pytest.raises(FinalTestAlreadyUsedError):
            state.set_final_metrics(metrics)

    def test_lock_search_prevents_new_experiments(self) -> None:
        cfg = RunConfig.development()
        state = RunState(cfg, "RUN_TEST03")
        state.lock_search()
        assert state.search_locked is True
        with pytest.raises(SearchLockedError):
            state.assert_search_open()

    def test_incumbent_updated_on_result_registration(self) -> None:
        cfg = RunConfig.development()
        state = RunState(cfg, "RUN_TEST04")
        result = ExperimentResult(
            experiment_id="EXP_001",
            status=ExperimentStatus.SUCCESS,
            cv_score_mean=0.85,
            cv_score_std=0.01,
        )
        state.register_result(result)
        assert state.search_state.incumbent_score == pytest.approx(0.85)
        assert state.search_state.incumbent_experiment_id == "EXP_001"

    def test_better_result_updates_incumbent(self) -> None:
        cfg = RunConfig.development()
        state = RunState(cfg, "RUN_TEST05")
        r1 = ExperimentResult(
            experiment_id="EXP_001",
            status=ExperimentStatus.SUCCESS,
            cv_score_mean=0.83,
        )
        r2 = ExperimentResult(
            experiment_id="EXP_002",
            status=ExperimentStatus.SUCCESS,
            cv_score_mean=0.87,
        )
        state.register_result(r1)
        state.register_result(r2)
        assert state.search_state.incumbent_score == pytest.approx(0.87)
        assert state.search_state.incumbent_experiment_id == "EXP_002"

    def test_hardware_info_captured(self) -> None:
        cfg = RunConfig.development()
        state = RunState(cfg, "RUN_TEST06")
        assert "total_ram_gb" in state.hardware_info
        assert "cpu_count_logical" in state.hardware_info
        assert state.hardware_info["total_ram_gb"] > 0


# ---------------------------------------------------------------------------
# DatasetFingerprint
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    def test_lookup_key_is_stable(self) -> None:
        fp = DatasetFingerprint(
            sample_count=1000,
            numeric_fraction=0.6,
            categorical_fraction=0.4,
            missing_fraction=0.05,
            nonlinear_signal=0.3,
            redundancy=0.1,
            size_label="small",
        )
        key1 = fp.to_lookup_key()
        key2 = fp.to_lookup_key()
        assert key1 == key2

    def test_different_fingerprints_have_different_keys(self) -> None:
        fp1 = DatasetFingerprint(numeric_fraction=0.8, categorical_fraction=0.2)
        fp2 = DatasetFingerprint(numeric_fraction=0.2, categorical_fraction=0.8)
        assert fp1.to_lookup_key() != fp2.to_lookup_key()


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_presplit_fit_error(self) -> None:
        err = PreSplitFitError("StandardScaler")
        assert "StandardScaler" in str(err)
        assert err.severity == LeakageSeverity.CRITICAL

    def test_final_test_already_used(self) -> None:
        err = FinalTestAlreadyUsedError()
        assert "final holdout" in str(err).lower()

    def test_metric_locked_error(self) -> None:
        err = MetricLockedError()
        assert "locked" in str(err).lower()
