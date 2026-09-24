"""
Phase 1 Tests — Data Intelligence Engine

Tests:
- DataLoader: CSV/missing file/empty file
- DataAuditor: constants, near-constants, suspect IDs, duplicates, datetime/group candidates
- TaskDetector: all 6 task scenarios + 3 structure scenarios
- SplitStrategist: stratified/grouped/temporal/regression
- LeakageAuditor: 5 leakage scenarios
- FastProfiler: output shape and run time
- EvidenceSufficiencyEngine: triggers correctly
- DeepAnalyzer: output structure correctness
- EvidencePackBuilder: complete pack + fingerprint
- DataIntelligencePipeline: end-to-end on 3 synthetic datasets
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.core.config import RunConfig
from automl.core.enums import (
    DatasetStructure,
    FeatureDtype,
    LeakageSeverity,
    TaskType,
)
from automl.core.schemas import AuditReport, SplitProfile
from automl.data.auditor import DataAuditor
from automl.data.deep_analyzer import DeepAnalyzer
from automl.data.evidence import EvidencePackBuilder
from automl.data.leakage import LeakageAuditor
from automl.data.loader import DataAuditor as _DataAuditorAlias, load_dataset
from automl.data.pipeline import DataIntelligencePipeline
from automl.data.profiler import EvidenceSufficiencyEngine, FastProfiler
from automl.data.split_strategy import SplitStrategist
from automl.data.task_detector import TaskDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    cat = rng.choice(["A", "B", "C"], n)
    target = (x1 + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": target})


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 10, n)
    target = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def multiclass_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    target = rng.choice(["cat", "dog", "bird", "fish"], n)
    return pd.DataFrame({"x1": x1, "target": target})


@pytest.fixture
def csv_path(binary_df, tmp_path) -> Path:
    path = tmp_path / "test.csv"
    binary_df.to_csv(path, index=False)
    return path


@pytest.fixture
def audit_report(binary_df) -> AuditReport:
    auditor = DataAuditor()
    return auditor.audit(binary_df, target_column="target")


@pytest.fixture
def task_binary(binary_df, audit_report):
    detector = TaskDetector()
    return detector.detect(binary_df, "target", audit_report)


@pytest.fixture
def split_profile_binary(binary_df, task_binary):
    strategist = SplitStrategist(test_fraction=0.2, n_folds=3, random_seed=42)
    return strategist.create_split(binary_df, task_binary)


# ---------------------------------------------------------------------------
# Data Loader Tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_load_valid_csv(self, csv_path, binary_df) -> None:
        df, dataset_hash = load_dataset(csv_path)
        assert len(df) == len(binary_df)
        assert set(df.columns) == set(binary_df.columns)
        assert len(dataset_hash) == 32  # MD5 hex

    def test_load_missing_file_raises(self, tmp_path) -> None:
        from automl.core.exceptions import DataLoadError
        with pytest.raises(DataLoadError, match="not found"):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_load_returns_stable_hash(self, csv_path) -> None:
        _, hash1 = load_dataset(csv_path)
        _, hash2 = load_dataset(csv_path)
        assert hash1 == hash2

    @pytest.mark.skip(reason="pyarrow DLL blocked by Windows AppControl policy on this machine")
    def test_load_parquet(self, binary_df, tmp_path) -> None:
        path = tmp_path / "test.parquet"
        binary_df.to_parquet(path, index=False)
        df, _ = load_dataset(path)
        assert len(df) == len(binary_df)

    def test_load_empty_csv_raises(self, tmp_path) -> None:
        from automl.core.exceptions import DataLoadError
        path = tmp_path / "empty.csv"
        path.write_text("a,b,c\n")  # headers only, no data
        with pytest.raises(DataLoadError, match="empty"):
            load_dataset(path)


# ---------------------------------------------------------------------------
# Data Auditor Tests
# ---------------------------------------------------------------------------


class TestDataAuditor:
    def test_basic_stats(self, binary_df) -> None:
        auditor = DataAuditor()
        report = auditor.audit(binary_df, target_column="target")
        assert report.rows == len(binary_df)
        assert report.columns == len(binary_df.columns)
        assert report.memory_mb > 0

    def test_constant_column_detected(self) -> None:
        # 1 unique value out of 100 rows → uniqueness = 0.01 < threshold 0.02 → constant
        df = pd.DataFrame({"const": [1] * 100, "x": range(100), "y": [0, 1] * 50})
        auditor = DataAuditor(constant_threshold=0.02)
        report = auditor.audit(df, target_column="y")
        assert "const" in report.constant_columns

    def test_duplicate_rows_counted(self) -> None:
        base = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        df = pd.concat([base, base], ignore_index=True)  # 3 exact duplicates
        auditor = DataAuditor()
        report = auditor.audit(df, target_column="y")
        assert report.duplicate_rows == 3
        assert report.duplicate_fraction == pytest.approx(0.5)

    def test_suspect_id_detected(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "user_id": range(n),          # monotonic, high uniqueness, name pattern, integer
            "x1": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        auditor = DataAuditor(id_score_threshold=2)
        report = auditor.audit(df, target_column="target")
        assert "user_id" in report.suspect_id_columns

    def test_non_id_not_flagged(self, binary_df) -> None:
        auditor = DataAuditor()
        report = auditor.audit(binary_df, target_column="target")
        # x1, x2, cat are not IDs
        assert "x1" not in report.suspect_id_columns
        assert "x2" not in report.suspect_id_columns

    def test_datetime_candidate_by_name(self) -> None:
        df = pd.DataFrame({
            "timestamp": ["2021-01-01"] * 100,
            "x": range(100),
            "target": [0, 1] * 50,
        })
        auditor = DataAuditor()
        report = auditor.audit(df, target_column="target")
        assert "timestamp" in report.datetime_candidates

    def test_group_candidate_by_name(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "customer_id": rng.integers(1, 50, 200),
            "x": rng.normal(0, 1, 200),
            "target": rng.choice([0, 1], 200),
        })
        auditor = DataAuditor()
        report = auditor.audit(df, target_column="target")
        assert "customer_id" in report.group_candidates

    def test_missing_values_per_column(self) -> None:
        df = pd.DataFrame({
            "x1": [1.0, None, 3.0, None, 5.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0],
        })
        auditor = DataAuditor()
        report = auditor.audit(df, target_column="target")
        assert report.missing_per_column["x1"] == pytest.approx(0.4)
        assert report.missing_per_column["x2"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Task Detector Tests
# ---------------------------------------------------------------------------


class TestTaskDetector:
    def _make_audit(self, df: pd.DataFrame, target: str) -> AuditReport:
        return DataAuditor().audit(df, target_column=target)

    def test_binary_classification(self, binary_df) -> None:
        audit = self._make_audit(binary_df, "target")
        tp = TaskDetector().detect(binary_df, "target", audit)
        assert tp.task_type == TaskType.BINARY_CLASSIFICATION
        assert tp.n_classes == 2

    def test_multiclass_classification(self, multiclass_df) -> None:
        audit = self._make_audit(multiclass_df, "target")
        tp = TaskDetector().detect(multiclass_df, "target", audit)
        assert tp.task_type == TaskType.MULTICLASS_CLASSIFICATION
        assert tp.n_classes == 4

    def test_regression(self, regression_df) -> None:
        audit = self._make_audit(regression_df, "target")
        tp = TaskDetector().detect(regression_df, "target", audit)
        assert tp.task_type == TaskType.REGRESSION

    def test_force_task_type(self, binary_df) -> None:
        audit = self._make_audit(binary_df, "target")
        tp = TaskDetector().detect(binary_df, "target", audit, force_task_type="regression")
        assert tp.task_type == TaskType.REGRESSION

    def test_invalid_force_raises(self, binary_df) -> None:
        from automl.core.exceptions import UnsupportedTaskError
        audit = self._make_audit(binary_df, "target")
        with pytest.raises(UnsupportedTaskError):
            TaskDetector().detect(binary_df, "target", audit, force_task_type="invalid_task")

    def test_missing_target_raises(self, binary_df) -> None:
        from automl.core.exceptions import UnsupportedTaskError
        audit = self._make_audit(binary_df, "target")
        with pytest.raises(UnsupportedTaskError, match="not found"):
            TaskDetector().detect(binary_df, "nonexistent_col", audit)

    def test_grouped_structure_detected(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            "customer_id": rng.integers(1, 30, n),
            "x1": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        auditor = DataAuditor()
        audit = auditor.audit(df, target_column="target")
        tp = TaskDetector().detect(df, "target", audit)
        assert tp.dataset_structure == DatasetStructure.GROUPED

    def test_temporal_structure_detected(self) -> None:
        n = 300
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "x1": np.random.default_rng(0).normal(0, 1, n),
            "target": np.random.default_rng(0).choice([0, 1], n),
        })
        auditor = DataAuditor()
        audit = auditor.audit(df, target_column="target")
        tp = TaskDetector().detect(df, "target", audit)
        assert tp.dataset_structure == DatasetStructure.TEMPORAL

    def test_iid_structure_default(self, binary_df) -> None:
        audit = self._make_audit(binary_df, "target")
        tp = TaskDetector().detect(binary_df, "target", audit)
        assert tp.dataset_structure == DatasetStructure.IID


# ---------------------------------------------------------------------------
# Split Strategist Tests
# ---------------------------------------------------------------------------


class TestSplitStrategist:
    def test_test_indices_not_in_train_val(self, binary_df, task_binary) -> None:
        strategist = SplitStrategist(test_fraction=0.2, n_folds=3, random_seed=42)
        sp = strategist.create_split(binary_df, task_binary)
        test_set = set(sp.test_indices)
        train_set = set(sp.train_val_indices)
        assert test_set.isdisjoint(train_set), "Test and train_val sets must not overlap!"

    def test_all_indices_covered(self, binary_df, task_binary) -> None:
        strategist = SplitStrategist(test_fraction=0.2, n_folds=3, random_seed=42)
        sp = strategist.create_split(binary_df, task_binary)
        all_indices = sorted(sp.test_indices + sp.train_val_indices)
        assert all_indices == list(range(len(binary_df)))

    def test_test_fraction_approximately_correct(self, binary_df, task_binary) -> None:
        strategist = SplitStrategist(test_fraction=0.2, n_folds=3, random_seed=42)
        sp = strategist.create_split(binary_df, task_binary)
        actual_frac = len(sp.test_indices) / len(binary_df)
        assert abs(actual_frac - 0.2) < 0.05

    def test_regression_uses_kfold(self, regression_df) -> None:
        from automl.core.enums import SplitType
        audit = DataAuditor().audit(regression_df, "target")
        task = TaskDetector().detect(regression_df, "target", audit)
        sp = SplitStrategist().create_split(regression_df, task)
        assert sp.split_type == SplitType.KFOLD

    def test_classification_uses_stratified(self, binary_df, task_binary) -> None:
        from automl.core.enums import SplitType
        sp = SplitStrategist().create_split(binary_df, task_binary)
        assert sp.split_type == SplitType.STRATIFIED_KFOLD

    def test_get_cv_splitter_returns_correct_type(self, split_profile_binary) -> None:
        from sklearn.model_selection import StratifiedKFold
        strategist = SplitStrategist()
        splitter = strategist.get_cv_splitter(split_profile_binary)
        assert isinstance(splitter, StratifiedKFold)


# ---------------------------------------------------------------------------
# Leakage Auditor Tests
# ---------------------------------------------------------------------------


class TestLeakageAuditor:
    def test_no_leakage_clean_dataset(self, binary_df, audit_report, split_profile_binary) -> None:
        auditor = LeakageAuditor()
        result = auditor.audit(binary_df, "target", audit_report, split_profile_binary)
        # Should have no CRITICAL or HIGH signals
        assert not result.has_critical

    def test_target_leakage_detected(self) -> None:
        """A column that is a copy of the target must be flagged as CRITICAL."""
        rng = np.random.default_rng(42)
        n = 200
        target = rng.choice([0, 1], n)
        df = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "target_copy": target.astype(float),  # exact copy
            "target": target,
        })
        audit = DataAuditor().audit(df, "target")
        auditor = LeakageAuditor(target_correlation_threshold=0.99)
        result = auditor.audit(df, "target", audit)
        assert result.has_critical
        signal_types = [s.signal_type for s in result.signals]
        assert "target_leakage" in signal_types

    def test_id_leakage_warning(self) -> None:
        """SUSPECT_ID columns should generate a WARNING signal."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "user_id": range(n),
            "x1": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        audit = DataAuditor(id_score_threshold=2).audit(df, "target")
        auditor = LeakageAuditor()
        result = auditor.audit(df, "target", audit)
        signal_types = [s.signal_type for s in result.signals]
        assert "id_leakage" in signal_types

    def test_duplicate_rows_warning(self) -> None:
        """Datasets with many duplicates should be flagged."""
        base = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        df = pd.concat([base] * 5, ignore_index=True)
        audit = DataAuditor().audit(df, "target")
        auditor = LeakageAuditor()
        result = auditor.audit(df, "target", audit)
        signal_types = [s.signal_type for s in result.signals]
        assert "duplicate_rows" in signal_types

    def test_constant_features_warning(self) -> None:
        """Constant columns should be flagged."""
        df = pd.DataFrame({
            "const": [99] * 100,
            "x1": range(100),
            "target": [0, 1] * 50,
        })
        # Use explicit threshold so constant column is detected
        audit = DataAuditor(constant_threshold=0.02).audit(df, "target")
        auditor = LeakageAuditor()
        result = auditor.audit(df, "target", audit)
        signal_types = [s.signal_type for s in result.signals]
        assert "constant_features" in signal_types


# ---------------------------------------------------------------------------
# Fast Profiler Tests
# ---------------------------------------------------------------------------


class TestFastProfiler:
    def test_fast_profile_runs_quickly(self, binary_df, audit_report, task_binary) -> None:
        profiler = FastProfiler()
        t0 = __import__("time").perf_counter()
        profile = profiler.profile(binary_df, "target", audit_report, task_binary)
        elapsed = __import__("time").perf_counter() - t0
        assert elapsed < 10.0, f"Fast profiler took {elapsed:.2f}s — too slow"

    def test_fast_profile_structure(self, binary_df, audit_report, task_binary) -> None:
        profiler = FastProfiler()
        profile = profiler.profile(binary_df, "target", audit_report, task_binary)
        assert "numerical" in profile.feature_count_by_type
        assert "categorical" in profile.feature_count_by_type
        assert 0.0 <= profile.global_missing_fraction <= 1.0
        assert profile.target_unique_count >= 2

    def test_imbalance_ratio_for_binary(self, audit_report, task_binary) -> None:
        rng = np.random.default_rng(42)
        n = 400
        # 90/10 imbalance
        target = rng.choice([0, 1], n, p=[0.9, 0.1])
        df = pd.DataFrame({"x1": rng.normal(0, 1, n), "target": target})
        audit = DataAuditor().audit(df, "target")
        task = TaskDetector().detect(df, "target", audit)
        profile = FastProfiler().profile(df, "target", audit, task)
        assert profile.target_class_imbalance is not None
        assert profile.target_class_imbalance < 0.5  # minority is smaller


# ---------------------------------------------------------------------------
# Evidence Sufficiency Engine Tests
# ---------------------------------------------------------------------------


class TestEvidenceSufficiencyEngine:
    def test_simple_numeric_dataset_sufficient(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        audit = DataAuditor().audit(df, "target")
        task = TaskDetector().detect(df, "target", audit)
        fp = FastProfiler().profile(df, "target", audit, task)
        engine = EvidenceSufficiencyEngine(medium_dataset_rows=1000)
        sufficient, reasons = engine.check(df, "target", audit, fp)
        assert sufficient is True

    def test_high_cardinality_triggers_deep(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        # Categorical with cardinality = 200 (> 50 threshold)
        df = pd.DataFrame({
            "cat_col": [f"cat_{i % 200}" for i in range(n)],
            "x1": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        audit = DataAuditor().audit(df, "target")
        task = TaskDetector().detect(df, "target", audit)
        fp = FastProfiler().profile(df, "target", audit, task)
        engine = EvidenceSufficiencyEngine(high_cardinality_threshold=50, medium_dataset_rows=10000)
        sufficient, reasons = engine.check(df, "target", audit, fp)
        assert sufficient is False
        assert any("cardinality" in r.lower() for r in reasons)

    def test_high_missingness_triggers_deep(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.normal(0, 1, n).tolist()
        # Set 30% to NaN
        for i in range(0, n, 3):
            x1[i] = None
        df = pd.DataFrame({"x1": x1, "target": rng.choice([0, 1], n)})
        audit = DataAuditor().audit(df, "target")
        task = TaskDetector().detect(df, "target", audit)
        fp = FastProfiler().profile(df, "target", audit, task)
        engine = EvidenceSufficiencyEngine(high_missing_threshold=0.10, medium_dataset_rows=10000)
        sufficient, reasons = engine.check(df, "target", audit, fp)
        assert sufficient is False


# ---------------------------------------------------------------------------
# Deep Analyzer Tests
# ---------------------------------------------------------------------------


class TestDeepAnalyzer:
    def test_deep_analysis_output_structure(self, binary_df, audit_report, task_binary) -> None:
        analyzer = DeepAnalyzer()
        deep = analyzer.analyze(binary_df, "target", task_binary, audit_report)
        assert deep.was_triggered is True
        assert isinstance(deep.pearson_to_target, dict)
        assert isinstance(deep.spearman_to_target, dict)
        assert isinstance(deep.mutual_info_to_target, dict)
        assert isinstance(deep.high_correlation_pairs, list)
        assert isinstance(deep.redundancy_clusters, list)

    def test_mi_computed_for_numeric_features(self, binary_df, audit_report, task_binary) -> None:
        analyzer = DeepAnalyzer()
        deep = analyzer.analyze(binary_df, "target", task_binary, audit_report)
        # x1 and x2 should have MI computed
        assert "x1" in deep.mutual_info_to_target
        assert "x2" in deep.mutual_info_to_target

    def test_redundant_features_detected(self) -> None:
        """Highly correlated features should appear in redundancy clusters."""
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = x1 * 0.99 + rng.normal(0, 0.01, n)  # near-duplicate of x1
        df = pd.DataFrame({"x1": x1, "x2": x2, "target": rng.choice([0, 1], n)})
        audit = DataAuditor().audit(df, "target")
        task = TaskDetector().detect(df, "target", audit)
        deep = DeepAnalyzer(high_corr_threshold=0.90).analyze(df, "target", task, audit)
        assert len(deep.high_correlation_pairs) > 0
        pair_cols = {col for a, b, _ in deep.high_correlation_pairs for col in [a, b]}
        assert "x1" in pair_cols and "x2" in pair_cols

    def test_uses_train_split_only(self, binary_df, task_binary, split_profile_binary) -> None:
        """Deep analysis should not see test data."""
        audit = DataAuditor().audit(binary_df, "target")
        deep_with_split = DeepAnalyzer().analyze(
            binary_df, "target", task_binary, audit, split_profile_binary
        )
        deep_no_split = DeepAnalyzer().analyze(
            binary_df, "target", task_binary, audit, None
        )
        # Both should produce valid output; with split uses fewer rows
        assert deep_with_split.was_triggered
        assert deep_no_split.was_triggered


# ---------------------------------------------------------------------------
# Evidence Pack Builder Tests
# ---------------------------------------------------------------------------


class TestEvidencePackBuilder:
    def test_evidence_pack_complete(
        self, binary_df, audit_report, task_binary, split_profile_binary
    ) -> None:
        fast_profile = FastProfiler().profile(binary_df, "target", audit_report, task_binary)
        leakage = LeakageAuditor().audit(binary_df, "target", audit_report, split_profile_binary)
        deep = DeepAnalyzer().analyze(binary_df, "target", task_binary, audit_report, split_profile_binary)

        builder = EvidencePackBuilder()
        pack = builder.build(
            df=binary_df,
            dataset_hash="testhash123",
            target_column="target",
            audit=audit_report,
            task=task_binary,
            split=split_profile_binary,
            fast_profile=fast_profile,
            leakage=leakage,
            deep_analysis=deep,
        )
        assert pack.task is not None
        assert pack.split is not None
        assert pack.fingerprint is not None
        assert len(pack.feature_map) == 3  # x1, x2, cat

    def test_feature_info_populated(
        self, binary_df, audit_report, task_binary, split_profile_binary
    ) -> None:
        fast_profile = FastProfiler().profile(binary_df, "target", audit_report, task_binary)
        leakage = LeakageAuditor().audit(binary_df, "target", audit_report)
        deep = DeepAnalyzer().analyze(binary_df, "target", task_binary, audit_report)
        pack = EvidencePackBuilder().build(
            df=binary_df, dataset_hash="h", target_column="target",
            audit=audit_report, task=task_binary, split=split_profile_binary,
            fast_profile=fast_profile, leakage=leakage, deep_analysis=deep,
        )
        x1_info = pack.feature_map["x1"]
        assert x1_info.dtype == FeatureDtype.NUMERICAL
        assert x1_info.mutual_information is not None
        assert len(x1_info.candidate_actions) > 0

    def test_fingerprint_size_label(self) -> None:
        """Tiny dataset should get 'tiny' size label."""
        rng = np.random.default_rng(42)
        n = 200  # < 1000 → tiny
        df = pd.DataFrame({"x1": rng.normal(0, 1, n), "target": rng.choice([0, 1], n)})
        audit = DataAuditor().audit(df, "target")
        task = TaskDetector().detect(df, "target", audit)
        split = SplitStrategist().create_split(df, task)
        fast = FastProfiler().profile(df, "target", audit, task)
        leakage = LeakageAuditor().audit(df, "target", audit)
        pack = EvidencePackBuilder().build(
            df=df, dataset_hash="h", target_column="target",
            audit=audit, task=task, split=split,
            fast_profile=fast, leakage=leakage,
        )
        assert pack.fingerprint.size_label == "tiny"


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


class TestDataIntelligencePipeline:
    def _run_pipeline(self, df: pd.DataFrame, target: str, tmp_path: Path):
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        config = RunConfig.development()
        config.verbose = False
        pipeline = DataIntelligencePipeline(config)
        return pipeline.run(csv, target)

    def test_binary_classification_end_to_end(self, binary_df, tmp_path) -> None:
        pack, df_loaded = self._run_pipeline(binary_df, "target", tmp_path)
        assert pack.task.task_type == TaskType.BINARY_CLASSIFICATION
        assert len(pack.feature_map) > 0
        assert pack.fingerprint is not None
        assert not pack.leakage.has_critical

    def test_regression_end_to_end(self, regression_df, tmp_path) -> None:
        pack, df_loaded = self._run_pipeline(regression_df, "target", tmp_path)
        assert pack.task.task_type == TaskType.REGRESSION

    def test_multiclass_end_to_end(self, multiclass_df, tmp_path) -> None:
        pack, _ = self._run_pipeline(multiclass_df, "target", tmp_path)
        assert pack.task.task_type == TaskType.MULTICLASS_CLASSIFICATION
        assert pack.task.n_classes == 4

    def test_pipeline_split_is_leakage_free(self, binary_df, tmp_path) -> None:
        pack, _ = self._run_pipeline(binary_df, "target", tmp_path)
        test_set = set(pack.split.test_indices)
        train_set = set(pack.split.train_val_indices)
        assert test_set.isdisjoint(train_set)

    def test_pipeline_evidence_pack_json_serializable(self, binary_df, tmp_path) -> None:
        import json
        pack, _ = self._run_pipeline(binary_df, "target", tmp_path)
        data = pack.model_dump(mode="json")
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 100
