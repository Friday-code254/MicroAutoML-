"""
MicroAutoML-Agent — Core Schemas (Pydantic Models)

All data-transfer objects (DTOs) and state containers used across the system.
Every module communicates through these schemas — never through raw dicts.

Invariants enforced here:
- ExperimentSpec always has a parent_id (except BASELINE experiments)
- FinalMetrics can only be set once (enforced in RunState)
- All Pydantic models are JSON-serializable (model_dump/model_validate)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from automl.core.enums import (
    BranchStatus,
    DatasetStructure,
    EventType,
    ExperimentStatus,
    FailureType,
    FeatureDtype,
    Fidelity,
    LeakageSeverity,
    MetricDirection,
    MetricName,
    ModelFamily,
    OperatorType,
    SaturationState,
    SearchPhase,
    SplitType,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_id(prefix: str = "ID") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8].upper()}"


# ---------------------------------------------------------------------------
# Feature Information Map
# ---------------------------------------------------------------------------


class FeatureInfo(BaseModel):
    """Per-feature evidence gathered during profiling and deep analysis."""

    feature: str
    dtype: FeatureDtype
    missing_ratio: float = 0.0
    unique_ratio: float = 0.0
    cardinality: int = 0

    # Statistical relationships
    pearson: float | None = None
    spearman: float | None = None
    mutual_information: float | None = None
    redundancy: float | None = None

    # Distribution
    skew: float | None = None
    kurtosis: float | None = None
    zero_fraction: float | None = None
    outlier_fraction: float | None = None

    # Flags
    is_constant: bool = False
    is_near_constant: bool = False
    is_suspect_id: bool = False
    is_datetime_candidate: bool = False
    is_group_candidate: bool = False

    # Recommended actions (from evidence)
    candidate_actions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset Fingerprint
# ---------------------------------------------------------------------------


class DatasetFingerprint(BaseModel):
    """
    Compact descriptor of the dataset used for meta-learning lookup.
    Built from the EvidencePack — never from the raw dataframe directly.
    """

    fingerprint_id: str = Field(default_factory=lambda: _new_id("DSET"))
    dataset_hash: str = ""  # MD5 of raw CSV bytes

    # Size
    sample_count: int = 0
    feature_count: int = 0

    # Type composition
    numeric_fraction: float = 0.0
    categorical_fraction: float = 0.0
    datetime_fraction: float = 0.0

    # Quality
    missing_fraction: float = 0.0
    duplicate_fraction: float = 0.0
    constant_fraction: float = 0.0

    # Cardinality
    mean_cardinality: float = 0.0
    max_cardinality: int = 0
    high_cardinality_fraction: float = 0.0  # fraction of features with cardinality > 50

    # Target
    class_entropy: float | None = None     # classification only
    imbalance_ratio: float | None = None   # minority/majority class ratio

    # Signal
    linear_signal: float = 0.0            # mean absolute Pearson to target
    nonlinear_signal: float = 0.0         # mean MI to target
    redundancy: float = 0.0               # mean pairwise feature correlation

    # Distribution
    mean_skew: float = 0.0

    # Structure
    task_type: TaskType | None = None
    dataset_structure: DatasetStructure = DatasetStructure.IID
    possible_group: bool = False
    possible_time: bool = False

    # Categorical label (for display / meta-learning)
    size_label: str = "medium"            # "tiny" | "small" | "medium" | "large"
    difficulty_label: str = "unknown"     # "easy" | "moderate" | "hard"

    def to_lookup_key(self) -> str:
        """Compact string key for meta-prior lookup."""
        parts = [
            self.size_label,
            f"num={self.numeric_fraction:.1f}",
            f"cat={self.categorical_fraction:.1f}",
            f"missing={self.missing_fraction:.2f}",
            f"nonlinear={self.nonlinear_signal:.2f}",
            f"redundancy={self.redundancy:.2f}",
            self.dataset_structure.value,
        ]
        return "|".join(parts)


# ---------------------------------------------------------------------------
# Audit / Profile
# ---------------------------------------------------------------------------


class AuditReport(BaseModel):
    """Raw statistics produced by the Data Auditor."""

    rows: int = 0
    columns: int = 0
    memory_mb: float = 0.0
    duplicate_rows: int = 0
    duplicate_fraction: float = 0.0
    constant_columns: list[str] = Field(default_factory=list)
    near_constant_columns: list[str] = Field(default_factory=list)
    suspect_id_columns: list[str] = Field(default_factory=list)
    datetime_candidates: list[str] = Field(default_factory=list)
    group_candidates: list[str] = Field(default_factory=list)
    missing_per_column: dict[str, float] = Field(default_factory=dict)
    cardinality_per_column: dict[str, int] = Field(default_factory=dict)
    dtype_per_column: dict[str, str] = Field(default_factory=dict)


class FastProfile(BaseModel):
    """Cheap statistics produced by the Fast Profiler (always runs)."""

    feature_count_by_type: dict[str, int] = Field(default_factory=dict)
    global_missing_fraction: float = 0.0
    target_value_counts: dict[str, int] = Field(default_factory=dict)
    target_dtype: str = ""
    target_unique_count: int = 0
    target_class_imbalance: float | None = None  # minority/majority
    feature_skew_stats: dict[str, float] = Field(default_factory=dict)  # feature -> skew
    run_seconds: float = 0.0


class DeepAnalysis(BaseModel):
    """Rich statistics produced by the Deep Analyzer (runs only when needed)."""

    # Relationship analysis
    pearson_to_target: dict[str, float] = Field(default_factory=dict)
    spearman_to_target: dict[str, float] = Field(default_factory=dict)
    mutual_info_to_target: dict[str, float] = Field(default_factory=dict)

    # Redundancy
    high_correlation_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    redundancy_clusters: list[list[str]] = Field(default_factory=list)

    # Distributions
    skew_per_feature: dict[str, float] = Field(default_factory=dict)
    outlier_fraction_per_feature: dict[str, float] = Field(default_factory=dict)
    zero_inflation_features: list[str] = Field(default_factory=list)

    # Drift signals (train vs test, if available)
    drifted_features: list[str] = Field(default_factory=list)
    drift_scores: dict[str, float] = Field(default_factory=dict)

    run_seconds: float = 0.0
    was_triggered: bool = False  # False = sufficiency engine skipped deep analysis


# ---------------------------------------------------------------------------
# Task / Split
# ---------------------------------------------------------------------------


class TaskProfile(BaseModel):
    """Output of the Task Detector."""

    task_type: TaskType
    dataset_structure: DatasetStructure = DatasetStructure.IID
    n_classes: int | None = None
    class_names: list[str] = Field(default_factory=list)
    target_column: str = ""
    group_column: str | None = None
    time_column: str | None = None
    detection_confidence: float = 1.0
    detection_notes: list[str] = Field(default_factory=list)
    completed_at: datetime | None = None
    experiment_history: list[str] = Field(default_factory=list)

    # Simple persistence flag
    _is_dirty: bool = False


# ---------------------------------------------------------------------------
# Event Log Architecture
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """
    An immutable record of a system transition.
    Forms the backbone of the Audit Trail.
    """

    event_id: str = Field(default_factory=lambda: _new_id("EVT"))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: EventType
    run_id: str
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class SplitProfile(BaseModel):
    """Output of the Split Strategist — defines the entire validation protocol."""

    split_type: SplitType
    n_folds: int = 5
    test_fraction: float = 0.2
    group_column: str | None = None
    time_column: str | None = None
    stratify: bool = False
    random_seed: int = 42

    # Actual indices (set after splitting)
    test_indices: list[int] = Field(default_factory=list)
    train_val_indices: list[int] = Field(default_factory=list)

    # Whether the final test set has been used
    test_set_used: bool = False

    def mark_test_used(self) -> None:
        from automl.core.exceptions import FinalTestAlreadyUsedError
        if self.test_set_used:
            raise FinalTestAlreadyUsedError()
        self.test_set_used = True


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


class LeakageSignal(BaseModel):
    """A single detected leakage signal."""

    signal_type: str
    affected_columns: list[str] = Field(default_factory=list)
    severity: LeakageSeverity = LeakageSeverity.WARNING
    description: str = ""
    auto_remedied: bool = False


class LeakageAudit(BaseModel):
    """Complete leakage audit report."""

    signals: list[LeakageSignal] = Field(default_factory=list)
    overall_severity: LeakageSeverity = LeakageSeverity.NONE
    pipeline_safe: bool = True  # True if all transforms are pipeline-wrapped
    audit_seconds: float = 0.0

    @property
    def has_critical(self) -> bool:
        return any(s.severity == LeakageSeverity.CRITICAL for s in self.signals)

    @property
    def has_high(self) -> bool:
        return any(s.severity == LeakageSeverity.HIGH for s in self.signals)


# ---------------------------------------------------------------------------
# Evidence Pack
# ---------------------------------------------------------------------------


class EvidencePack(BaseModel):
    """
    The central state object for Phase A.
    Everything the planner, selector, and search controller need to know
    about the dataset is encoded here.
    """

    evidence_id: str = Field(default_factory=lambda: _new_id("EV"))
    created_at: datetime = Field(default_factory=datetime.now)

    # Sub-reports
    audit: AuditReport = Field(default_factory=AuditReport)
    task: TaskProfile | None = None
    split: SplitProfile | None = None
    fast_profile: FastProfile = Field(default_factory=FastProfile)
    deep_analysis: DeepAnalysis | None = None
    leakage: LeakageAudit = Field(default_factory=LeakageAudit)

    # Per-feature map
    feature_map: dict[str, FeatureInfo] = Field(default_factory=dict)

    # Fingerprint
    fingerprint: DatasetFingerprint = Field(default_factory=DatasetFingerprint)

    # Metric that has been selected (set after Phase 2)
    primary_metric: MetricName | None = None
    metric_direction: MetricDirection | None = None
    metric_locked: bool = False

    def lock_metric(self, metric: MetricName, direction: MetricDirection) -> None:
        from automl.core.exceptions import MetricLockedError
        if self.metric_locked:
            raise MetricLockedError()
        self.primary_metric = metric
        self.metric_direction = direction
        self.metric_locked = True


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


class ResourceBudget(BaseModel):
    """Resource allocation for a single experiment."""

    ram_gb: float = 4.0
    max_threads: int = 4
    timeout_seconds: float = 300.0
    estimated_ram_gb: float | None = None
    estimated_runtime_seconds: float | None = None


class ResourceUsageRecord(BaseModel):
    """Observed resource usage of a completed experiment."""

    peak_ram_gb: float = 0.0
    runtime_seconds: float = 0.0
    cpu_percent_mean: float = 0.0
    n_threads_observed: int = 0


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    """
    A single testable hypothesis produced by the planner.
    Every field is mandatory — the schema is the contract between planner and selector.
    """

    hypothesis_id: str = Field(default_factory=lambda: _new_id("HYP"))
    hypothesis: str
    evidence: list[str] = Field(default_factory=list)
    experiment_type: OperatorType
    changes: list[str] = Field(default_factory=list)
    candidate_pipeline: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Any] = Field(default_factory=dict)
    expected_gain: float = 0.0
    expected_cost: float = 0.0          # normalised 0–1
    information_value: float = 0.0      # exploratory value (0-1)
    failure_risk: float = 0.0           # expected failure probability (0-1)
    risk: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    success_condition: str = ""

    # Computed by selector
    utility_score: float = 0.0
    source: str = "rules"               # "rules" | "slm" | "meta_prior"


class HypothesisSet(BaseModel):
    """Collection of ranked hypotheses produced in one planning round."""

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    planning_round: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)

    def best(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.utility_score)

    def by_operator(self, op: OperatorType) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.experiment_type == op]


# ---------------------------------------------------------------------------
# Experiment Spec
# ---------------------------------------------------------------------------


class ExperimentSpec(BaseModel):
    """
    Blueprint for a single experiment.
    Produced by the planner/selector, consumed by the compiler and runner.
    No LLM-generated Python code ever appears here — only declarative config.
    """

    experiment_id: str = Field(default_factory=lambda: _new_id("EXP"))
    parent_id: str | None = None         # None only for BASELINE experiments
    hypothesis_id: str | None = None

    # What this experiment tests
    hypothesis: str = ""
    operator: OperatorType = OperatorType.BASELINE

    # Pipeline definition (declarative)
    model_name: str = ""                 # must exist in ComponentRegistry
    model_family: ModelFamily = ModelFamily.GRADIENT_BOOSTING
    preprocessing: list[dict[str, Any]] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)   # feature transform names
    feature_selection: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    # Execution parameters
    fidelity: Fidelity = Fidelity.CHEAP
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)
    random_seed: int = 42

    # Expected outcomes
    success_condition: str = ""
    expected_gain: float = 0.0

    # Repair tracking
    repair_count: int = 0
    repair_history: list[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def validate_parent(self) -> ExperimentSpec:
        if self.operator != OperatorType.BASELINE and self.parent_id is None:
            raise ValueError(
                f"Non-baseline experiment '{self.experiment_id}' must have a parent_id. "
                "Every experiment must derive from a parent in the search tree."
            )
        return self

    def config_hash(self) -> str:
        """Stable hash of the pipeline configuration for deduplication."""
        config = {
            "model": self.model_name,
            "preprocessing": self.preprocessing,
            "features": sorted(self.features),
            "hyperparameters": self.hyperparameters,
        }
        return hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Experiment Result
# ---------------------------------------------------------------------------


class ExperimentResult(BaseModel):
    """
    Complete result of a single experiment execution.
    Written by the runner, read by the critic, attributed by the attribution engine.
    """

    experiment_id: str
    parent_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    operator: OperatorType = OperatorType.BASELINE  # preserved for attribution engine (Phase 8)

    # Error Tracking
    failure_type: FailureType | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    # Scores
    cv_score_mean: float | None = None
    cv_score_std: float | None = None
    fold_scores: list[float] = Field(default_factory=list)
    train_score: float | None = None    # for overfitting detection

    # Resource usage
    resource_usage: ResourceUsageRecord = Field(default_factory=ResourceUsageRecord)

    # Execution details
    fidelity: Fidelity = Fidelity.CHEAP
    n_features_used: int = 0
    model_size_mb: float | None = None
    random_seeds: dict[str, int] = Field(default_factory=dict)

    # Attribution
    score_delta: float | None = None    # vs parent
    runtime_delta: float | None = None
    ram_delta: float | None = None

    # Artifacts
    model_artifact_path: Path | None = None
    oof_predictions_path: Path | None = None

    # Critic output
    is_stable: bool = True
    is_overfitting: bool = False
    critic_flags: list[str] = Field(default_factory=list)
    reflection: str = ""                # SLM reflection text

    # Timestamps
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def runtime_seconds(self) -> float:
        return self.resource_usage.runtime_seconds

    @property
    def peak_ram_gb(self) -> float:
        return self.resource_usage.peak_ram_gb


# ---------------------------------------------------------------------------
# Failure Report
# ---------------------------------------------------------------------------


class FailureReport(BaseModel):
    """Structured record of a failed experiment and repair attempts."""

    experiment_id: str
    failure_type: FailureType = FailureType.UNKNOWN
    error_message: str = ""
    traceback: str = ""
    original_spec: ExperimentSpec | None = None
    repaired_spec: ExperimentSpec | None = None
    repair_applied: str | None = None
    repair_successful: bool = False
    repair_count: int = 0
    failed_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Critic Report
# ---------------------------------------------------------------------------


class CriticReport(BaseModel):
    """Output of the Numerical Critic and LLM Reflection."""

    experiment_id: str
    is_stable: bool = True
    is_overfitting: bool = False
    is_memory_heavy: bool = False
    is_slow: bool = False
    overfit_ratio: float = 0.0
    critic_flags: list[str] = Field(default_factory=list)
    reflection: str = ""
    analyzed_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Search Node & Tree
# ---------------------------------------------------------------------------


class SearchNode(BaseModel):
    """A single node in the search tree."""

    node_id: str = Field(default_factory=lambda: _new_id("NODE"))
    experiment_id: str
    parent_id: str | None = None
    operator: OperatorType = OperatorType.BASELINE
    fidelity: Fidelity = Fidelity.CHEAP
    status: BranchStatus = BranchStatus.ACTIVE

    # Scores
    score: float | None = None
    score_std: float | None = None
    score_delta: float | None = None    # vs parent

    # Resources
    runtime_seconds: float = 0.0
    peak_ram_gb: float = 0.0

    # Children
    children: list[str] = Field(default_factory=list)  # list of node_ids

    # Metadata
    hypothesis_summary: str = ""
    depth: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class SearchState(BaseModel):
    """
    Complete mutable state of the autonomous search loop.
    Serialized to SQLite after every meaningful event for crash recovery.
    """

    run_id: str
    phase: SearchPhase = SearchPhase.INITIALIZING
    saturation_state: SaturationState = SaturationState.SEARCHING

    # Incumbent
    incumbent_experiment_id: str | None = None
    incumbent_score: float | None = None
    incumbent_score_std: float | None = None

    # History
    completed_experiments: list[str] = Field(default_factory=list)
    failed_experiments: list[str] = Field(default_factory=list)
    pruned_branches: list[str] = Field(default_factory=list)

    # Search control
    explore_ratio: float = 0.5
    planning_round: int = 0
    consecutive_no_gain: int = 0

    # Budget
    budget_total_seconds: float = 3600.0
    budget_elapsed_seconds: float = 0.0

    # Ensemble candidates
    ensemble_candidates: list[str] = Field(default_factory=list)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    last_updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def budget_remaining_seconds(self) -> float:
        return max(0.0, self.budget_total_seconds - self.budget_elapsed_seconds)

    @property
    def budget_fraction_remaining(self) -> float:
        if self.budget_total_seconds <= 0:
            return 0.0
        return self.budget_remaining_seconds / self.budget_total_seconds

    @property
    def total_experiments(self) -> int:
        return len(self.completed_experiments) + len(self.failed_experiments)


# ---------------------------------------------------------------------------
# Final Model
# ---------------------------------------------------------------------------


class FinalMetrics(BaseModel):
    """Test-set metrics — populated exactly once after search is locked."""

    test_score: float
    test_metric: MetricName
    cv_score_mean: float
    cv_score_std: float
    n_test_samples: int = 0
    evaluated_at: datetime = Field(default_factory=datetime.now)


class FinalModel(BaseModel):
    """Complete description of the locked final model."""

    run_id: str
    experiment_id: str
    model_name: str
    model_family: ModelFamily

    # Pipeline config (declarative, not executable Python)
    pipeline_config: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    features_used: list[str] = Field(default_factory=list)
    n_features: int = 0

    # Metrics
    cv_score_mean: float = 0.0
    cv_score_std: float = 0.0
    final_metrics: FinalMetrics | None = None

    # Ensemble info
    is_ensemble: bool = False
    ensemble_members: list[str] = Field(default_factory=list)
    ensemble_weights: list[float] = Field(default_factory=list)

    # Reproducibility
    random_seed: int = 42
    python_version: str = ""
    library_versions: dict[str, str] = Field(default_factory=dict)
    dataset_hash: str = ""
    config_hash: str = ""
    hardware_info: dict[str, Any] = Field(default_factory=dict)

    # Artifact paths
    model_path: Path | None = None
    preprocessing_path: Path | None = None
    metadata_path: Path | None = None
    feature_schema_path: Path | None = None
    inference_script_path: Path | None = None

    locked_at: datetime = Field(default_factory=datetime.now)
