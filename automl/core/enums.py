"""
MicroAutoML-Agent — Core Enumerations

All enums used across the system are defined here. Import from this module only;
never define task/status/type enums inline in other modules to prevent drift.
"""

from __future__ import annotations

from enum import Enum, auto


# ---------------------------------------------------------------------------
# Task & structure
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """The supervised learning task the system is solving."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class DatasetStructure(str, Enum):
    """High-level dataset structure that drives split strategy selection."""

    IID = "iid"
    GROUPED = "grouped"
    TEMPORAL = "temporal"


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class SplitType(str, Enum):
    """The cross-validation / holdout strategy selected by the split strategist."""

    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    GROUP_KFOLD = "group_kfold"
    TIME_SERIES_SPLIT = "time_series_split"
    HOLDOUT = "holdout"


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


class Fidelity(str, Enum):
    """Multi-fidelity execution level."""

    SMOKE = "smoke"       # 5-10% data, sanity only
    CHEAP = "cheap"       # 20-30% data, 1 split
    MEDIUM = "medium"     # 50-70% data, 3-fold CV
    FULL = "full"         # 100% data, strong CV


class ExperimentStatus(str, Enum):
    """Lifecycle status of a single experiment."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REPAIRED = "repaired"
    PRUNED = "pruned"
    EARLY_STOPPED = "early_stopped"
    TIMEOUT = "timeout"


class OperatorType(str, Enum):
    """Search operator that generated this experiment."""

    MODEL_CHANGE = "model_change"
    FEATURE_ENGINEERING = "feature_engineering"
    PREPROCESSING_CHANGE = "preprocessing_change"
    FEATURE_SELECTION = "feature_selection"
    HPO = "hpo"
    ENSEMBLE = "ensemble"
    REPAIR = "repair"
    RETEST = "retest"
    BASELINE = "baseline"


# ---------------------------------------------------------------------------
# Event Architecture (Audit Trail)
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Discrete system transitions logged for the Audit Trail."""

    DATA_LOADED = "data_loaded"
    PROFILE_CREATED = "profile_created"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    REPAIR_APPLIED = "repair_applied"
    SEARCH_UPDATED = "search_updated"
    MODEL_PROMOTED = "model_promoted"
    SEARCH_LOCKED = "search_locked"
    FINAL_TEST_COMPLETED = "final_test_completed"
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    METRIC_LOCKED = "metric_locked"
    BASELINES_STARTED = "baselines_started"
    BASELINES_COMPLETED = "baselines_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


class FailureType(str, Enum):
    """Deterministic classification of experiment failures."""

    SYNTAX = "syntax"
    IMPORT = "import"
    DATA = "data"
    SHAPE = "shape"
    TYPE = "type"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    NUMERICAL = "numerical"
    GENERALIZATION = "generalization"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class BranchStatus(str, Enum):
    """Status of a branch in the search tree."""

    ACTIVE = "active"
    PROMISING = "promising"
    PRUNED = "pruned"
    FAILED = "failed"
    SATURATED = "saturated"
    COMPLETED = "completed"


class SearchPhase(str, Enum):
    """High-level phase of the autonomous search loop."""

    INITIALIZING = "initializing"
    PROFILING = "profiling"
    BASELINING = "baselining"
    PLANNING = "planning"
    SEARCHING = "searching"
    SATURATING = "saturating"
    LOCKED = "locked"
    ENSEMBLING = "ensembling"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"


class ExploitExplore(str, Enum):
    """The controller's current mode."""

    EXPLOIT = "exploit"
    EXPLORE = "explore"
    ENGINEER = "engineer"


# ---------------------------------------------------------------------------
# Feature / transform
# ---------------------------------------------------------------------------


class FeatureDtype(str, Enum):
    """Coarse feature type as detected by the auditor."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"
    CONSTANT = "constant"
    ID = "id"
    UNKNOWN = "unknown"


class TransformType(str, Enum):
    """Category of feature transformation."""

    NUMERICAL_TRANSFORM = "numerical_transform"
    ENCODING = "encoding"
    DATETIME_EXTRACTION = "datetime_extraction"
    INTERACTION = "interaction"
    AGGREGATION = "aggregation"
    SELECTION = "selection"
    IMPUTATION = "imputation"
    SCALING = "scaling"


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


class LeakageSeverity(str, Enum):
    """Severity of a detected data leakage signal."""

    NONE = "none"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Model family
# ---------------------------------------------------------------------------


class ModelFamily(str, Enum):
    """Broad model family for routing and meta-learning."""

    DUMMY = "dummy"
    LINEAR = "linear"
    TREE_ENSEMBLE = "tree_ensemble"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL = "neural"
    FOUNDATION = "foundation"
    ENSEMBLE = "ensemble"


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


class ResourceDecision(str, Enum):
    """Decision returned by the resource governor."""

    ALLOWED = "allowed"
    DENIED_MEMORY = "denied_memory"
    DENIED_TIME = "denied_time"
    DENIED_THREADS = "denied_threads"
    ALLOWED_DEGRADED = "allowed_degraded"  # allowed with reduced threads/settings


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class MetricName(str, Enum):
    """Supported evaluation metrics."""

    # Classification
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    F1 = "f1"
    MACRO_F1 = "macro_f1"
    ACCURACY = "accuracy"
    LOG_LOSS = "log_loss"
    PRECISION = "precision"
    RECALL = "recall"
    BRIER_SCORE = "brier_score"

    # Regression
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"


class MetricDirection(str, Enum):
    """Whether higher or lower is better for a given metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


class SaturationState(str, Enum):
    """State of the saturation detector."""

    SEARCHING = "searching"
    PLATEAU_DETECTED = "plateau_detected"
    SATURATED = "saturated"
    BUDGET_EXHAUSTED = "budget_exhausted"
