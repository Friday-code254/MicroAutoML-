"""
MicroAutoML-Agent — Run Configuration

`RunConfig` is the single configuration object passed to every subsystem.
It is loaded from a YAML file at startup and validated by Pydantic.
Never hard-code resource limits or paths inside subsystem modules —
always read from the config object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SLMConfig(BaseModel):
    """Configuration for the optional SLM planner backend."""

    enabled: bool = True
    provider: str = "ollama"          # "gemini" | "ollama" | "openai"
    model: str = "qwen3:4b-instruct"
    endpoint: str = "http://localhost:11434"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 600.0
    max_retries: int = 2

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class ResourceConfig(BaseModel):
    """Hard resource limits for the entire AutoML run."""

    # Memory
    total_ram_gb: float = Field(default=5.0, gt=0, description="Hard RAM ceiling in GB")
    os_reserved_gb: float = Field(default=1.5, gt=0, description="RAM reserved for OS/background")
    model_ram_estimate_multiplier: float = Field(
        default=1.5, gt=1.0,
        description="Safety multiplier applied to model RAM estimates"
    )

    # CPU / parallelism
    max_threads: int = Field(default=4, gt=0, description="Hard cap on OMP/MKL/n_jobs threads")
    n_jobs: int = Field(default=2, gt=0, description="Default n_jobs for sklearn estimators")

    # Time
    per_experiment_timeout_seconds: float = Field(
        default=300.0, gt=0,
        description="Wall-clock timeout per individual experiment"
    )

    # Disk
    max_disk_gb: float = Field(default=10.0, gt=0)

    @property
    def available_ram_gb(self) -> float:
        return self.total_ram_gb - self.os_reserved_gb


class FidelityConfig(BaseModel):
    """Data fractions and CV folds for each fidelity level."""

    smoke_fraction: float = Field(default=0.08, gt=0, le=1.0)
    cheap_fraction: float = Field(default=0.25, gt=0, le=1.0)
    medium_fraction: float = Field(default=0.60, gt=0, le=1.0)
    full_fraction: float = Field(default=1.0, gt=0, le=1.0)

    smoke_n_estimators: int = Field(default=20, gt=0)
    cheap_n_estimators: int = Field(default=50, gt=0)
    medium_n_estimators: int = Field(default=150, gt=0)
    full_n_estimators: int = Field(default=300, gt=0)

    cheap_cv_folds: int = Field(default=1, gt=0)
    medium_cv_folds: int = Field(default=3, gt=0)
    full_cv_folds: int = Field(default=5, gt=0)


class SearchConfig(BaseModel):
    """Parameters governing the autonomous search loop."""

    # Budget
    time_budget_seconds: float = Field(default=3600.0, gt=0)

    # Exploration / exploitation
    initial_explore_ratio: float = Field(default=0.50, ge=0, le=1.0)
    final_explore_ratio: float = Field(default=0.10, ge=0, le=1.0)
    exploit_adjust_delta: float = Field(default=0.05, ge=0)
    explore_adjust_delta: float = Field(default=0.05, ge=0)

    # Saturation
    plateau_window: int = Field(
        default=5, gt=0,
        description="Number of consecutive experiments with < min_gain to trigger saturation"
    )
    min_gain_threshold: float = Field(
        default=0.001, gt=0,
        description="Minimum score improvement considered meaningful"
    )

    # Repair
    max_repairs_per_experiment: int = Field(default=2, ge=0)

    # Promotion gates (how many survive each fidelity level)
    smoke_survivors: int = Field(default=20)
    cheap_survivors: int = Field(default=10)
    medium_survivors: int = Field(default=5)
    full_survivors: int = Field(default=3)

    # HPO
    hpo_n_trials: int = Field(default=30, gt=0)
    hpo_timeout_seconds: float = Field(default=600.0, gt=0)

    # Ensemble
    ensemble_max_size: int = Field(default=5, gt=0)
    ensemble_diversity_threshold: float = Field(
        default=0.85, gt=0, le=1.0,
        description="OOF prediction correlation above which two models are NOT considered complementary"
    )


class RunConfig(BaseModel):
    """
    Top-level configuration for a single MicroAutoML run.

    Loaded from a YAML file (cpu.yaml / benchmark.yaml / development.yaml)
    and overridden by CLI flags.
    """

    # Identity
    run_id: str | None = None          # auto-generated if None
    dataset_path: Path | None = None
    target_column: str | None = None
    output_dir: Path = Path("runs")

    # Reproducibility
    random_seed: int = 42

    # Sub-configs
    resource: ResourceConfig = Field(default_factory=ResourceConfig)
    fidelity: FidelityConfig = Field(default_factory=FidelityConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    slm: SLMConfig = Field(default_factory=SLMConfig)

    # Benchmark mode (stricter evaluation)
    benchmark_mode: bool = False

    # Verbose / debug
    verbose: bool = True
    debug: bool = False

    # Optional: force task type (override auto-detection)
    force_task_type: str | None = None

    # Optional: exclude specific model families
    excluded_models: list[str] = Field(default_factory=list)

    # Optional: force metric
    force_metric: str | None = None

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_output_dir(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("dataset_path", mode="before")
    @classmethod
    def resolve_dataset_path(cls, v: Any) -> Path | None:
        return Path(v) if v is not None else None

    @model_validator(mode="after")
    def validate_memory_budget(self) -> RunConfig:
        avail = self.resource.available_ram_gb
        if avail <= 0:
            raise ValueError(
                f"Available RAM ({avail:.2f} GB) is non-positive after OS reservation. "
                "Increase total_ram_gb or decrease os_reserved_gb."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
        """Load config from a YAML file with optional dict overrides."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if overrides:
            data.update(overrides)
        return cls.model_validate(data)

    @classmethod
    def cpu_default(cls) -> RunConfig:
        """Pre-built config for a standard 8 GB / CPU-only machine."""
        return cls(
            resource=ResourceConfig(
                total_ram_gb=5.0,
                os_reserved_gb=1.5,
                max_threads=4,
                n_jobs=2,
            ),
            search=SearchConfig(
                time_budget_seconds=3600.0,
                hpo_n_trials=30,
            ),
        )

    @classmethod
    def development(cls) -> RunConfig:
        """Fast config for development and testing (tiny budgets)."""
        return cls(
            resource=ResourceConfig(
                total_ram_gb=4.0,
                os_reserved_gb=1.0,
                max_threads=2,
                n_jobs=1,
            ),
            search=SearchConfig(
                time_budget_seconds=120.0,
                plateau_window=3,
                hpo_n_trials=5,
                smoke_survivors=5,
                cheap_survivors=3,
                medium_survivors=2,
                full_survivors=1,
            ),
            slm=SLMConfig(enabled=False),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config back to YAML (for reproducibility artifacts)."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)
