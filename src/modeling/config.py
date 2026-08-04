"""Pydantic config models for dataset, model, and experiment YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

class MergeConfig(BaseModel):
    """Column-name mapping for the curves <-> strain-metadata join."""

    left: str = "strain_name"
    right: str = "strain"


class DatasetConfig(BaseModel):
    """Config for ``create_modeling_dataset.py``."""

    model_config = ConfigDict(extra="forbid")

    curves_file: str | list[str]

    @field_validator("curves_file", mode="before")
    @classmethod
    def _ensure_curves_file_list(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            return v
        if not v:
            raise ValueError("curves_file must be a non-empty string or list of strings")
        return v

    strain_metadata_file: str | None = None
    output_name: str
    grain: Literal["per_well", "averaged"] = "per_well"
    merge: MergeConfig = MergeConfig()

    features: dict[str, Any] = {}
    targets: list[str] = ["growth_class", "max_od", "growth_rate"]


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Config for a single model type + hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linear", "random_forest", "xgboost", "mlp", "dummy"]
    params: dict[str, Any] = {}
    task_overrides: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

class CVConfig(BaseModel):
    strategy: list[str] | str = "loso"
    n_repeats: int = 1
    group_on: str = "strain_name"

    @field_validator("strategy", mode="before")
    @classmethod
    def _wrap_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class InterpretConfig(BaseModel):
    shap: bool = True
    group_importance: bool = True
    shap_class: Literal["positive", "negative"] = "positive"


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "microbenet"
    entity: str | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment config that references dataset + model configs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: str
    model: str
    task: Literal["growth_class", "max_od", "growth_rate"]
    cv: CVConfig = CVConfig()
    class_weight: str | dict[Any, float] | None = None
    metrics: list[str] = []
    interpret: InterpretConfig = InterpretConfig()
    seed: int = 42
    output_dir: str = "reports"
    wandb: WandbConfig = WandbConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig(**load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**load_yaml(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig(**load_yaml(path))


def resolve_experiment(path: str | Path, root: Path | None = None):
    """Load an experiment config and resolve its nested dataset + model configs.

    Returns ``(experiment_cfg, dataset_cfg, model_cfg)``.
    """
    if root is None:
        root = Path(path).resolve().parents[1]
        # climb from config/experiments/<file>.yaml -> project root
        # but if path is already absolute, fall back to cwd
        if not (root / "src").exists():
            root = Path.cwd()

    exp_cfg = load_experiment_config(path)
    ds_cfg = load_dataset_config(root / exp_cfg.dataset)
    mdl_cfg = load_model_config(root / exp_cfg.model)
    return exp_cfg, ds_cfg, mdl_cfg
