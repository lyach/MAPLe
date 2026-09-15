"""Pydantic config models for dataset, model, and experiment YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Curves dataset config (create_modeling_dataset.py)
# ---------------------------------------------------------------------------

class MergeConfig(BaseModel):
    """Column-name mapping for the curves <-> strain-metadata join."""

    left: str = "strain_name"
    right: str = "strain"


class CurvesDatasetConfig(BaseModel):
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


# ---------------------------------------------------------------------------
# Modeling dataset config (run_experiment.py)
# ---------------------------------------------------------------------------

class RowFilterConfig(BaseModel):
    """Row-level include/exclude rules, applied by column value.

    Each entry is ``column: [value, ...]``. ``exclude`` drops rows whose value
    in that column is in the list; ``include`` keeps only those rows.

    Rows are filtered AFTER the split-map row-count check, so a filtered run
    reuses the unfiltered split map.
    """

    model_config = ConfigDict(extra="forbid")

    include: dict[str, list[Any]] = {}
    exclude: dict[str, list[Any]] = {}


class DatasetConfig(BaseModel):
    """Config for experiment featurization: path + features + targets."""

    model_config = ConfigDict(extra="forbid")

    modeling_dataset_path: str
    features: dict[str, Any] = {}
    targets: list[str] = ["growth_class", "max_od", "growth_rate"]
    filters: RowFilterConfig = RowFilterConfig()


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
    splits_file: str | None = None
    split_on: str | None = None

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


# Dataset keys that may appear inline in an experiment config
_INLINE_DATASET_KEYS = frozenset(DatasetConfig.model_fields)


class ExperimentConfig(BaseModel):
    """Top-level experiment config.

    The dataset is given either inline -- flat ``modeling_dataset_path`` /
    ``features`` / ``targets`` / ``filters`` keys, -- or as a path to a
    standalone dataset YAML (``dataset: config/dataset/...``, legacy).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: str | DatasetConfig
    model: str
    task: Literal["growth_class", "max_od", "growth_rate"]
    cv: CVConfig = CVConfig()
    class_weight: str | dict[Any, float] | None = None
    metrics: list[str] = []
    interpret: InterpretConfig = InterpretConfig()
    seed: int = 42
    output_dir: str = "reports"
    wandb: WandbConfig = WandbConfig()

    @model_validator(mode="before")
    @classmethod
    def _lift_inline_dataset(cls, data: Any) -> Any:
        """Fold flat dataset keys into ``dataset`` as an inline DatasetConfig.
        """
        if not isinstance(data, dict):
            return data
        inline = {k: v for k, v in data.items() if k in _INLINE_DATASET_KEYS}
        if not inline:
            return data  # legacy form: `dataset:` must be a path

        if data.get("dataset") is not None:
            raise ValueError(
                f"config sets both `dataset: {data['dataset']}` (a path to a standalone "
                f"dataset YAML) and inline dataset keys {sorted(inline)}. Use one or the "
                "other: drop the `dataset:` line to keep the inline block, or drop the "
                "inline keys to keep the referenced file."
            )
        try:
            dataset = DatasetConfig(**inline)
        except ValidationError as exc:
            raise ValueError(f"invalid inline dataset block:\n{exc}") from exc

        rest = {k: v for k, v in data.items() if k not in _INLINE_DATASET_KEYS}
        rest["dataset"] = dataset
        return rest


# ---------------------------------------------------------------------------
# Split config
# ---------------------------------------------------------------------------

class SplitConfig(BaseModel):
    """Config for ``create_splits.py`` (precompute reusable CV splits)."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    split_on_column: str
    target: str
    cv: CVConfig = CVConfig()
    n_splits: int = 5
    seed: int = 42
    output_format: Literal["csv", "parquet"] = "csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig(**load_yaml(path))


def load_curves_dataset_config(path: str | Path) -> CurvesDatasetConfig:
    return CurvesDatasetConfig(**load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**load_yaml(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    try:
        return ExperimentConfig(**load_yaml(path))
    except ValidationError as exc:
        raise ValueError(f"invalid experiment config {path}:\n{exc}") from exc


def load_split_config(path: str | Path) -> SplitConfig:
    return SplitConfig(**load_yaml(path))


def _find_root(start: Path) -> Path:
    """Ascend from ``start`` to the directory containing ``pyproject.toml``."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def resolve_experiment(path: str | Path, root: Path | None = None):
    """Load an experiment config and resolve its dataset + model configs.

    Returns ``(experiment_cfg, dataset_cfg, model_cfg)``. The dataset config is
    taken from the experiment's inline block, or loaded from the path in
    ``dataset:`` for configs that still use a standalone dataset YAML.
    """
    if root is None:
        root = _find_root(Path(path).resolve())

    exp_cfg = load_experiment_config(path)
    if isinstance(exp_cfg.dataset, DatasetConfig):
        ds_cfg = exp_cfg.dataset
    else:
        ds_cfg = load_dataset_config(root / exp_cfg.dataset)
    mdl_cfg = load_model_config(root / exp_cfg.model)
    return exp_cfg, ds_cfg, mdl_cfg
