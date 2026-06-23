"""Model registry"""

from __future__ import annotations

from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.modeling.config import ModelConfig

TaskType = Literal["growth_class", "max_od", "growth_rate"]

_REGISTRY: dict[str, dict[str, type[BaseEstimator]]] = {
    "linear": {
        "classification": LogisticRegression,
        "regression": Ridge,
    },
    "random_forest": {
        "classification": RandomForestClassifier,
        "regression": RandomForestRegressor,
    },
    "xgboost": {
        "classification": XGBClassifier,
        "regression": XGBRegressor,
    },
    "mlp": {
        "classification": MLPClassifier,
        "regression": MLPRegressor,
    },
    "dummy": {
        "classification": DummyClassifier,
        "regression": DummyRegressor,
    },
}


def _task_mode(task: TaskType) -> str:
    return "classification" if task == "growth_class" else "regression"


def build_model(
    cfg: ModelConfig,
    task: TaskType,
    seed: int = 42,
) -> BaseEstimator:
    """Instantiate a model from config + task type.

    Parameters
    ----------
    cfg:
        Model YAML config with ``type``, ``params``, and optional
        ``task_overrides``.
    task:
        One of ``growth_class``, ``max_od``, ``growth_rate``.
    seed:
        Random seed injected into the model where supported.

    Returns
    -------
    Unfitted sklearn-compatible estimator.
    """
    mode = _task_mode(task)
    entry = _REGISTRY.get(cfg.type)
    if entry is None:
        raise ValueError(f"Unknown model type '{cfg.type}'. Available: {list(_REGISTRY)}")

    cls = entry[mode]

    params: dict[str, Any] = {**cfg.params}
    if task in cfg.task_overrides:
        params.update(cfg.task_overrides[task])

    # inject seed / random_state where the estimator accepts it
    _seed_keys = {"random_state", "seed"}
    if not _seed_keys & set(params):
        import inspect
        sig = inspect.signature(cls)
        if "random_state" in sig.parameters:
            params["random_state"] = seed
        elif "seed" in sig.parameters:
            params["seed"] = seed

    # XGBoost classification defaults
    if cfg.type == "xgboost" and mode == "classification":
        params.setdefault("eval_metric", "logloss")

    return cls(**params)
