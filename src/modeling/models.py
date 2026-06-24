"""Model registry"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.modeling.config import ModelConfig

TaskType = Literal["growth_class", "max_od", "growth_rate"]


class _XGBClassifierGPUTrain(XGBClassifier):
    """Train on GPU when configured, then use CPU for inference."""
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        _switch_xgb_to_cpu_inference(self)
        return self


class _XGBRegressorGPUTrain(XGBRegressor):
    """Train on GPU when configured, then use CPU for inference."""

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        _switch_xgb_to_cpu_inference(self)
        return self


def _switch_xgb_to_cpu_inference(model: XGBClassifier | XGBRegressor) -> None:
    params = model.get_params()
    device = params.get("device", "cpu")
    if device not in (None, "cpu"):
        model.set_params(device="cpu")
        return
    if params.get("tree_method") in {"gpu_hist", "cuda_hist"}:
        model.set_params(tree_method="hist")

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
        "classification": _XGBClassifierGPUTrain,
        "regression": _XGBRegressorGPUTrain,
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


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return True when a CUDA GPU is present and XGBoost was built with CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    try:
        import xgboost as xgb
        cuda_version = xgb.build_info().get("CUDA_VERSION")
        return bool(cuda_version)
    except (ImportError, AttributeError, TypeError):
        return False


def _gpu_device_configured(params: dict[str, Any]) -> bool:
    device = params.get("device")
    if device is not None and device != "cpu":
        return True
    return params.get("tree_method") in {"gpu_hist", "cuda_hist"}


def _apply_xgboost_gpu(params: dict[str, Any]) -> None:
    """Set XGBoost GPU params in-place when a GPU is available."""
    if _gpu_device_configured(params) or not gpu_available():
        return
    import xgboost as xgb
    major = int(xgb.__version__.split(".")[0])
    if major >= 2:
        params.setdefault("device", "cuda")
        params.setdefault("tree_method", "hist")
    else:
        params.setdefault("tree_method", "gpu_hist")


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
    if cfg.type == "xgboost":
        if mode == "classification":
            params.setdefault("eval_metric", "logloss")
        _apply_xgboost_gpu(params)

    return cls(**params)
