"""Run a modeling experiment end-to-end: CV, evaluation, interpretability.

Usage:
    python scripts/run_experiment.py config/experiments/maxod_aero_anaero.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.config import resolve_experiment  # noqa: E402
from src.modeling.cv import get_fold_group_label, make_splits  # noqa: E402
from src.modeling.evaluate import (  # noqa: E402
    aggregate_metrics,
    evaluate_fold,
    per_group_metrics,
)
from src.modeling.features import (  # noqa: E402
    build_feature_group_index,
    build_feature_pipeline,
    get_feature_names,
    merge_external_features,
    warn_loso_onehot,
)
from src.modeling.interpret import run_interpretation  # noqa: E402
from src.modeling.models import build_model, gpu_available  # noqa: E402


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
        colorize=sys.stderr.isatty(),
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
            colorize=False,
            mode="w",
        )


def _filter_for_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """For growth_rate regression, keep only growing wells."""
    if task == "growth_rate":
        if "growth_class" in df.columns:
            n_before = len(df)
            df = df[df["growth_class"] == 1].copy()
            logger.info(
                "growth_rate task: filtered to growing wells ({} -> {} rows)",
                n_before, len(df),
            )
    return df


def run_cv_loop(
    df: pd.DataFrame,
    exp_cfg,
    ds_cfg,
    mdl_cfg,
) -> tuple[list[dict], pd.DataFrame]:
    """Run the full CV loop: featurize per fold -> fit -> predict -> evaluate.

    Returns fold metrics and a predictions DataFrame.
    """
    task = exp_cfg.task
    all_fold_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []

    for strategy in exp_cfg.cv.strategy:
        warn_loso_onehot(ds_cfg, strategy)

        fold_num = 0
        for train_idx, test_idx in make_splits(
            strategy,
            df,
            y=df[task].values,
            seed=exp_cfg.seed,
            group_on=exp_cfg.cv.group_on,
        ):
            fold_num += 1
            fold_label = get_fold_group_label(strategy, df, test_idx)
            logger.info("--- {}: fold {} (held-out: {}) ---", strategy, fold_num, fold_label)

            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            # Fit feature pipeline on train only (leakage-safe)
            ct, feature_cols, fgm = build_feature_pipeline(ds_cfg, df_train, root=ROOT)
            logger.info(" features: {}", feature_cols)
            ct.fit(df_train)
            X_train = ct.transform(df_train)
            X_test = ct.transform(df_test)
            y_train = df_train[task].values
            y_test = df_test[task].values

            # Build and fit model
            model = build_model(mdl_cfg, task, seed=exp_cfg.seed)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_prob = None
            if task == "growth_class" and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            # Evaluate
            metrics = evaluate_fold(y_test, y_pred, y_prob, task, exp_cfg.metrics or None)
            metrics["strategy"] = strategy
            metrics["fold"] = fold_num
            metrics["fold_label"] = fold_label
            metrics["n_train"] = len(train_idx)
            metrics["n_test"] = len(test_idx)
            all_fold_metrics.append(metrics)

            logger.info("  metrics: {}", {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})

            # Collect predictions for per-group analysis
            pred_df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "strategy": strategy,
                "fold": fold_num,
                "fold_label": fold_label,
            })
            if y_prob is not None:
                pred_df["y_prob"] = y_prob
            if "strain_name" in df.columns:
                pred_df["strain_name"] = df_test["strain_name"].values
            if "media_name" in df.columns:
                pred_df["media_name"] = df_test["media_name"].values
            if "atmosphere" in df.columns:
                pred_df["atmosphere"] = df_test["atmosphere"].values
            all_predictions.append(pred_df)

    return all_fold_metrics, pd.concat(all_predictions, ignore_index=True)


def run_interpretability(
    df: pd.DataFrame,
    exp_cfg,
    ds_cfg,
    mdl_cfg,
    output_dir: Path,
) -> None:
    """Refit on all data and run SHAP analysis."""
    task = exp_cfg.task

    ct, _, fgm = build_feature_pipeline(ds_cfg, df, root=ROOT)
    ct.fit(df)
    X = ct.transform(df)
    y = df[task].values
    feature_names = get_feature_names(ct)

    model = build_model(mdl_cfg, task, seed=exp_cfg.seed)
    model.fit(X, y)

    group_index = build_feature_group_index(feature_names, fgm)

    run_interpretation(
        model, X, feature_names, group_index, output_dir,
        do_group_importance=exp_cfg.interpret.group_importance,
        shap_class=exp_cfg.interpret.shap_class,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to experiment YAML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    exp_cfg, ds_cfg, mdl_cfg = resolve_experiment(args.config, root=ROOT)

    # Output directory
    output_dir = ROOT / exp_cfg.output_dir / exp_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "experiment.log"
    setup_logging(args.log_level, log_file=log_path)

    logger.info("experiment: {}", exp_cfg.name)
    logger.info("task: {}", exp_cfg.task)
    logger.info("model: {} ({})", mdl_cfg.type, mdl_cfg.params)
    if mdl_cfg.type == "xgboost":
        if gpu_available():
            logger.info("GPU available, XGBoost will train on CUDA")
        else:
            logger.info("No GPU available, XGBoost will train on CPU")
    logger.info("CV strategies: {}", exp_cfg.cv.strategy)
    logger.info("output: {}", output_dir)

    # Save config snapshot
    config_snapshot = {
        "experiment": exp_cfg.model_dump(),
        "dataset": ds_cfg.model_dump(),
        "model": mdl_cfg.model_dump(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

    # Load data
    interim_path = ROOT / "data" / "interim" / ds_cfg.output_name
    logger.info("loading modeling dataset: {}", interim_path)
    df = pd.read_csv(interim_path)
    logger.info("dataset shape: {} rows x {} cols", *df.shape)

    df = _filter_for_task(df, exp_cfg.task)

    # Merge external feature tables (path: specs) before splitting
    df = merge_external_features(ds_cfg, df, root=ROOT)

    # --- CV loop ---
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION")
    logger.info("=" * 60)

    fold_metrics, predictions_df = run_cv_loop(df, exp_cfg, ds_cfg, mdl_cfg)

    # Aggregate metrics per strategy
    for strategy in exp_cfg.cv.strategy:
        strat_folds = [m for m in fold_metrics if m.get("strategy") == strategy]
        agg = aggregate_metrics(strat_folds)
        logger.info("--- {} aggregate ---", strategy)
        for k, v in agg.items():
            logger.info("  {}: {:.4f} +/- {:.4f}", k, v["mean"], v["std"])

    # Save fold-level metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    # Save aggregate
    all_agg: dict[str, dict] = {}
    for strategy in exp_cfg.cv.strategy:
        strat_folds = [m for m in fold_metrics if m.get("strategy") == strategy]
        all_agg[strategy] = aggregate_metrics(strat_folds)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_agg, f, indent=2, default=str)

    # Save predictions
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    # Per-strain and per-media breakdowns
    for group_col in ["strain_name", "media_name"]:
        if group_col in predictions_df.columns:
            pg = per_group_metrics(predictions_df, group_col, exp_cfg.task, exp_cfg.metrics or None)
            pg.to_csv(output_dir / f"metrics_by_{group_col}.csv")
            logger.info("per-{} metrics:\n{}", group_col, pg.to_string())

    # --- Interpretability ---
    if exp_cfg.interpret.shap:
        logger.info("=" * 60)
        logger.info("INTERPRETABILITY (refit on all data)")
        logger.info("=" * 60)
        run_interpretability(df, exp_cfg, ds_cfg, mdl_cfg, output_dir)

    logger.success("experiment '{}' complete -> {}", exp_cfg.name, output_dir)
    logger.success("log -> {}", log_path)


if __name__ == "__main__":
    main()
