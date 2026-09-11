#!/usr/bin/env python3
"""Roll every finished run under ``reports/`` into one CSV.

So the whole experiment grid can be sorted and scanned in a spreadsheet.
Reads only artifacts ``run_experiment.py`` already writes.

One row per (experiment, CV strategy); in practice every config declares a
single strategy, so that is one row per experiment.

Usage:
    python scripts/utils/paper/compile_results_csv.py
    python scripts/utils/paper/compile_results_csv.py --core   # headline columns only
    python scripts/utils/paper/compile_results_csv.py --print  # also echo to stdout
    python scripts/utils/paper/compile_results_csv.py --out /tmp/results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

CORE_COLUMNS = [
    "experiment", "dataset", "ablation",
    "datapoints", "negatives", "prevalence",
    "pr_auc_nogrowth", "stdev_pr_auc_nogrowth",
    "mcc", "stdev_mcc", "recall_nogrowth", "auroc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reports", type=Path, default=ROOT / "reports",
                   help="Directory of run folders (default: %(default)s)")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "all_results.csv",
                   help="CSV to write (default: %(default)s)")
    p.add_argument("--core", action="store_true",
                   help="Emit only the headline columns")
    p.add_argument("--print", dest="do_print", action="store_true",
                   help="Also print the table to stdout")
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _ablation(features: dict) -> str:
    """Comma-separated feature names declared for this run.

    Derived from the config snapshot's ``dataset.features`` keys rather than
    the experiment name, so a misleading name cannot produce a misleading row.
    """
    return ", ".join(features)


def _n_groups(splits_file: str | None) -> int | None:
    """Group count for the split, from the split map's companion stats JSON."""
    if not splits_file:
        return None
    path = Path(splits_file)
    if not path.is_absolute():
        path = ROOT / path
    stats = path.with_name(f"{path.stem}_stats.json")
    if not stats.exists():
        return None
    try:
        return int(json.loads(stats.read_text()).get("n_groups_total"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _fold_count(run_dir: Path, strategy: str) -> int | None:
    path = run_dir / "fold_metrics.csv"
    if not path.exists():
        return None
    folds = pd.read_csv(path)
    if "strategy" in folds.columns:
        folds = folds[folds["strategy"] == strategy]
    return len(folds) or None


def collect_run(run_dir: Path) -> list[dict]:
    """One row per CV strategy for a single run folder."""
    cfg_path, metrics_path = run_dir / "config.yaml", run_dir / "metrics.json"
    if not cfg_path.exists() or not metrics_path.exists():
        return []

    cfg = _load_yaml(cfg_path)
    exp = cfg.get("experiment", {}) or {}
    ds = cfg.get("dataset", {}) or {}
    model = cfg.get("model", {}) or {}
    cv = exp.get("cv", {}) or {}
    metrics = json.loads(metrics_path.read_text())

    features = ds.get("features", {}) or {}
    filters = ds.get("filters", {}) or {}
    pooled = metrics.get("pooled", {}) or {}
    dataset_stem = Path(ds.get("modeling_dataset_path", "")).stem

    rows: list[dict] = []
    for strategy, vals in metrics.items():
        if strategy == "pooled":
            continue
        # Fold-aggregated entries are {"mean": x, "std": y}; skip scalars.
        agg = {k: v for k, v in vals.items() if isinstance(v, dict) and "mean" in v}
        if not agg:
            continue

        n = pooled.get("n")
        n_neg = pooled.get("n_nogrowth")
        prevalence = (n_neg / n) if n and n_neg is not None else None
        split_on = cv.get("split_on")
        row = {
            "experiment": exp.get("name", run_dir.name),
            "dataset": dataset_stem,
            "split_group": split_on or "",
            "n_groups": _n_groups(cv.get("splits_file")),
            "strategy": strategy,
            "folds": _fold_count(run_dir, strategy),
            "seed": exp.get("seed"),
            "model": model.get("type"),
            "class_weight": exp.get("class_weight"),
            "ablation": _ablation(features),
            "datapoints": n,
            "negatives": n_neg,
            "prevalence": round(prevalence, 5) if prevalence else None,
            "row_filters": "; ".join(
                f"-{c}={','.join(map(str, v))}"
                for c, v in (filters.get("exclude") or {}).items()
            ),
        }
        for metric, stat in agg.items():
            row[metric] = round(stat["mean"], 4)
            row[f"stdev_{metric}"] = round(stat["std"], 4)

        for metric in ("pr_auc_nogrowth", "mcc", "recall_nogrowth", "threshold"):
            if metric in pooled:
                row[f"pooled_{metric}"] = round(pooled[metric], 4)
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    if not args.reports.exists():
        print(f"error: no reports directory at {args.reports}", file=sys.stderr)
        return 1

    rows, skipped = [], []
    for run_dir in sorted(p for p in args.reports.iterdir() if p.is_dir()):
        collected = collect_run(run_dir)
        if collected:
            rows.extend(collected)
        else:
            skipped.append(run_dir.name)

    if not rows:
        print(f"error: no finished runs found under {args.reports}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    # Dataset, then split, then the fullest feature set first so the
    # ablations read as deltas from it.
    df["_n_features"] = df["ablation"].str.count(",") + 1
    df = (
        df.sort_values(
            ["dataset", "_n_features", "ablation"],
            ascending=[True, False, True],
        )
        .drop(columns=["_n_features"])
        .reset_index(drop=True)
    )

    lead = [
        "experiment", "dataset", "split_group", "n_groups",
        "strategy", "folds", "model", "ablation",
        "datapoints", "negatives", "prevalence",
        "pr_auc_nogrowth", "stdev_pr_auc_nogrowth",
        "mcc", "stdev_mcc",
    ]
    ordered = [c for c in lead if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    if args.core:
        df = df[[c for c in CORE_COLUMNS if c in df.columns]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} folder(s) from {args.reports}")
    if skipped:
        print(f"skipped {len(skipped)} folder(s) with no config.yaml+metrics.json: "
              f"{', '.join(skipped)}")
    print(f"wrote {args.out} ({len(df.columns)} columns)")
    if args.do_print:
        show = [c for c in CORE_COLUMNS if c in df.columns]
        print()
        print(df[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
