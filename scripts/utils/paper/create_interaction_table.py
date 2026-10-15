#!/usr/bin/env python3
"""Build the identifiability table: how often each axis alone flips the label.

For each corpus and each experimental axis, how often does changing *only*
that axis flip the growth label? Runs over the four BacDive tables under
``data/processed/bacdive/`` and MAPLE-Lab.

The test: for axis X, hold every *other* axis fixed and count how many of
the cells that vary X contain both a growth and a no-growth label. That
fraction bounds what any model conditioned on the other axes could possibly
learn about X. Which cells are conditioned on is derived from the corpus,
not hardcoded, so a corpus recording a single temperature is not asked a
temperature question.

Atmosphere is not an axis and is not held fixed; see ``_paper_datasets`` for
what that means when choosing which MAPLE-Lab (``LAB_CSV``).

Includes a strain-driven row alongside medium/temperature (swaps
the organism instead of the axis), and keys cells on the canonicalised
medium rather than the raw name (``--raw-medium`` reruns on raw names to
show how much alias-splitting distorts the estimate).

Usage:
    python scripts/utils/paper/create_interaction_table.py
    python scripts/utils/paper/create_interaction_table.py --format all
    python scripts/utils/paper/create_interaction_table.py --raw-medium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the sibling helper whatever the working directory is, so the
# script runs from the repo root as well as from its own folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_datasets import (
    LAB_CSV,
    ROOT,
    Corpus,
    emit,
    load_all,
)

# Axis -> how it reads in the table. Order fixes the row order per corpus.
AXIS_ORDER = ("medium", "temperature", "strain")
AXIS_PHRASE = {
    "medium": "medium-driven flip",
    "temperature": "temperature-driven flip",
    "strain": "strain-driven flip",
}
AXIS_NOUN = {
    "medium": "medium", "temperature": "temperature",
    "strain": "strain",
}


def _resolve(name: str, medium_col: str) -> str:
    return {"medium": medium_col, "strain": "strain"}.get(name, name)


def flip_rate(
    df: pd.DataFrame, axis: str, conditioned_on: list[str], medium_col: str
) -> dict[str, float]:
    """Identifiability of ``axis`` with ``conditioned_on`` held fixed.

    Returns ``cells`` / ``flips`` / ``frac`` (among cells that vary the axis,
    how many carry both labels), plus ``mean_n`` (mean distinct axis values
    per varying cell) and ``pair_frac`` (fraction of within-cell value pairs
    that disagree).

    ``pair_frac`` exists because ``frac`` isn't comparable across axes -- a
    cell with more distinct values is more likely to contain both labels
    regardless of any real effect. Counting discordant pairs instead of mixed
    cells is invariant to cell size, so it's the number to compare across
    axes.
    """
    col = _resolve(axis, medium_col)
    keys = [_resolve(k, medium_col) for k in conditioned_on]

    sub = df.dropna(subset=[col, *keys, "label"])
    empty = {"cells": 0, "flips": 0, "frac": np.nan, "mean_n": np.nan,
             "pair_frac": np.nan, "pairs": 0}
    if sub.empty:
        return empty

    cells = sub.groupby(keys, dropna=True).agg(
        n_axis=(col, "nunique"), n_lab=("label", "nunique")
    )
    varying = cells[cells["n_axis"] > 1]
    if not len(varying):
        return empty

    # Pairwise discordance, cell-size invariant. Collapse to one entry per
    # (cell, axis value); keep only values with an unambiguous label.
    per_value = sub.groupby([*keys, col])["label"].agg(["nunique", "min"])
    per_value = per_value[per_value["nunique"] == 1]
    per_cell = per_value.groupby(level=list(range(len(keys))))["min"].agg(
        n="size", n1="sum"
    )
    per_cell = per_cell[per_cell["n"] > 1]
    n, n1 = per_cell["n"], per_cell["n1"]
    n_pairs = int((n * (n - 1) // 2).sum())
    n_disc = int((n1 * (n - n1)).sum())

    return {
        "cells": len(varying),
        "flips": int((varying["n_lab"] > 1).sum()),
        "frac": float((varying["n_lab"] > 1).mean()),
        "mean_n": float(varying["n_axis"].mean()),
        "pair_frac": (n_disc / n_pairs) if n_pairs else np.nan,
        "pairs": n_pairs,
    }


def temperature_separable(df: pd.DataFrame, medium_col: str) -> tuple[int, int]:
    """(dual-label (strain, medium) pairs, how many temperature alone separates).

    Among (strain, medium) pairs carrying both labels, a pair is "separable
    by temperature" if the temperatures at which it grew and those at which
    it did not don't overlap -- i.e. the disagreement is fully explained by
    temperature, not by the medium.
    """
    sub = df.dropna(subset=["temperature", "label", medium_col, "strain"])
    if sub.empty:
        return 0, 0
    pair = sub.groupby(["strain", medium_col])
    both = pair["label"].nunique().pipe(lambda s: s[s == 2]).index
    if not len(both):
        return 0, 0
    ranges = (
        sub.set_index(["strain", medium_col]).loc[both]
        .groupby(["strain", medium_col, "label"])["temperature"].agg(["min", "max"])
        .unstack("label")
    )
    ranges.columns = [f"{a}_{b}" for a, b in ranges.columns]
    need = {"min_0", "max_0", "min_1", "max_1"}
    if not need <= set(ranges.columns):
        return len(both), 0
    ranges = ranges.dropna(subset=list(need))
    # Disjoint temperature intervals => temperature alone explains the split.
    disjoint = (ranges["max_0"] < ranges["min_1"]) | (ranges["min_0"] > ranges["max_1"])
    return len(both), int(disjoint.sum())


def build(corpora: list[Corpus], medium_col: str) -> tuple[pd.DataFrame, list[str]]:
    """The table, plus warnings for axes a corpus cannot identify."""
    rows: list[dict] = []
    warnings: list[str] = []
    for c in corpora:
        axes = c.axes
        for axis in AXIS_ORDER:
            if axis not in axes:
                continue
            others = [a for a in axes if a != axis]
            if not others:
                continue
            r = flip_rate(c.df, axis, others, medium_col)
            held = ", ".join(AXIS_NOUN[a] for a in others)
            if r["cells"] == 0:
                # The axis exists in the data but no cell varies it with
                # everything else fixed -- it is confounded, not merely weak.
                warnings.append(
                    f"{c.label}: the {AXIS_NOUN[axis]} axis is UNIDENTIFIABLE "
                    f"-- no ({held}) cell varies it, so it is confounded with "
                    f"something held fixed and cannot be reported from "
                    f"{c.path.name}."
                )
            rows.append({
                "dataset": c.label,
                "test": f"({held}) cells with >1 {AXIS_NOUN[axis]} "
                        f"-> {AXIS_PHRASE[axis]}",
                "cells": r["cells"],
                "flips": r["flips"],
                "frac": r["frac"],
                "mean values/cell": r["mean_n"],
                "pair frac": r["pair_frac"],
            })
        if "temperature" in axes:
            cells, sep = temperature_separable(c.df, medium_col)
            rows.append({
                "dataset": c.label,
                "test": "dual-label (strain, medium) cells, separable by "
                        "temperature alone",
                "cells": cells,
                "flips": sep,
                "frac": (sep / cells) if cells else np.nan,
                "mean values/cell": np.nan,
                "pair frac": np.nan,
            })

    table = pd.DataFrame(rows)
    for col in ("cells", "flips"):
        table[col] = table[col].map(lambda v: f"{v:,}")
    table["frac"] = table["frac"].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")
    table["mean values/cell"] = table["mean values/cell"].map(
        lambda v: "--" if pd.isna(v) else f"{v:.1f}")
    table["pair frac"] = table["pair frac"].map(
        lambda v: "--" if pd.isna(v) else f"{v:.4f}")
    return table, warnings


def summarise(corpora: list[Corpus], medium_col: str) -> None:
    """Compare the medium and strain axes within each corpus.

    Uses ``pair frac``, not ``frac``: on the plain ``frac`` statistic the
    strain axis looks more identifiable than medium, but that's a cell-size
    artifact (a strain-varying cell holds far more distinct values than a
    medium-varying one, so it contains both labels more easily); the
    pairwise statistic corrects for this.

    Flip rates aren't compared across corpora, since prevalence differs
    between them and that alone changes how much room there is for any
    pair to disagree.
    """
    print("\n=== medium vs strain axis, within each dataset ===")
    print("Prevalence = % of negatives")
    print(f"{'dataset':22} {'prevalence':>11} {'medium':>9} {'strain':>9}   "
          f"{'more identifiable':>20}")
    for c in corpora:
        axes = c.axes
        got = {}
        for axis in ("medium", "strain"):
            others = [a for a in axes if a != axis]
            got[axis] = flip_rate(c.df, axis, others, medium_col)["pair_frac"]
        med, strn = got["medium"], got["strain"]
        prev = float((c.df["label"] == 0).mean())
        both_known = pd.notna(med) and pd.notna(strn)
        if both_known and med > 0 and strn > 0:
            verdict = (f"medium" if med > strn
                       else f"strain")
        else:
            verdict = "--"
        med_s = f"{med:.4f}" if pd.notna(med) else "--"
        strn_s = f"{strn:.4f}" if pd.notna(strn) else "--"
        print(f"{c.label:22} {prev:>11.4f} {med_s:>9} {strn_s:>9}   {verdict:>20}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", action="append", default=None,
                   help="Restrict to these corpus keys (repeatable)")
    p.add_argument("--lab-csv", type=Path, default=LAB_CSV,
                   help="MAPLE-Lab CSV to use (default: %(default)s)")
    p.add_argument("--raw-medium", action="store_true",
                   help="Key cells on the raw media_name instead of media_canon, "
                        "to show how much the alias problem distorts the estimate")
    p.add_argument("--format", choices=["md", "csv", "latex", "all"], default="md")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "paper" / "interaction_table")
    args = p.parse_args()

    corpora = load_all(args.only, lab_csv=args.lab_csv)
    if not corpora:
        print("error: no corpora loaded")
        return 1

    medium_col = "medium_raw" if args.raw_medium else "medium"
    out = args.out if not args.raw_medium else args.out.with_name(
        args.out.name + "_rawmedium")
    print(f"medium keyed on: {medium_col}")

    table, warnings = build(corpora, medium_col)
    emit(table, args.format, out,
         caption="Identifiability of each experimental axis: how often changing "
                 "only that axis flips the growth label.",
         tex_label="tab:interaction")

    for w in warnings:
        print(f"\nWARNING: {w}")
    summarise(corpora, medium_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
