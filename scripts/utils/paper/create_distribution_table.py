#!/usr/bin/env python3
"""Build the datasets' stats table (distribution and metrics).

One column per dataset, one row per descriptive statistic. Corpora and their
normalisation live in ``_paper_datasets.py`` so this script and
``create_interaction_table.py`` cannot disagree about what a dataset is.

Usage:
    python scripts/utils/paper/create_distribution_table.py
    python scripts/utils/paper/create_distribution_table.py --format all
    python scripts/utils/paper/create_distribution_table.py --only bd_med_w_genome --only lab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _fmt(n: object) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "--"
    if isinstance(n, float):
        return f"{n:,.4f}".rstrip("0").rstrip(".")
    return f"{n:,}"


def _pct(part: int, whole: int) -> str:
    return "--" if not whole else f"{part:,} ({100 * part / whole:.1f}%)"


def describe(c: Corpus) -> dict[str, str]:
    """Every statistic for one corpus, as display strings."""
    df = c.df
    neg = df["label"] == 0
    pos = df["label"] == 1
    n = len(df)

    n_strains = df["strain"].nunique()
    n_media_raw = df["medium_raw"].nunique()
    n_media_canon = df["medium"].nunique()
    n_temps = int(df["temperature"].nunique(dropna=True))

    pairs = df.groupby(["strain", "medium"]).ngroups
    fill = (pairs / (n_strains * n_media_canon)
            if n_strains and n_media_canon else None)

    # A strain whose label never changes contributes no within-strain contrast,
    # so it cannot help an organism-level claim however many rows it has.
    per_strain = df.groupby("strain")["label"].nunique()
    never_grew = int(df.groupby("strain")["label"].max().eq(0).sum())
    always_grew = int(df.groupby("strain")["label"].min().eq(1).sum())

    # A (strain, medium) pair carrying both labels is condition-dependent: no
    # (genome, medium)-only model can be right about both rows.
    pair_labels = df.groupby(["strain", "medium"])["label"].nunique()

    stats: dict[str, str] = {
        # --- size
        "Datapoints (rows)": _fmt(n),
        "Unique strains": _fmt(n_strains),
        "Unique medium (raw names)": _fmt(n_media_raw),
        "Unique medium (canonicalised)": _fmt(n_media_canon),
        "Distinct temperatures": _fmt(n_temps),
        # --- label balance
        "Positives / negatives": f"{int(pos.sum()):,} / {int(neg.sum()):,}",
        "Positive rate": _fmt(round(float(pos.mean()), 4)),
        "No-growth prevalence": _fmt(round(float(neg.mean()), 4)),
        # --- where the negatives live
        "Strains with >=1 negative": _pct(int(df.loc[neg, "strain"].nunique()), n_strains),
        "Media with >=1 negative (raw)": _pct(
            int(df.loc[neg, "medium_raw"].nunique()), n_media_raw),
        "Media with >=1 negative (canon)": _pct(
            int(df.loc[neg, "medium"].nunique()), n_media_canon),
        "Top canonical medium's share of negatives": (
            f"{100 * df.loc[neg, 'medium'].value_counts().iloc[0] / neg.sum():.1f}%"
            if neg.any() else "--"),
        "Top-5 canonical media share of negatives": (
            f"{100 * df.loc[neg, 'medium'].value_counts().head(5).sum() / neg.sum():.1f}%"
            if neg.any() else "--"),
        # --- design shape
        "Observed (strain, medium) pairs": _fmt(pairs),
        "Matrix fill (pairs / strains x media)": (
            f"{100 * fill:.2f}%" if fill is not None else "--"),
        "Rows per strain (median)": _fmt(float(df.groupby("strain").size().median())),
        "Rows per canonical medium (median)": _fmt(
            float(df.groupby("medium").size().median())),
        # --- how much of the corpus can identify anything
        "Strains with both labels": _pct(int(per_strain.eq(2).sum()), n_strains),
        "Strains that never grew": _pct(never_grew, n_strains),
        "Strains that always grew": _pct(always_grew, n_strains),
        "Condition-dependent (strain, medium) pairs": _pct(
            int(pair_labels.eq(2).sum()), pairs),
        # --- taxonomy
        "Genera / families / phyla": " / ".join(
            _fmt(int(df[col].nunique())) if col in df.columns else "--"
            for col in ("genus", "family", "phylum")),
    }
    return stats


def build(corpora: list[Corpus]) -> pd.DataFrame:
    per = {c.label: describe(c) for c in corpora}
    row_order = list(next(iter(per.values())))
    table = pd.DataFrame(
        [{"Statistic": r, **{lab: s.get(r, "--") for lab, s in per.items()}}
         for r in row_order]
    )
    # Drop rows that are "--" for every corpus so the table never carries an
    # all-empty line.
    value_cols = [c for c in table.columns if c != "Statistic"]
    keep = ~(table[value_cols].apply(lambda r: set(r) <= {"--", "0"}, axis=1))
    return table[keep].reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", action="append", default=None,
                   help="Restrict to these corpus keys (repeatable)")
    p.add_argument("--lab-csv", type=Path, default=LAB_CSV,
                   help="MAPLE-Lab CSV to use (default: %(default)s)")
    p.add_argument("--format", choices=["md", "csv", "latex", "all"], default="md")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "paper" / "distribution_table")
    args = p.parse_args()

    corpora = load_all(args.only, lab_csv=args.lab_csv)
    if not corpora:
        print("error: no corpora loaded")
        return 1

    table = build(corpora)
    emit(table, args.format, args.out,
         caption="Dataset distribution across the MAPLE corpora.",
         tex_label="tab:distribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
