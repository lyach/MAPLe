#!/usr/bin/env python3
"""
add_media_canon.py
==================
Add (or refresh) the ``media_canon`` / ``media_canon_status`` columns on A
built BacDive parquets.

``media_canon`` is a function of an existing column in a BacDive parquet.
It does not need the 100k-record cache re-read, row counts are never changed, 
and can work newly written ``data/external/media_alias_map.csv``.

USAGE:
    python scripts/bacdive/add_media_canon.py                    # all four tables
    python scripts/bacdive/add_media_canon.py --dry-run          # report only
    python scripts/bacdive/add_media_canon.py --parquet data/processed/bacdive/bd_med.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bacdive.media import (
    canonical_media_name,
    load_media_alias_map,
    media_canon_status,
)

BACDIVE_DIR = ROOT / "data" / "processed" / "bacdive"
DEFAULT_PARQUETS = [
    BACDIVE_DIR / f"{stem}.parquet"
    for stem in ("bd_temp", "bd_med", "bd_temp_w_genome", "bd_med_w_genome")
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parquet", type=Path, action="append", default=None,
                   help="Table to annotate (repeatable; default: all four)")
    p.add_argument("--alias-map", type=Path, default=None,
                   help="Curated media_name -> canonical_key CSV "
                        "(default: data/external/media_alias_map.csv, optional)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the effect without writing")
    return p.parse_args()


def annotate(df: pd.DataFrame, overrides: dict[str, str]) -> pd.DataFrame:
    """Return ``df`` with media_canon / media_canon_status set.

    Canonicalisation is computed once per distinct ``media_name`` and mapped
    back, rather than per row -- there are ~3.6k distinct names behind ~133k
    rows, so this is the difference between 3.6k and 133k regex passes.
    """
    names = df["media_name"].astype(str)
    distinct = pd.Index(names.unique())
    canon = {n: canonical_media_name(n, overrides) for n in distinct}
    status = {n: media_canon_status(n, overrides) for n in distinct}
    out = df.copy()
    out["media_canon"] = names.map(canon)
    out["media_canon_status"] = names.map(status)
    return out


def report(path: Path, before: pd.DataFrame, after: pd.DataFrame) -> None:
    n_raw = before["media_name"].nunique()
    n_canon = after["media_canon"].nunique()
    neg = after["growth_class"] == 0 if "growth_class" in after else pd.Series(False, index=after.index)
    status = after["media_canon_status"].value_counts()

    print(f"\n{path.name}: {len(after):,} rows (unchanged)")
    print(f"  media_name {n_raw:,} -> media_canon {n_canon:,} "
          f"({100 * (1 - n_canon / n_raw):.1f}% collapse)")
    print(f"  media_canon_status: {status.to_dict()}")
    if neg.any():
        print(f"  negative-bearing groups: {after.loc[neg, 'media_canon'].nunique()} "
              f"(was {before.loc[neg, 'media_name'].nunique()} raw names)")
        bad = int((neg & (after["media_canon_status"] != "ok")).sum())
        print(f"  negatives on a non-identifiable medium: {bad}")
        top = after.loc[neg, "media_canon"].value_counts().head(5)
        print(f"  top negative groups: {[(k, int(v)) for k, v in top.items()]}")

    # The groups that actually merged -- the point of the exercise.
    merged = (
        after.groupby("media_canon")["media_name"].nunique().pipe(lambda s: s[s > 1])
    )
    print(f"  canon groups covering >1 raw name: {len(merged)} "
          f"(folding {int(merged.sum())} raw names)")


def main() -> int:
    args = parse_args()
    overrides = load_media_alias_map(args.alias_map)
    targets = args.parquet or DEFAULT_PARQUETS

    missing = [p for p in targets if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: no such table: {p}", file=sys.stderr)
        return 1

    for path in targets:
        before = pd.read_parquet(path)
        after = annotate(before, overrides)
        if len(after) != len(before):
            raise AssertionError(
                f"{path}: row count changed {len(before)} -> {len(after)}"
            )
        report(path, before, after)
        if not args.dry_run:
            after.to_parquet(path, index=False)
            print(f"  wrote {path}")

    if args.dry_run:
        print("\n[dry run] nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
