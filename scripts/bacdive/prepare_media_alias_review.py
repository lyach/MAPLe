#!/usr/bin/env python3
"""Build a review worksheet for medium-name aliases.

Audits what the deterministic rules in
``src.bacdive.media.canonical_media_name`` did and did not merge, for the
grouping key used by the held-out-medium split. Identity sibling of
``prepare_media_name_review.py``, which instead resolves a name to a
MediaDive recipe.

For a leakage guard, under-merging is worse than over-merging: the held-out
medium can still leak into training under another spelling. So the review
leans toward merging, and this sheet surfaces under-merges first. Similarity
reuses ``prepare_media_name_review._score``.

Two review kinds are emitted:

``under_merge``  Two in-scope names score as aliases but currently have
                 different ``media_canon``. Fill ``canonical_key`` with the
                 shared key to merge them (or leave blank to keep them apart).
``over_merge``   Names sharing one ``media_canon`` that differ by a
                 formulation-changing word the rules stripped ("half
                 strength", "modified", ...). Fill ``canonical_key`` with a
                 distinct key to split them.

Scope defaults to names with at least one no-growth label or at least
``--min-rows`` rows. Feed the completed sheet to
``apply_media_alias_review.py``, then re-run ``add_media_canon.py``.

Usage:
    python scripts/bacdive/prepare_media_alias_review.py
    python scripts/bacdive/prepare_media_alias_review.py --min-rows 20 --threshold 0.72
    python scripts/bacdive/prepare_media_alias_review.py --negatives-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.bacdive.prepare_media_name_review import _score
from src.bacdive.media import RISKY_STRIPPED_TOKENS

BACDIVE_DIR = ROOT / "data" / "processed" / "bacdive"
DEFAULT_PARQUETS = [
    BACDIVE_DIR / f"{stem}.parquet"
    for stem in ("bd_temp", "bd_med", "bd_temp_w_genome", "bd_med_w_genome")
]
REVIEW_CSV = ROOT / "data" / "external" / "media_alias_review.csv"

FIELDNAMES = [
    "review_kind", "pair_id", "score", "media_name", "row_count", "n_negatives",
    "current_canon", "other_media_name", "other_canon", "canonical_key", "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parquet", type=Path, action="append", default=None,
                   help="Table(s) to pool names from (default: all four)")
    p.add_argument("--out", type=Path, default=REVIEW_CSV)
    p.add_argument("--min-rows", type=int, default=20,
                   help="In scope if a name has at least this many rows "
                        "(default: %(default)s)")
    p.add_argument("--negatives-only", action="store_true",
                   help="Restrict scope to names carrying a no-growth label")
    p.add_argument("--threshold", type=float, default=0.72,
                   help="Score at or above which two names are alias candidates "
                        "(default: %(default)s)")
    return p.parse_args()


def load_names(parquets: list[Path]) -> pd.DataFrame:
    """Pool per-name row counts, negative counts and current canon across tables."""
    frames = []
    for path in parquets:
        cols = ["media_name", "media_canon", "growth_class"]
        frames.append(pd.read_parquet(path, columns=cols))
    pooled = pd.concat(frames, ignore_index=True)
    return (
        pooled.groupby("media_name")
        .agg(
            row_count=("media_name", "size"),
            n_negatives=("growth_class", lambda s: int((s == 0).sum())),
            current_canon=("media_canon", "first"),
        )
        .reset_index()
    )


def find_under_merges(scope: pd.DataFrame, threshold: float) -> list[dict]:
    """Alias-scoring name pairs that currently have different canon keys."""
    names = scope["media_name"].tolist()
    meta = scope.set_index("media_name")
    rows: list[dict] = []
    for i, a in enumerate(names):
        canon_a = meta.at[a, "current_canon"]
        for b in names[i + 1:]:
            if meta.at[b, "current_canon"] == canon_a:
                continue
            score = _score(a, b)
            if score < threshold:
                continue
            pair_id = f"u{len(rows) // 2:04d}"
            for first, second in ((a, b), (b, a)):
                rows.append({
                    "review_kind": "under_merge",
                    "pair_id": pair_id,
                    "score": round(score, 3),
                    "media_name": first,
                    "row_count": int(meta.at[first, "row_count"]),
                    "n_negatives": int(meta.at[first, "n_negatives"]),
                    "current_canon": meta.at[first, "current_canon"],
                    "other_media_name": second,
                    "other_canon": meta.at[second, "current_canon"],
                    "canonical_key": "",
                    "notes": "",
                })
    return rows


def _risky_tokens(name: str) -> frozenset[str]:
    """Formulation-changing words present in ``name`` that canonicalisation strips."""
    tokens = set(re.findall(r"[a-z0-9]+", str(name).lower()))
    return frozenset(tokens & RISKY_STRIPPED_TOKENS)


def find_over_merges(scope: pd.DataFrame) -> list[dict]:
    """Names merged into one canon key that differ by a formulation-changing word.

    Deliberately NOT based on scoring the raw names against each other: two
    spellings of one medium can share almost no characters ("NB" vs "NUTRIENT
    AGAR (DSMZ Medium 1)"), so a low raw score marks a *successful* acronym
    merge far more often than a mistaken one. What actually risks a bad merge
    is stripping a word that changes the recipe -- "half strength", "modified",
    "minimal" -- so those are what we surface.
    """
    rows: list[dict] = []
    meta = scope.set_index("media_name")
    for canon, group in scope.groupby("current_canon"):
        members = group["media_name"].tolist()
        if len(members) < 2:
            continue
        by_risk: dict[frozenset[str], list[str]] = {}
        for name in members:
            by_risk.setdefault(_risky_tokens(name), []).append(name)
        if len(by_risk) < 2:
            continue  # every member carries the same qualifiers -- nothing lost
        # Pair the plainest variant against each qualified one.
        plainest = min(by_risk, key=len)
        base_name = by_risk[plainest][0]
        for risk, names in by_risk.items():
            if risk == plainest:
                continue
            for name in names:
                pair_id = f"o{len(rows) // 2:04d}"
                note = "differs by: " + ", ".join(sorted(risk - plainest))
                for first, second in ((name, base_name), (base_name, name)):
                    rows.append({
                        "review_kind": "over_merge",
                        "pair_id": pair_id,
                        "score": round(_score(first, second), 3),
                        "media_name": first,
                        "row_count": int(meta.at[first, "row_count"]),
                        "n_negatives": int(meta.at[first, "n_negatives"]),
                        "current_canon": canon,
                        "other_media_name": second,
                        "other_canon": canon,
                        "canonical_key": "",
                        "notes": note,
                    })
    return rows


def main() -> int:
    args = parse_args()
    parquets = args.parquet or DEFAULT_PARQUETS
    missing = [p for p in parquets if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: no such table: {p}", file=sys.stderr)
        return 1

    agg = load_names(parquets)
    if "media_canon" not in pd.read_parquet(parquets[0]).columns:
        print("error: media_canon column missing -- run add_media_canon.py first",
              file=sys.stderr)
        return 1

    in_scope = agg["n_negatives"] >= 1
    if not args.negatives_only:
        in_scope = in_scope | (agg["row_count"] >= args.min_rows)
    scope = agg[in_scope].sort_values("row_count", ascending=False).reset_index(drop=True)

    print(f"pooled {len(parquets)} table(s): {len(agg):,} distinct media_name")
    print(f"in scope: {len(scope):,} names "
          f"({int(scope.n_negatives.sum()):,} negatives, "
          f"{int(scope.row_count.sum()):,} rows)")
    print(f"current media_canon groups in scope: {scope.current_canon.nunique():,}")

    under = find_under_merges(scope, args.threshold)
    over = find_over_merges(scope)
    print(f"under_merge candidate pairs (score >= {args.threshold}): {len(under) // 2}")
    print(f"over_merge candidate pairs  (formulation word stripped): "
          f"{len(over) // 2}")

    review = pd.DataFrame(under + over, columns=FIELDNAMES)
    if review.empty:
        print("nothing to review -- the rules merged everything the scorer agrees on")
        return 0

    review = review.sort_values(
        ["review_kind", "n_negatives", "score"], ascending=[True, False, False]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(review)} rows, "
          f"{review.pair_id.nunique()} pairs)")
    print("Fill 'canonical_key' to merge (same key on both rows of a pair) or "
          "split (distinct keys), then run apply_media_alias_review.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
