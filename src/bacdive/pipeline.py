"""
pipeline.py
===========
Orchestrates the BacDive growth pipeline end to end:

  1. read the taxonomic scope (taxa.yaml),
  2. fetch every matching record per taxon (cached),
  3. flatten records into (strain x culture medium) rows,
  4. de-duplicate rows across publications (same strain + medium),
  5. write ``bacdive_growth.csv`` + ``bacdive_failures.csv`` and a summary.

Takes plain arguments so it can be driven from the CLI, a notebook, or a test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.bacdive import config
from src.bacdive.client import BacDiveFetcher
from src.bacdive.extract import record_to_rows

log = logging.getLogger("bacdive.pipeline")


def _media_key(row: pd.Series) -> str:
    """Stable key identifying a medium: MediaDive id if present, else name."""
    mid = str(row.get("mediadive_id", "") or "").strip()
    if mid:
        return f"md:{mid}"
    return "nm:" + str(row.get("media_name", "") or "").strip().lower()


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (strain, medium) rows coming from multiple refs.

    On a growth conflict the positive label wins (``growth_class`` = max);
    all other columns take the first non-null value. Conflicts are logged.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_strain_key"] = df["strain_name"].astype(str).str.strip().str.lower()
    df["_media_key"] = df.apply(_media_key, axis=1)

    grp = df.groupby(["_strain_key", "_media_key"], sort=False)

    n_conflicts = int((grp["growth_class"].nunique() > 1).sum())
    if n_conflicts:
        log.warning(
            "%d (strain, medium) pairs had conflicting growth labels "
            "-- resolved to positive.",
            n_conflicts,
        )

    agg = {col: "first" for col in df.columns if col not in
           {"growth_class", "_strain_key", "_media_key"}}
    agg["growth_class"] = "max"

    deduped = grp.agg(agg).reset_index(drop=True)
    return deduped.drop(columns=["_strain_key", "_media_key"], errors="ignore")


def run(
    taxa: list[str],
    output_csv: Path = config.OUTPUT_CSV,
    failures_csv: Path = config.FAILURES_CSV,
    cache_dir: Path = config.CACHE_DIR,
    request_delay: float = config.REQUEST_DELAY,
    limit: Optional[int] = None,
    force: bool = False,
) -> tuple[int, int, int]:
    """Run the pipeline for a list of taxonomy queries.

    Parameters
    ----------
    taxa:
        Taxonomy queries (see ``taxa.yaml``).
    limit:
        Optional cap on the total number of *records* processed (for quick
        smoke tests). ``None`` = no cap.
    force:
        Reserved for cache-busting; the fetcher always writes fresh JSON.

    Returns ``(n_rows, n_taxa_ok, n_failures)``.
    """
    fetcher = BacDiveFetcher(cache_dir=cache_dir, request_delay=request_delay, force=force)

    rows: list[dict] = []
    failures: list[dict] = []
    n_taxa_ok = 0
    n_records = 0

    for taxon in taxa:
        if limit is not None and n_records >= limit:
            log.info("record limit (%d) reached -- stopping.", limit)
            break
        try:
            got_any = False
            for record in fetcher.fetch_taxon(taxon):
                got_any = True
                n_records += 1
                rows.extend(record_to_rows(record))
                if limit is not None and n_records >= limit:
                    break
            if got_any:
                n_taxa_ok += 1
            else:
                failures.append({"taxon": taxon, "reason": "no records returned"})
        except LookupError as exc:
            log.warning("%s", exc)
            failures.append({"taxon": taxon, "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001 - keep going on any one taxon
            log.error("error processing taxon %r: %s", taxon, exc)
            failures.append({"taxon": taxon, "reason": f"error: {exc}"})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _dedupe(df)
        # enforce column order / presence
        for col in config.OUTPUT_FIELDNAMES:
            if col not in df.columns:
                df[col] = ""
        df = df[config.OUTPUT_FIELDNAMES]
        # keep categoricals NaN-free so downstream OneHotEncoder is safe
        for col in config.CATEGORICAL_FEATURES:
            df[col] = df[col].replace("", "unknown").fillna("unknown")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("wrote %d row(s) -> %s", len(df), output_csv)

    _write_failures(failures, failures_csv)

    _log_summary(df, n_records, n_taxa_ok, failures)
    return len(df), n_taxa_ok, len(failures)


def _write_failures(failures: list[dict], failures_csv: Path) -> None:
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=config.FAILURE_FIELDNAMES).to_csv(
        failures_csv, index=False
    )
    if failures:
        log.info("wrote %d failure(s) -> %s", len(failures), failures_csv)


def _log_summary(
    df: pd.DataFrame,
    n_records: int,
    n_taxa_ok: int,
    failures: list[dict],
) -> None:
    lines = [
        "=== BacDive pipeline summary ===",
        f"records processed : {n_records}",
        f"taxa with hits    : {n_taxa_ok}",
        f"taxa failed       : {len(failures)}",
        f"datapoints (rows) : {len(df)}",
    ]
    if not df.empty:
        pos = int((df["growth_class"] == 1).sum())
        neg = int((df["growth_class"] == 0).sum())
        lines += [
            f"unique strains    : {df['strain_name'].nunique()}",
            f"unique media      : {df['media_name'].nunique()}",
            f"growth_class      : {pos} positive / {neg} negative",
        ]
    summary = "\n".join(lines)
    print(summary)
    log.info("run summary:\n%s", summary)
