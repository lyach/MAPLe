"""
pipeline.py
===========
Runs the BacDive pipeline end to end for a chosen extraction target:

  1. read the taxonomic scope + target (taxa.yaml),
  2. fetch every matching record per taxon (cached),
  3. flatten records into rows for the target
     (``(strain x medium)`` or ``(strain x metabolite x assay)``),
  4. de-duplicate rows across publications (same strain + unit),
  5. write the target CSV + ``bacdive_failures.csv`` and a summary.

Takes plain arguments so it can be driven from the CLI, a notebook, or a test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.bacdive import config
from src.bacdive.client import BacDiveFetcher
from src.bacdive.config import TargetSpec
from src.bacdive.extract import record_to_rows

log = logging.getLogger("bacdive.pipeline")


def _unit_key(row: pd.Series, spec: TargetSpec) -> str:
    """Stable key for a row's de-dup unit: preferred id, else name, plus extras.

    e.g. medium -> MediaDive id or media name; metabolite -> ChEBI id or
    substrate name, combined with the assay kind.
    """
    ident = str(row.get(spec.id_col, "") or "").strip()
    if ident:
        key = f"id:{ident}"
    else:
        key = "nm:" + str(row.get(spec.name_col, "") or "").strip().lower()
    for col in spec.extra_key_cols:
        key += "|" + str(row.get(col, "") or "").strip().lower()
    return key


def _dedupe(df: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    """Collapse duplicate (strain, unit) rows coming from multiple refs.

    On a ``growth_class`` conflict the positive label wins (max); all other
    columns take the first non-null value. Conflicts are logged.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_strain_key"] = df["strain_name"].astype(str).str.strip().str.lower()
    df["_unit_key"] = df.apply(lambda r: _unit_key(r, spec), axis=1)

    grp = df.groupby(["_strain_key", "_unit_key"], sort=False)

    n_conflicts = int((grp["growth_class"].nunique() > 1).sum())
    if n_conflicts:
        log.warning(
            "%d (strain, unit) pairs had conflicting labels -- resolved to positive.",
            n_conflicts,
        )

    agg = {col: "first" for col in df.columns if col not in
           {"growth_class", "_strain_key", "_unit_key"}}
    agg["growth_class"] = "max"

    deduped = grp.agg(agg).reset_index(drop=True)
    return deduped.drop(columns=["_strain_key", "_unit_key"], errors="ignore")


def run(
    taxa: list[str],
    target: str = config.DEFAULT_TARGET,
    output_csv: Optional[Path] = None,
    failures_csv: Path = config.FAILURES_CSV,
    cache_dir: Path = config.CACHE_DIR,
    request_delay: float = config.REQUEST_DELAY,
    limit: Optional[int] = None,
    force: bool = False,
    show_progress: bool = True,
) -> tuple[int, int, int]:
    """Run the pipeline for a list of taxonomy queries and a target.

    Parameters
    ----------
    taxa:
        Taxonomy queries (see ``taxa.yaml``).
    target:
        One of ``config.TARGETS`` (``medium_growth`` or
        ``metabolite_utilization``).
    output_csv:
        Output path; defaults to the target's spec path.
    limit:
        Optional cap on the total number of *records* processed (for quick
        smoke tests). ``None`` = no cap.
    force:
        Reserved for cache-busting; the fetcher always writes fresh JSON.
    show_progress:
        Show tqdm progress bars (one per taxon, plus one per-taxon for record
        fetching). Disable for non-interactive/log-only runs.

    Returns ``(n_rows, n_taxa_ok, n_failures)``.
    """
    if target not in config.TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(config.TARGETS)}"
        )
    spec = config.TARGETS[target]
    output_csv = output_csv or spec.output_csv

    log.info("target: %s -> %s", target, output_csv)

    fetcher = BacDiveFetcher(
        cache_dir=cache_dir,
        request_delay=request_delay,
        force=force,
        show_progress=show_progress,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    n_taxa_ok = 0
    n_records = 0

    taxa_bar = tqdm(
        taxa, desc="Taxa", unit="taxon", position=0, disable=not show_progress,
    )
    for taxon in taxa_bar:
        taxa_bar.set_postfix(records=n_records, current=taxon[:20])
        if limit is not None and n_records >= limit:
            log.info("record limit (%d) reached -- stopping.", limit)
            break
        try:
            got_any = False
            for record in fetcher.fetch_taxon(taxon):
                got_any = True
                n_records += 1
                rows.extend(record_to_rows(record, target))
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
    taxa_bar.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _dedupe(df, spec)
        # enforce column order / presence
        for col in spec.fieldnames:
            if col not in df.columns:
                df[col] = ""
        df = df[spec.fieldnames]
        # keep categoricals NaN-free so downstream OneHotEncoder is safe
        for col in spec.categorical_features:
            df[col] = df[col].replace("", "unknown").fillna("unknown")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("wrote %d row(s) -> %s", len(df), output_csv)

    _write_failures(failures, failures_csv)

    _log_summary(df, spec, n_records, n_taxa_ok, failures)
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
    spec: TargetSpec,
    n_records: int,
    n_taxa_ok: int,
    failures: list[dict],
) -> None:
    lines = [
        "=== BacDive pipeline summary ===",
        f"target            : {spec.name}",
        f"records processed : {n_records}",
        f"taxa with hits    : {n_taxa_ok}",
        f"taxa failed       : {len(failures)}",
        f"datapoints (rows) : {len(df)}",
    ]
    pos = neg = 0
    if not df.empty:
        pos = int((df["growth_class"] == 1).sum())
        neg = int((df["growth_class"] == 0).sum())
        lines += [
            f"unique strains    : {df['strain_name'].nunique()}",
            f"unique {spec.unit_col:<11}: {df[spec.unit_col].nunique()}",
            f"growth_class      : {pos} positive / {neg} negative",
        ]
    summary = "\n".join(lines)
    print(summary)
    log.info("run summary:\n%s", summary)

    # PU warning: positive-only labels can't be used as a headline metric.
    if spec.positive_only and pos and neg == 0:
        log.warning(
            "target '%s' is positive-only (%d positive / 0 negative). This is a "
            "positive-unlabeled (PU) signal: absence of a (strain, unit) pair is "
            "NOT a negative. Use it as weak supervision / pretraining, not a "
            "reported classification metric. See bacdive/README.md.",
            spec.name, pos,
        )