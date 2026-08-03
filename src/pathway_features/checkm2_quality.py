"""Parses CheckM2's quality_report.tsv into the genome quality confounder feature."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.checkm2_quality")


def load_quality_report(quality_report_tsv: Path) -> pd.DataFrame:
    """Load completeness/contamination from quality_report.tsv, rescaled to [0, 1]."""
    if not quality_report_tsv.exists():
        raise FileNotFoundError(
            f"CheckM2 quality_report.tsv not found at {quality_report_tsv} -- "
            "run the modular pipeline on DRAC first and copy "
            "$MODULAR_ROOT/output/checkm2/quality_report.tsv back into "
            "data/external/modular_pathway/checkm2/. See docs/modular_pathway_features.md."
        )

    df = pd.read_csv(quality_report_tsv, sep="\t", index_col="Name")
    df.index.name = "ncbi_assembly_accession"

    out = df[["Completeness", "Contamination"]].rename(
        columns={"Completeness": "checkm2_completeness", "Contamination": "checkm2_contamination"}
    ) / 100.0

    log.info("Loaded quality_report.tsv: %d genome(s)", len(out))
    return out
