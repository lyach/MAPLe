"""
checkm2_quality.py
====================
Parses CheckM2's quality_report.tsv into the strain x [completeness,
contamination] confounder feature this branch's spec calls "required" --
independent of, and not used to compute, the KEGG-module completeness
matrix (see modular_completeness.py). CheckM2 measures genome assembly/
annotation quality (how complete and how contaminated a genome is), a
different axis entirely from pathway/module completeness.

CheckM2 reports Completeness/Contamination as 0-100 percentages; converted
to 0.0-1.0 fractions here to match this repo's established convention
(kegg_pathway_completeness.csv, see distillate.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.checkm2_quality")


def load_quality_report(quality_report_tsv: Path) -> pd.DataFrame:
    """Load CheckM2's quality_report.tsv, indexed by genome accession (its
    'Name' column -- the FASTA filename stem, same accession convention as
    the rest of this package). Returns just the two confounder columns,
    renamed and rescaled to [0, 1]; CheckM2's other columns (GC_Content,
    Coding_Density, Genome_Size, ...) are assembly statistics, not part of
    this branch's "required confounder feature" spec, so left out.
    """
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
