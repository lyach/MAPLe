"""
modular_completeness.py
=========================
Builds this pipeline's canonical strain x KEGG-module completeness matrix
using EBI's kegg-pathways-completeness-tool (give_completeness) against
PyKofamSearch's KO hits.

give_completeness reports REAL KEGG module IDs directly for every module it
scores -- no free-text-name recovery step needed here.

give_completeness --add-per-contig emits a SPARSE long table: one row per
(genome, module) pair, but ONLY for modules with nonzero completeness for
that genome -- confirmed empirically that a genome with zero KO hits
produces ZERO rows, not a zero-filled row. build_completeness_matrix's
reindex/fillna step turns that sparsity into a dense strain x module
matrix with explicit zero cells; a plain pivot would silently drop
genomes/modules with nothing to report instead of scoring them 0.0.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.modular_completeness")


def write_give_completeness_input(accession_to_kos: dict[str, list[str]], out_tsv: Path) -> None:
    """One row per genome: accession, then its unique KO IDs, tab-separated
    -- give_completeness's documented per-genome input format. A genome with
    zero KOs gets a single-column row (accession only); give_completeness
    accepts this fine (confirmed: it just contributes zero rows to the
    --add-per-contig output for that genome, not an error).
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8") as f:
        for accession, kos in accession_to_kos.items():
            f.write("\t".join([accession, *kos]) + "\n")
    log.info("Wrote give_completeness input for %d genome(s) -> %s", len(accession_to_kos), out_tsv)


def run_give_completeness(input_tsv: Path, outdir: Path, outprefix: str = "modular") -> Path:
    """Invokes give_completeness locally via subprocess -- it's a fast,
    database-free CLI (bundles its own reference data), so this step runs
    here rather than on DRAC. Returns the path to the per-contig
    output TSV (--add-per-contig is always passed; the plain aggregate
    output pools every genome in the input into one combined result and
    isn't useful for a strain x module matrix -- confirmed empirically).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "give_completeness", "--input", str(input_tsv), "--add-per-contig",
        "--outdir", str(outdir), "--outprefix", outprefix,
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"give_completeness failed (exit {result.returncode}):\n{result.stderr}")
    return outdir / f"{outprefix}_contigs.tsv"


def build_completeness_matrix(contigs_tsv: Path, accessions: list[str]) -> pd.DataFrame:
    """Reshape give_completeness's sparse long per-contig output into a
    dense strain x module matrix of [0, 1] fractions (give_completeness
    reports 0-100 percentages). Explicitly reindexed against the FULL known
    accession list passed in -- genomes entirely absent from the sparse
    output (zero KO hits) become real all-zero rows, not missing ones; the
    module (column) universe is whatever set of modules appears anywhere in
    the output, which the gene x pathway membership matrix reuses as its own
    column axis (see modular_ko.py / docs/pathway_features.md).
    """
    if not contigs_tsv.exists():
        raise FileNotFoundError(f"give_completeness output not found at {contigs_tsv}")

    long_df = pd.read_csv(contigs_tsv, sep="\t", usecols=["contig", "module_accession", "completeness"])
    wide = long_df.pivot_table(index="contig", columns="module_accession", values="completeness", aggfunc="first")
    wide = wide.reindex(index=accessions).fillna(0.0) / 100.0
    wide.index.name = "ncbi_assembly_accession"
    wide.columns.name = None

    log.info("Built completeness matrix: %d strain(s) x %d module(s)", *wide.shape)
    return wide
