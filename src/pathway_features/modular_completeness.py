"""Builds the strain x KEGG-module completeness matrix via give_completeness."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.modular_completeness")


def write_give_completeness_input(accession_to_kos: dict[str, list[str]], out_tsv: Path) -> None:
    """Write give_completeness's per-genome input format: accession + KO IDs, tab-separated."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8") as f:
        for accession, kos in accession_to_kos.items():
            f.write("\t".join([accession, *kos]) + "\n")
    log.info("Wrote give_completeness input for %d genome(s) -> %s", len(accession_to_kos), out_tsv)


def run_give_completeness(input_tsv: Path, outdir: Path, outprefix: str = "modular") -> Path:
    """Run give_completeness locally, return the per-contig output path."""
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
    """Reshape give_completeness's sparse output into a dense strain x
    module [0, 1] matrix, zero-filling genomes/modules with no rows.
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
