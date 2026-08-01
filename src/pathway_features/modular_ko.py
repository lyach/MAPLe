"""
modular_ko.py
==============
Loads this pipeline's KO annotation output (PyKofamSearch's ko_hits.tsv,
long format: one row per (protein, KO) hit -- a protein CAN have more than
one row if it genuinely hits multiple distinct KOs, confirmed in
PyKofamSearch's own README) and the per-genome protein universe from
Pyrodigal's .faa headers.

The .faa headers are needed because ko_hits.tsv only contains genes that DID
hit a KO -- there is no separate "every predicted gene" table, so the full
per-strain gene universe (including zero-hit genes, required for the gene x
pathway membership matrix's all-zero-row spec) has to come from Pyrodigal's
own output instead.

Gene/protein IDs are '{accession}_{contig}_{gene_number}', produced by
run_modular_annotate.sbatch's header-prefixing step (Pyrodigal's own
default header has no genome identifier at all). Matched by prefix against
the accession, not split on '_', since both accessions (e.g.
'GCF_002008145.1') and contig names (e.g. 'NZ_MRAC01000001.1') already
contain underscores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.modular_ko")


def load_ko_hits(ko_hits_tsv: Path) -> pd.DataFrame:
    """Load PyKofamSearch's ko_hits.tsv. Returns just id_protein/id_ko --
    the other columns (threshold, score, e-value, definition,
    enzyme_commission) aren't needed for module-membership/completeness
    purposes.
    """
    if not ko_hits_tsv.exists():
        raise FileNotFoundError(
            f"PyKofamSearch output not found at {ko_hits_tsv} -- run the "
            "pipeline on DRAC first and copy "
            "$MODULAR_ROOT/output/ko_annotation/ back into "
            "data/external/modular_pathway/ko_annotation/. See "
            "docs/modular_pathway_features.md."
        )
    df = pd.read_csv(ko_hits_tsv, sep="\t", usecols=["id_protein", "id_ko"])
    log.info("Loaded ko_hits.tsv: %d hit(s) across %d protein(s)", len(df), df["id_protein"].nunique())
    return df


def kos_for_accession(ko_hits: pd.DataFrame, accession: str) -> pd.Series:
    """gene_id -> ko_id for one genome's hits (LONG -- a gene_id can repeat
    if that protein hit multiple KOs). Does NOT raise on zero rows: a genome
    with real predicted genes but genuinely zero KO hits is a normal,
    non-error outcome here (ko_hits.tsv only ever contains hits, so it can't
    distinguish "no hits" from "genome missing entirely" -- use
    load_gene_universe for that check instead).
    """
    subset = ko_hits.loc[ko_hits["id_protein"].str.startswith(accession + "_")]
    return subset.set_index("id_protein")["id_ko"]


def load_gene_universe(genes_dir: Path, accession: str) -> list[str]:
    """Full list of gene IDs Pyrodigal predicted for one genome, read from
    that genome's protein FASTA header lines (>gene_id ...). Raises if the
    file is missing or has zero genes -- a genuinely failed/missing genome,
    distinct from (and the only real failure case for) a genome with
    predicted genes but no KO hits.

    The per-genome .faa files on disk still have Pyrodigal's RAW headers
    (e.g. '>NZ_AP023410.1_1 # 186 # ...', no accession) -- only the
    COMBINED all_proteins.faa that PyKofamSearch actually ran on got
    accession-prefixed (see run_modular_annotate.sbatch's sed step). Gene
    IDs returned here are prefixed the same way ('{accession}_...') so they
    match ko_hits.tsv's id_protein values exactly (verified directly
    against real output).
    """
    faa_path = genes_dir / f"{accession}.faa"
    if not faa_path.exists():
        raise ValueError(
            f"No protein FASTA found for accession {accession!r} at {faa_path} -- "
            "Pyrodigal may have failed on this genome, or the accession is wrong."
        )
    gene_ids = [
        f"{accession}_{line[1:].split(None, 1)[0]}"
        for line in faa_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
    ]
    if not gene_ids:
        raise ValueError(f"{faa_path} has zero predicted genes -- Pyrodigal may have failed on this genome.")
    return gene_ids
