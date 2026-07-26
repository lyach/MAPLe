"""
gene_pathway_membership.py
============================
Builds a per-strain, gene x KEGG-module BINARY MEMBERSHIP matrix from DRAM's
raw annotation output (annotations.tsv, the per-gene table -- NOT the
genome-level distillate/product.tsv this package's other modules read).

Rows are individual predicted genes (every ORF DRAM predicted for that
genome, including ones with no KO hit at all -- confirmed empirically that
annotations.tsv already has one row per predicted ORF regardless of hit
status, by cross-checking row counts against genes.faa). Columns are the 32
real KEGG module IDs covered by kegg_modules.py (see that module's docstring
for why the other 66 columns in kegg_pathway_completeness.csv aren't real
KEGG modules and aren't covered here).

MULTI-LABEL, not one-hot: a single gene's KO can legitimately belong to
several KEGG modules at once (e.g. K00024, malate dehydrogenase, belongs to
9 different modules in DRAM's own reference data -- it's a central-carbon
enzyme that feeds the TCA cycle, glyoxylate cycle, C4 pathways, and more).
Forcing a single pathway assignment per gene would misrepresent real
biology, so every applicable module gets marked 1 for that gene, not just
the "best" or first match.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.gene_pathway_membership")


def load_annotations(annotations_tsv: Path) -> pd.DataFrame:
    """Load DRAM's combined, multi-genome annotations.tsv. Returns a
    DataFrame indexed by gene_id (DRAM's own '{accession}_{scaffold}_{gene_
    position}' identifier) with columns 'fasta' (genome accession) and
    'ko_id' (empty string, not NaN, for genes with no KO hit).
    """
    if not annotations_tsv.exists():
        raise FileNotFoundError(
            f"DRAM annotations.tsv not found at {annotations_tsv} -- this is "
            "the raw per-gene annotation output, a DIFFERENT rsync target "
            "than distillate/ (which only has the genome-level summary). "
            "rsync $DRAM_ROOT/output/annotation/ back from the cluster. "
            "See docs/pathway_features.md."
        )

    # The gene-id column's header is blank in the file; pandas auto-names it
    # "Unnamed: 0" rather than "" once parsed, so select positionally (first
    # column) instead of relying on that generated name.
    df = pd.read_csv(annotations_tsv, sep="\t")
    df = df.set_index(df.columns[0])[["fasta", "ko_id"]]
    df.index.name = "gene_id"
    df["ko_id"] = df["ko_id"].fillna("")

    log.info(
        "Loaded annotations.tsv: %d gene(s) across %d genome(s)",
        len(df), df["fasta"].nunique(),
    )
    return df


def genes_for_accession(annotations: pd.DataFrame, accession: str) -> pd.Series:
    """Return gene_id -> ko_id for one genome. Raises if the accession has
    zero rows in annotations.tsv at all (a genuinely missing/malformed
    strain) -- this is distinct from, and does not raise on, a strain whose
    genes are all valid rows but happen to have no KO hits, which is normal
    and produces an all-zero matrix, not an error.
    """
    subset = annotations.loc[annotations["fasta"] == accession, "ko_id"]
    if subset.empty:
        raise ValueError(
            f"No genes found for accession {accession!r} in annotations.tsv -- "
            "DRAM may have failed on this genome, or the accession is wrong."
        )
    return subset


def build_gene_module_matrix(
    gene_ko: pd.Series,
    ko_to_modules: dict[str, list[str]],
    module_ids: list[str],
) -> pd.DataFrame:
    """Binary gene_id x module_id DataFrame. 1 where that gene's KO belongs
    to that module -- MULTI-LABEL, a gene can have more than one 1 in its
    row. Genes with no KO hit, or whose KO isn't part of any of these 32
    modules (most KOs aren't -- KOfam alone calls ~2,500+ distinct KOs
    across a genome set, and only ~2,445 KOs total appear anywhere in DRAM's
    module reference data), get an explicit all-zero row -- never dropped.
    Every strain's matrix has the identical `module_ids` column set (even a
    module with zero hits in this particular strain still appears, all
    zero), so matrices are directly comparable/stackable across strains.
    """
    pairs = [
        (gene_id, module_id)
        for gene_id, ko in gene_ko.items()
        if ko
        for module_id in ko_to_modules.get(ko, ())
    ]

    if pairs:
        pair_df = pd.DataFrame(pairs, columns=["gene_id", "module_id"]).drop_duplicates()
        matrix = pd.crosstab(pair_df["gene_id"], pair_df["module_id"])
    else:
        matrix = pd.DataFrame(index=pd.Index([], name="gene_id"))

    # Reindex to the FULL gene universe (every predicted gene, hit or not)
    # and the FULL fixed module-id column set -- this is what turns "no
    # pairs recorded" into explicit zeros rather than missing rows/columns.
    matrix = matrix.reindex(index=gene_ko.index, columns=module_ids, fill_value=0)
    matrix = matrix.clip(upper=1).astype(int)
    matrix.index.name = "gene_id"
    return matrix
