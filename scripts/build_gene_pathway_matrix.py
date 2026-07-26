#!/usr/bin/env python3
"""
build_gene_pathway_matrix.py
==============================
Local, post-DRAM step. ADDITIVE to the existing pathway-features pipeline --
does not read, write, or modify kegg_pathway_completeness.csv or any of its
generation logic (build_pathway_features.py, distillate.py, matrix.py,
pca.py are all untouched by this script).

Parses DRAM's raw per-gene annotation output (annotations.tsv -- NOT the
genome-level distillate/product.tsv) into ONE gene x KEGG-module binary
membership matrix PER STRAIN:

    data/external/dram/gene_pathway_membership/gene_pathway_membership_{strain}.csv
        Rows: every gene DRAM predicted for that strain (including genes
              with no KO hit at all -- an all-zero row, not omitted).
        Columns: KEGG module IDs (e.g. M00001) -- specifically the same
              ~31 real-KEGG-module columns already used in
              kegg_pathway_completeness.csv (see src/pathway_features/
              kegg_modules.py for why that's a subset of its 98 columns,
              not all of them -- most of them are DRAM-curated categories
              with no official KEGG module ID).
        Values: binary 0/1. MULTI-LABEL -- a gene's KO can and often does
              belong to more than one module; all applicable modules get a
              1, not just one (see kegg_modules.py's docstring for a
              concrete example: K00024 belongs to 9 different modules).

    data/external/dram/gene_pathway_membership/kegg_module_lookup.csv
        module_id -> human-readable module_name, written once (not
        per-strain) -- column headers in the matrices above are bare module
        IDs, not names, so this is where the readable labels live instead.

Requires the RAW annotation/ output rsynced back from DRAC, not just
distillate/ (see docs/pathway_features.md):
    rsync -avz Nibi:scratch/dram/output/annotation/ data/external/dram/annotation/

Usage:
    python scripts/build_gene_pathway_matrix.py config/dataset/gene_pathway_membership.yaml
    python scripts/build_gene_pathway_matrix.py config/dataset/gene_pathway_membership.yaml --force
    python scripts/build_gene_pathway_matrix.py config/dataset/gene_pathway_membership.yaml --refresh-mapping
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pathway_features import config as pf_config
from src.common.logging_setup import setup_logging
from src.pathway_features.kegg_modules import load_ko_module_mapping, modules_in_completeness_matrix
from src.pathway_features.gene_pathway_membership import load_annotations, genes_for_accession, build_gene_module_matrix

log = logging.getLogger("build_gene_pathway_matrix")

_REQUIRED_KEYS = {"strains_metadata", "annotation_dir", "distillate_dir", "product_tsv", "output_dir"}

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(strain_name: str) -> str:
    """'Escherichia coli K12' -> 'escherichia_coli_k12', for filenames."""
    slug = _SLUG_PATTERN.sub("_", strain_name.strip().lower()).strip("_")
    return slug


def load_config(config_path: Path) -> dict:
    """Load and validate the YAML config file."""
    if not config_path.exists():
        sys.exit(f"Error: config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        sys.exit(f"Error: config is missing required key(s): {sorted(missing)}")

    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to the YAML config file (e.g. config/dataset/gene_pathway_membership.yaml)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output CSVs")
    parser.add_argument(
        "--refresh-mapping", action="store_true",
        help="Re-download DRAM's module_step_form.tsv/etc_module_database.tsv "
             "reference sheets instead of using the local cache.",
    )
    args = parser.parse_args(argv)

    log_path = setup_logging(pf_config.ROOT)

    cfg = load_config(args.config)

    strains_metadata = pf_config.ROOT / cfg["strains_metadata"]
    annotation_dir = pf_config.ROOT / cfg["annotation_dir"]
    annotations_tsv = annotation_dir / "annotations.tsv"
    product_tsv = pf_config.ROOT / cfg["distillate_dir"] / cfg["product_tsv"]
    output_dir = pf_config.ROOT / cfg["output_dir"]
    lookup_csv = output_dir / pf_config.MODULE_LOOKUP_CSV_NAME

    if not strains_metadata.exists():
        sys.exit(f"Error: strains_metadata.csv not found at {strains_metadata}")
    if not product_tsv.exists():
        sys.exit(
            f"Error: {product_tsv} not found -- run build_pathway_features.py "
            "first (this script reuses its module-ID scope, see docstring)."
        )

    strains_df = pd.read_csv(strains_metadata)

    # ── 1. Determine the module-ID column universe (shared with the existing completeness matrix) ──
    ko_to_modules, module_id_to_name = load_ko_module_mapping(refresh=args.refresh_mapping)
    with product_tsv.open(encoding="utf-8") as f:
        product_columns = next(csv.reader(f, delimiter="\t"))[1:]
    module_id_to_name = modules_in_completeness_matrix(product_columns, module_id_to_name)
    module_ids = sorted(module_id_to_name.keys())

    # ── 2. Figure out output paths up front, for the --force guard ─────────
    per_strain_outputs = {
        strain: output_dir / f"gene_pathway_membership_{_slugify(strain)}.csv"
        for strain in strains_df["strain"]
    }
    all_outputs = [lookup_csv, *per_strain_outputs.values()]
    existing = [p for p in all_outputs if p.exists()]
    if existing and not args.force:
        sys.exit(
            f"Error: output file(s) already exist: {[str(p) for p in existing[:5]]}"
            f"{' ...' if len(existing) > 5 else ''}. Pass --force to overwrite "
            "(DRAM runs are expensive/remote, so this script won't silently "
            "clobber existing output)."
        )

    # ── 3. Load the full per-gene annotation table once ─────────────────────
    annotations = load_annotations(annotations_tsv)

    # ── 4. Validate every strain FIRST, before writing anything -- matches
    #        build_pathway_features.py's all-or-nothing pattern: either every
    #        strain resolves and we write a complete result set, or nothing
    #        gets written and the failures are reported together. ─────────
    gene_ko_by_strain: dict[str, pd.Series] = {}
    failures = []
    for _, row in strains_df.iterrows():
        strain, accession = row["strain"], row["ncbi_assembly_accession"]
        try:
            gene_ko_by_strain[strain] = genes_for_accession(annotations, accession)
        except ValueError as exc:
            log.error("Strain %r (%s): %s", strain, accession, exc)
            failures.append((strain, accession, str(exc)))

    if failures:
        sys.exit(
            f"Error: {len(failures)} strain(s) failed to parse -- aborting "
            f"before writing any output: {[s for s, _, _ in failures]}"
        )

    # ── 5. All strains validated -- now build and write ─────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    for strain, gene_ko in gene_ko_by_strain.items():
        accession = strains_df.loc[strains_df["strain"] == strain, "ncbi_assembly_accession"].iloc[0]
        matrix = build_gene_module_matrix(gene_ko, ko_to_modules, module_ids)
        out_path = per_strain_outputs[strain]
        matrix.to_csv(out_path)

        row_sums = matrix.sum(axis=1)
        n_genes = len(matrix)
        n_with_hit = int((row_sums > 0).sum())
        coverage_rows.append({
            "strain": strain, "accession": accession, "total_genes": n_genes,
            "genes_with_module_hit": n_with_hit, "genes_all_zero": n_genes - n_with_hit,
        })
        print(f"Wrote {out_path} ({n_genes} genes x {len(module_ids)} modules, {n_with_hit} genes with >=1 module hit)")

    # ── 6. Write the module_id -> module_name lookup, once ─────────────────
    lookup_df = pd.Series(module_id_to_name, name="module_name")
    lookup_df.index.name = "module_id"
    lookup_df.sort_index().to_csv(lookup_csv)
    print(f"\nWrote {lookup_csv} ({len(module_id_to_name)} module(s))")

    # ── 7. Summary ───────────────────────────────────────────────────────────
    coverage_df = pd.DataFrame(coverage_rows)
    print(f"\n{len(coverage_df)} strain(s) processed, {len(module_ids)} KEGG module column(s) each.")
    print(f"Total genes: {coverage_df['total_genes'].sum()}")
    print(f"Genes with >=1 module hit: {coverage_df['genes_with_module_hit'].sum()}")
    print(f"Genes all-zero: {coverage_df['genes_all_zero'].sum()}")
    print(f"Log file: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
