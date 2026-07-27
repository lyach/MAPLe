#!/usr/bin/env python3
"""
build_modular_gene_pathway_matrix.py
=======================================
Local, post-annotation step for the modular pathway pipeline. ADDITIVE:
does not read, write, or modify kegg_pathway_completeness.csv or
checkm2_quality.csv, or their generation logic.

Parses PyKofamSearch's ko_hits.tsv plus Pyrodigal's per-genome protein
FASTAs into ONE gene x KEGG-module binary membership matrix PER STRAIN:

    data/external/modular_pathway/gene_pathway_membership/gene_pathway_membership_{strain}.csv
        Rows: every gene Pyrodigal predicted for that strain (including
              genes with no KO hit at all -- an all-zero row, not omitted).
        Columns: KEGG module IDs (e.g. M00001) -- the SAME module set
              build_modular_pathway_features.py's kegg_pathway_completeness
              .csv actually has as columns (read from that file's header).
        Values: binary 0/1. MULTI-LABEL -- a gene's KO can and often does
              belong to more than one module; all applicable modules get a
              1, not just one.

    data/external/modular_pathway/gene_pathway_membership/kegg_module_lookup.csv
        module_id -> human-readable module_name, written once, sourced
        directly from give_completeness's own reference data.

Usage:
    python scripts/build_modular_gene_pathway_matrix.py config/dataset/modular_gene_pathway_membership.yaml
    python scripts/build_modular_gene_pathway_matrix.py config/dataset/modular_gene_pathway_membership.yaml --force
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
from src.pathway_features.modular_kegg_modules import load_ko_module_mapping
from src.pathway_features.modular_ko import load_ko_hits, kos_for_accession, load_gene_universe
from src.pathway_features.modular_gene_pathway_membership import build_gene_module_matrix

log = logging.getLogger("build_modular_gene_pathway_matrix")

_REQUIRED_KEYS = {"strains_metadata", "ko_annotation_dir", "genes_dir", "completeness_csv", "output_dir"}
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(strain_name: str) -> str:
    return _SLUG_PATTERN.sub("_", strain_name.strip().lower()).strip("_")


def load_config(config_path: Path) -> dict:
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
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output CSVs")
    args = parser.parse_args(argv)

    log_path = setup_logging(pf_config.ROOT)
    cfg = load_config(args.config)

    strains_metadata = pf_config.ROOT / cfg["strains_metadata"]
    ko_hits_tsv = pf_config.ROOT / cfg["ko_annotation_dir"] / "ko_hits.tsv"
    genes_dir = pf_config.ROOT / cfg["genes_dir"]
    completeness_csv = pf_config.ROOT / cfg["completeness_csv"]
    output_dir = pf_config.ROOT / cfg["output_dir"]
    lookup_csv = output_dir / pf_config.MODULE_LOOKUP_CSV_NAME

    if not strains_metadata.exists():
        sys.exit(f"Error: strains_metadata.csv not found at {strains_metadata}")
    if not completeness_csv.exists():
        sys.exit(
            f"Error: {completeness_csv} not found -- run build_modular_pathway_features.py "
            "first (this script reuses its module-ID scope, see docstring)."
        )

    strains_df = pd.read_csv(strains_metadata)

    # ── 1. Module-ID column universe: reuse kegg_pathway_completeness.csv's
    #        own columns (this branch's give_completeness run), joined to
    #        readable names from give_completeness's own reference data ────
    _, module_id_to_name = load_ko_module_mapping()
    with completeness_csv.open(encoding="utf-8") as f:
        module_ids = next(csv.reader(f))[1:]
    unknown = [m for m in module_ids if m not in module_id_to_name]
    if unknown:
        sys.exit(
            f"Error: {len(unknown)} module ID(s) in {completeness_csv} have no name in "
            f"give_completeness's reference data -- unexpected, investigate before proceeding: {unknown}"
        )

    # ── 2. Output paths up front, for the --force guard ─────────────────────
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
            "(the annotation run is expensive/remote, so this script won't silently clobber existing output)."
        )

    ko_hits = load_ko_hits(ko_hits_tsv)

    # ── 3. Validate every strain FIRST (full gene universe must exist),
    #        before writing anything -- all-or-nothing, so a failure never
    #        leaves a partial/misleading result set on disk ────────────────
    ko_to_modules, _ = load_ko_module_mapping()
    gene_universe_by_strain: dict[str, list[str]] = {}
    hits_by_strain: dict[str, pd.Series] = {}
    failures = []
    for _, row in strains_df.iterrows():
        strain, accession = row["strain"], row["ncbi_assembly_accession"]
        try:
            gene_universe_by_strain[strain] = load_gene_universe(genes_dir, accession)
        except ValueError as exc:
            log.error("Strain %r (%s): %s", strain, accession, exc)
            failures.append((strain, accession, str(exc)))
            continue
        hits_by_strain[strain] = kos_for_accession(ko_hits, accession)

    if failures:
        sys.exit(
            f"Error: {len(failures)} strain(s) failed to parse -- aborting "
            f"before writing any output: {[s for s, _, _ in failures]}"
        )

    # ── 4. All strains validated -- now build and write ─────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    for strain, gene_universe in gene_universe_by_strain.items():
        accession = strains_df.loc[strains_df["strain"] == strain, "ncbi_assembly_accession"].iloc[0]
        matrix = build_gene_module_matrix(gene_universe, hits_by_strain[strain], ko_to_modules, module_ids)
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

    # ── 5. Write the module_id -> module_name lookup, once ─────────────────
    lookup_df = pd.Series({m: module_id_to_name[m] for m in module_ids}, name="module_name")
    lookup_df.index.name = "module_id"
    lookup_df.sort_index().to_csv(lookup_csv)
    print(f"\nWrote {lookup_csv} ({len(module_ids)} module(s))")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    coverage_df = pd.DataFrame(coverage_rows)
    print(f"\n{len(coverage_df)} strain(s) processed, {len(module_ids)} KEGG module column(s) each.")
    print(f"Total genes: {coverage_df['total_genes'].sum()}")
    print(f"Genes with >=1 module hit: {coverage_df['genes_with_module_hit'].sum()}")
    print(f"Genes all-zero: {coverage_df['genes_all_zero'].sum()}")
    print(f"Log file: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
