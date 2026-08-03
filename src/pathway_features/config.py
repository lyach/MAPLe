"""Paths and constants for the modular pathway-features pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # pathway_features/ -> src/ -> root

# --- inputs ---
STRAINS_METADATA_CSV = ROOT / "data" / "external" / "strains_metadata.csv"
FASTA_DIR = ROOT / "data" / "external" / "fasta"

MODULE_LOOKUP_CSV_NAME = "kegg_module_lookup.csv"

# --- DRAC credentials (gitignored) ---
DRAC_CREDENTIALS_FILE = ROOT / ".drac_credentials.json"

# --- modular pipeline paths ---
MODULAR_INTERIM_DIR = ROOT / "data" / "interim" / "modular_pathway"
MODULAR_FASTA_MANIFEST_CSV = MODULAR_INTERIM_DIR / "fasta_manifest.csv"
MODULAR_KO_ANNOTATION_DIR = ROOT / "data" / "external" / "modular_pathway" / "ko_annotation"
MODULAR_CHECKM2_DIR = ROOT / "data" / "external" / "modular_pathway" / "checkm2"
MODULAR_GENE_PATHWAY_OUTPUT_DIR = ROOT / "data" / "external" / "modular_pathway" / "gene_pathway_membership"
MODULAR_GENES_DIR = ROOT / "data" / "external" / "modular_pathway" / "genes"
MODULAR_CHECKM2_QUALITY_CSV_NAME = "checkm2_quality.csv"
MODULAR_COMPLETENESS_CSV_NAME = "kegg_pathway_completeness.csv"
