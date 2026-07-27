#!/usr/bin/env python3
"""
build_modular_pathway_features.py
====================================
Local, post-annotation step for the modular pathway pipeline. Produces the
two REQUIRED outputs from this pipeline's spec:

    data/external/modular_pathway/kegg_pathway_completeness.csv
        CANONICAL modeling input: strain x KEGG-module, continuous [0, 1]
        completeness fractions, from EBI's give_completeness run locally
        against PyKofamSearch's KO hits (see src/pathway_features/
        modular_completeness.py -- give_completeness itself is invoked here,
        not on DRAC, since it's fast and needs no database).

    data/external/modular_pathway/checkm2_quality.csv
        REQUIRED confounder feature: strain x [checkm2_completeness,
        checkm2_contamination], [0, 1], from CheckM2's quality_report.tsv.
        Independent of the KEGG-module completeness matrix above -- CheckM2
        measures genome assembly/annotation quality, not pathway coverage.

CheckM2's quality_report.tsv also doubles as this script's per-strain
validation: every genome the pipeline actually processed gets a row there
regardless of KO-hit outcome, so a strain missing from it means the
pipeline genuinely failed/skipped that genome -- distinct from (and not
raised on) a genome with zero KO hits, which is a normal, valid all-zero
completeness row, not an error.

Requires ko_annotation/ (PyKofamSearch's ko_hits.tsv) and checkm2/
(quality_report.tsv) rsynced back from DRAC first:
    rsync -avz Nibi:scratch/modular_pathway/output/ko_annotation/ data/external/modular_pathway/ko_annotation/
    rsync -avz Nibi:scratch/modular_pathway/output/checkm2/quality_report.tsv data/external/modular_pathway/checkm2/

Usage:
    python scripts/build_modular_pathway_features.py config/dataset/modular_pathway_features.yaml
    python scripts/build_modular_pathway_features.py config/dataset/modular_pathway_features.yaml --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pathway_features import config as pf_config
from src.common.logging_setup import setup_logging
from src.pathway_features.strain_names import join_strain_names
from src.pathway_features.checkm2_quality import load_quality_report
from src.pathway_features.modular_ko import load_ko_hits, kos_for_accession
from src.pathway_features.modular_completeness import (
    write_give_completeness_input, run_give_completeness, build_completeness_matrix,
)

log = logging.getLogger("build_modular_pathway_features")

_REQUIRED_KEYS = {"strains_metadata", "ko_annotation_dir", "checkm2_dir", "output_dir"}


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
    quality_report_tsv = pf_config.ROOT / cfg["checkm2_dir"] / "quality_report.tsv"
    output_dir = pf_config.ROOT / cfg["output_dir"]

    completeness_csv = output_dir / pf_config.MODULAR_COMPLETENESS_CSV_NAME
    checkm2_csv = output_dir / pf_config.MODULAR_CHECKM2_QUALITY_CSV_NAME

    existing = [p for p in (completeness_csv, checkm2_csv) if p.exists()]
    if existing and not args.force:
        sys.exit(
            f"Error: output file(s) already exist: {[str(p) for p in existing]}. "
            "Pass --force to overwrite (the annotation run is expensive/remote, "
            "so this script won't silently clobber existing output)."
        )

    if not strains_metadata.exists():
        sys.exit(f"Error: strains_metadata.csv not found at {strains_metadata}")
    strains_df = pd.read_csv(strains_metadata)
    accessions = strains_df["ncbi_assembly_accession"].tolist()

    # ── 1. CheckM2 confounder feature -- also this script's per-strain
    #        validation (see module docstring) ─────────────────────────────
    quality = load_quality_report(quality_report_tsv)
    missing = sorted(set(accessions) - set(quality.index))
    if missing:
        strain_names = strains_df.set_index("ncbi_assembly_accession")["strain"]
        sys.exit(
            f"Error: {len(missing)} strain(s) missing from CheckM2's quality_report.tsv "
            f"-- the pipeline may have failed on them; aborting before writing any "
            f"output: {[strain_names[a] for a in missing]}"
        )
    quality_named = join_strain_names(quality, strains_metadata)

    # ── 2. Module completeness via give_completeness (runs locally) ────────
    ko_hits = load_ko_hits(ko_hits_tsv)
    accession_to_kos = {a: sorted(kos_for_accession(ko_hits, a).unique()) for a in accessions}
    n_zero_hit = sum(1 for kos in accession_to_kos.values() if not kos)
    if n_zero_hit:
        log.warning("%d genome(s) have zero KO hits -- valid, produces an all-zero completeness row.", n_zero_hit)

    give_completeness_input = pf_config.MODULAR_INTERIM_DIR / "give_completeness_input.tsv"
    write_give_completeness_input(accession_to_kos, give_completeness_input)
    contigs_tsv = run_give_completeness(give_completeness_input, pf_config.MODULAR_INTERIM_DIR)
    completeness = build_completeness_matrix(contigs_tsv, accessions)
    completeness_named = join_strain_names(completeness, strains_metadata)

    # ── 3. Write ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    completeness_named.to_csv(completeness_csv)
    quality_named.to_csv(checkm2_csv)

    print(f"Wrote {completeness_csv} ({completeness_named.shape[0]} strain(s) x {completeness_named.shape[1]} module(s))")
    print(f"Wrote {checkm2_csv} ({len(quality_named)} strain(s))")
    print(f"Log file: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
