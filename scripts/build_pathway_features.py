#!/usr/bin/env python3
"""
build_pathway_features.py
==========================
Local, post-DRAM step. Parses DRAM's distillate output (product.tsv) into
the CANONICAL modeling input:

    data/external/kegg_pathway_completeness.csv   -- strain x pathway,
                                                       continuous 0.0-1.0
                                                       completeness fractions

This is the file downstream feature pipelines should read. Two additional
outputs are available with --emit-diagnostics, OFF by default, and are NOT
modeling inputs:

    data/external/kegg_pathway_presence.csv        -- binary presence/absence,
                                                        thresholded from the
                                                        completeness matrix,
                                                        for manual inspection
    data/external/eda_pca_exploratory.csv           -- a single PCA fit across
                                                         ALL strains, for
                                                         visualization only.
                                                         MUST NOT be joined
                                                         into training data:
                                                         a global PCA fit
                                                         leaks test-strain
                                                         information. If the
                                                         modeling pipeline
                                                         wants pathway
                                                         features reduced, it
                                                         must fit its own PCA
                                                         inside the CV loop,
                                                         on training strains
                                                         only.
    data/external/eda_pca_exploratory_loadings.csv

Run this only after DRAM has been run on DRAC and its distillate/ output has
been copied back locally -- see docs/pathway_features.md for the full
two-environment workflow. This script does not invoke DRAM itself.

Usage:
    python scripts/build_pathway_features.py config/dataset/pathway_features.yaml
    python scripts/build_pathway_features.py config/dataset/pathway_features.yaml --force
    python scripts/build_pathway_features.py config/dataset/pathway_features.yaml --emit-diagnostics
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
from src.pathway_features.distillate import load_product_tsv, join_strain_names
from src.pathway_features.matrix import build_presence_matrix
from src.pathway_features.pca import run_eda_pca
from src.pathway_features.validation import validate_completeness_matrix

log = logging.getLogger("build_pathway_features")

_REQUIRED_KEYS = {"strains_metadata", "distillate_dir", "product_tsv", "output_dir"}


def load_config(config_path: Path) -> dict:
    """Load and validate the YAML config file."""
    if not config_path.exists():
        sys.exit(f"Error: config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        sys.exit(f"Error: config is missing required key(s): {sorted(missing)}")

    cfg.setdefault("completeness_threshold", pf_config.DEFAULT_COMPLETENESS_THRESHOLD)
    cfg.setdefault("pca_n_components", pf_config.DEFAULT_PCA_N_COMPONENTS)
    cfg.setdefault("pca_random_state", pf_config.DEFAULT_PCA_RANDOM_STATE)

    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to the YAML config file (e.g. config/dataset/pathway_features.yaml)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output CSVs")
    parser.add_argument(
        "--emit-diagnostics", action="store_true",
        help="Also write kegg_pathway_presence.csv (binary, thresholded) and "
             "eda_pca_exploratory.csv (a single global PCA fit) for manual "
             "inspection/visualization. Neither is a modeling input -- see "
             "this script's docstring. Off by default.",
    )
    args = parser.parse_args(argv)

    log_path = setup_logging(pf_config.ROOT)

    cfg = load_config(args.config)

    strains_metadata = pf_config.ROOT / cfg["strains_metadata"]
    distillate_dir = pf_config.ROOT / cfg["distillate_dir"]
    product_tsv = distillate_dir / cfg["product_tsv"]
    output_dir = pf_config.ROOT / cfg["output_dir"]

    completeness_csv = output_dir / pf_config.COMPLETENESS_CSV_NAME
    presence_csv = output_dir / pf_config.PRESENCE_CSV_NAME
    eda_pca_csv = output_dir / pf_config.EDA_PCA_CSV_NAME
    eda_loadings_csv = output_dir / pf_config.EDA_PCA_LOADINGS_CSV_NAME

    outputs = [completeness_csv]
    if args.emit_diagnostics:
        outputs += [presence_csv, eda_pca_csv, eda_loadings_csv]

    existing = [p for p in outputs if p.exists()]
    if existing and not args.force:
        sys.exit(
            f"Error: output file(s) already exist: {[str(p) for p in existing]}. "
            "Pass --force to overwrite (DRAM runs are expensive/remote, so this "
            "script won't silently clobber existing output)."
        )

    # ── 1. Load DRAM's distillate output ────────────────────────────────────
    completeness = load_product_tsv(product_tsv)
    completeness = join_strain_names(completeness, strains_metadata)

    expected_strains = pd.read_csv(strains_metadata)["strain"].tolist()
    validate_completeness_matrix(completeness, expected_strains)

    # Drop constant columns flagged by validation -- no signal for PCA/matrix.
    non_constant_cols = [c for c in completeness.columns if completeness[c].nunique(dropna=False) > 1]
    completeness = completeness[non_constant_cols]

    # ── 2. Write the canonical output ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    completeness.to_csv(completeness_csv)
    print(f"\nWrote {completeness_csv} ({completeness.shape[0]} strains x {completeness.shape[1]} pathways) -- canonical modeling input")

    # ── 3. Optional diagnostics (NOT modeling inputs) ───────────────────────
    if args.emit_diagnostics:
        presence = build_presence_matrix(completeness, cfg["completeness_threshold"])
        presence.to_csv(presence_csv)
        print(f"Wrote {presence_csv} ({presence.shape[0]} strains x {presence.shape[1]} pathways) -- diagnostic only, not for modeling")

        scores, loadings = run_eda_pca(completeness, cfg["pca_n_components"], cfg["pca_random_state"])
        scores.to_csv(eda_pca_csv)
        loadings.to_csv(eda_loadings_csv)
        print(f"Wrote {eda_pca_csv} ({scores.shape[0]} strains x {scores.shape[1]} components) -- EDA only, global fit, DO NOT join into training data")
        print(f"Wrote {eda_loadings_csv} ({loadings.shape[0]} components x {loadings.shape[1]} pathways) -- EDA only")

    print(f"Log file: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
