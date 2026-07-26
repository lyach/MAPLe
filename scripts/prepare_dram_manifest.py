#!/usr/bin/env python3
"""
prepare_dram_manifest.py
=========================
Local, pre-DRAM step. Cross-validates data/external/strains_metadata.csv
against the FASTA files in data/external/fasta/, and writes:

    data/interim/dram/fasta_manifest.csv   -- strain, accession, fasta filename

Prints the rsync command to ship the FASTA files + scripts/hpc/ to DRAC.
This is the local half of a two-environment pipeline -- the actual DRAM
annotation runs on a DRAC HPC cluster, not here. See docs/pathway_features.md.

USAGE:
    python scripts/prepare_dram_manifest.py
    python scripts/prepare_dram_manifest.py --remote-host narval.alliancecan.ca --remote-dir scratch/dram

NOTE: --remote-dir must not start with '~' -- if passed as its own shell
argument, your LOCAL shell expands a leading '~' to your own home directory
before Python ever sees it (see the --remote-dir help text below for why).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pathway_features import config
from src.common.logging_setup import setup_logging
from src.pathway_features.manifest import build_manifest, write_manifest
from src.pathway_features.drac_credentials import resolve_drac_account

log = logging.getLogger("prepare_dram_manifest")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strains-metadata", type=Path, default=config.STRAINS_METADATA_CSV,
                   help=f"Path to strains_metadata.csv (default: {config.STRAINS_METADATA_CSV})")
    p.add_argument("--fasta-dir", type=Path, default=config.FASTA_DIR,
                   help=f"Directory of genome FASTA files (default: {config.FASTA_DIR})")
    p.add_argument("--output", type=Path, default=config.FASTA_MANIFEST_CSV,
                   help=f"Path to write the manifest CSV (default: {config.FASTA_MANIFEST_CSV})")
    p.add_argument("--remote-host", default="<your-drac-cluster>",
                   help="DRAC hostname to fill in the printed rsync command (e.g. beluga.alliancecan.ca)")
    p.add_argument("--remote-dir", default="scratch/dram",
                   help="Destination directory on the remote host for the printed rsync command "
                        "(relative to the remote $HOME -- avoid a leading '~': if this flag is "
                        "passed as its own shell argument, your LOCAL shell tilde-expands it to "
                        "your own home directory before Python ever sees it, since '~' only "
                        "escapes expansion when it's not at the start of a word)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log_path = setup_logging(config.ROOT)

    manifest = build_manifest(args.strains_metadata, args.fasta_dir)
    if manifest.empty:
        log.error("No strains matched to a FASTA file -- nothing to write.")
        return 1

    write_manifest(manifest, args.output)

    try:
        account = resolve_drac_account()
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    print(f"\nManifest: {len(manifest)} strain(s) -> {args.output}")
    print(f"Log file: {log_path}")
    print("\nTo ship the FASTA files and HPC scripts to DRAC, run:")
    print(f"  rsync -avz {args.fasta_dir}/ {args.remote_host}:{args.remote_dir}/fasta/")
    print(f"  rsync -avz {config.ROOT / 'scripts' / 'hpc'}/ {args.remote_host}:{args.remote_dir}/hpc/")
    print("\nThen on the cluster (these use $HOME-relative paths, so you can run them from anywhere):")
    print(f"  sbatch --account={account} --mem=128G --cpus-per-task=8 --time=6:00:00 \\")
    print('    --job-name=dram-setup --output=%x-%j.out \\')
    print(f'    --wrap="bash $HOME/{args.remote_dir}/hpc/setup_dram_env.sh"')
    print(f"  sbatch --account={account} $HOME/{args.remote_dir}/hpc/run_dram_annotate.sbatch")
    print("See docs/pathway_features.md for the full workflow.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
