#!/usr/bin/env python3
"""Writes data/external/strain_phylum.csv (strain, ncbi_taxon_id, phylum).

USAGE:
    python scripts/fetch_strain_phylum.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strain_metadata import config
from src.strain_metadata.logging_setup import setup_logging
from src.strain_metadata.credentials import resolve_ncbi_credentials
from src.strain_metadata.taxonomy import fetch_phyla

log = logging.getLogger("fetch_strain_phylum")


def main() -> int:
    log_path = setup_logging(config.ROOT)

    if not config.OUTPUT_CSV.exists():
        log.error("%s not found -- run scripts/get_strain_metadata.py first.", config.OUTPUT_CSV)
        return 1

    with config.OUTPUT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    strain_to_taxon = {}
    for row in rows:
        taxon_id_raw = row.get("ncbi_taxon_id", "").strip()
        if not taxon_id_raw:
            log.warning("Strain %r has no ncbi_taxon_id -- skipped.", row["strain"])
            continue
        strain_to_taxon[row["strain"]] = int(taxon_id_raw)

    email, api_key = resolve_ncbi_credentials()
    phyla_by_taxon = fetch_phyla(list(strain_to_taxon.values()), email, api_key)

    config.PHYLUM_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.PHYLUM_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strain", "ncbi_taxon_id", "phylum"])
        for strain, taxon_id in strain_to_taxon.items():
            writer.writerow([strain, taxon_id, phyla_by_taxon.get(taxon_id, "")])

    n_found = sum(1 for t in strain_to_taxon.values() if t in phyla_by_taxon)
    print(f"Wrote {config.PHYLUM_CSV} ({n_found}/{len(strain_to_taxon)} strain(s) matched to a phylum)")
    print(f"Log file: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
