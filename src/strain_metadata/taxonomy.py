"""Looks up phylum for strains by NCBI taxon ID."""

from __future__ import annotations

import logging
import time
from typing import Optional

from Bio import Entrez

log = logging.getLogger("taxonomy")

_BATCH_SIZE = 50  # NCBI efetch batch size


def fetch_phyla(
    taxon_ids: list[int],
    email: str,
    api_key: Optional[str] = None,
) -> dict[int, str]:
    """Return {taxon_id: phylum_name}; unresolvable IDs are simply absent."""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    request_delay = 0.11 if api_key else 0.34

    unique_ids = sorted(set(taxon_ids))
    phyla: dict[int, str] = {}

    for start in range(0, len(unique_ids), _BATCH_SIZE):
        batch = unique_ids[start : start + _BATCH_SIZE]
        handle = Entrez.efetch(db="taxonomy", id=",".join(str(t) for t in batch))
        records = Entrez.read(handle)
        handle.close()
        time.sleep(request_delay)

        for record in records:
            taxon_id = int(record["TaxId"])
            phylum = next(
                (t["ScientificName"] for t in record.get("LineageEx", []) if t["Rank"] == "phylum"),
                None,
            )
            if phylum:
                phyla[taxon_id] = phylum
            else:
                log.warning("No phylum rank found in lineage for taxon %d", taxon_id)

    missing = set(unique_ids) - set(phyla)
    if missing:
        log.warning("%d taxon ID(s) had no NCBI taxonomy record or no phylum: %s", len(missing), sorted(missing))

    return phyla
