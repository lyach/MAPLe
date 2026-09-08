"""Looks up NCBI taxonomic lineage ranks for strains by taxon ID."""

from __future__ import annotations

import logging
import time
from typing import Optional

from Bio import Entrez
from tqdm import tqdm

log = logging.getLogger("taxonomy")

_BATCH_SIZE = 50  # NCBI efetch batch size

# Ranks recovered into strains_metadata.csv (and strain_phylum.csv for phylum).
LINEAGE_RANKS = ("genus", "family", "order", "phylum")


def _extract_ranks(record: dict) -> dict[str, str]:
    """Pull genus/family/order/phylum from a Taxonomy efetch record.

    Ranks usually live in LineageEx. If the taxon itself is one of those
    ranks (e.g. a genus-level TaxId), fall back to the record's own name.
    """
    ranks = {rank: "" for rank in LINEAGE_RANKS}
    for entry in record.get("LineageEx", []):
        rank = entry.get("Rank", "")
        if rank in ranks:
            ranks[rank] = entry["ScientificName"]

    own_rank = record.get("Rank", "")
    if own_rank in ranks and not ranks[own_rank]:
        ranks[own_rank] = record.get("ScientificName", "") or ""
    return ranks


def fetch_lineage(
    taxon_ids: list[int],
    email: str,
    api_key: Optional[str] = None,
) -> dict[int, dict[str, str]]:
    """Return {taxon_id: {genus, family, order, phylum}}.

    Unresolvable IDs (or ranks missing from the lineage) are simply absent /
    empty-string; callers should treat missing keys as blank columns.
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    request_delay = 0.11 if api_key else 0.34

    unique_ids = sorted(set(taxon_ids))
    lineage: dict[int, dict[str, str]] = {}

    starts = range(0, len(unique_ids), _BATCH_SIZE)
    for start in tqdm(starts, desc="NCBI lineage", unit="batch", dynamic_ncols=True):
        batch = unique_ids[start : start + _BATCH_SIZE]
        handle = Entrez.efetch(db="taxonomy", id=",".join(str(t) for t in batch))
        records = Entrez.read(handle)
        handle.close()
        time.sleep(request_delay)

        for record in records:
            taxon_id = int(record["TaxId"])
            ranks = _extract_ranks(record)
            lineage[taxon_id] = ranks
            missing_ranks = [r for r in LINEAGE_RANKS if not ranks[r]]
            if missing_ranks:
                log.warning(
                    "Taxon %d missing rank(s) in lineage: %s",
                    taxon_id, ", ".join(missing_ranks),
                )

    missing = set(unique_ids) - set(lineage)
    if missing:
        log.warning(
            "%d taxon ID(s) had no NCBI taxonomy record: %s",
            len(missing), sorted(missing),
        )

    return lineage


def fetch_phyla(
    taxon_ids: list[int],
    email: str,
    api_key: Optional[str] = None,
) -> dict[int, str]:
    """Return {taxon_id: phylum_name}; unresolvable IDs are simply absent.

    Thin wrapper kept for scripts/fetch_strain_phylum.py.
    """
    lineage = fetch_lineage(taxon_ids, email, api_key)
    return {
        taxon_id: ranks["phylum"]
        for taxon_id, ranks in lineage.items()
        if ranks.get("phylum")
    }
