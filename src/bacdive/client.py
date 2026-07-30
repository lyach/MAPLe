"""
client.py
=========
Thin wrapper around the installed ``bacdive`` PyPI client (BacDive API v2,
public / no credentials required). Handles:

  * searching by taxonomy and iterating every matching record,
  * on-disk JSON caching (one file per BacDive-ID) so re-runs don't re-hit
    the API,
  * polite pacing between paged calls.

Only the small surface the pipeline needs is exposed; everything else is left
to the vendor client.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import bacdive  # the installed PyPI client (NOT this package)

from src.bacdive import config

log = logging.getLogger("bacdive.client")


def _bacdive_id(record: dict) -> Optional[int]:
    """Pull the BacDive-ID out of a record's ``General`` block."""
    general = record.get("General", {})
    if isinstance(general, dict):
        bid = general.get("BacDive-ID")
        if bid is not None:
            try:
                return int(bid)
            except (TypeError, ValueError):
                return None
    return None


class BacDiveFetcher:
    """Search BacDive by taxonomy and yield full records, with JSON caching."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        request_delay: float = config.REQUEST_DELAY,
        force: bool = False,
    ) -> None:
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.request_delay = request_delay
        self.force = force
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # A single client instance is reused across taxa.
        self._client = bacdive.BacdiveClient()

    # -- caching helpers ----------------------------------------------------

    def _cache_path(self, bacdive_id: int) -> Path:
        return self.cache_dir / f"{bacdive_id}.json"

    def _write_cache(self, bacdive_id: int, record: dict) -> None:
        try:
            self._cache_path(bacdive_id).write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("could not cache BacDive-ID %s: %s", bacdive_id, exc)

    # -- public API ---------------------------------------------------------

    def fetch_taxon(self, taxon: str) -> Iterator[dict]:
        """Yield every BacDive record matching a taxonomy query.

        Records are cached to ``cache_dir/<bacdive_id>.json`` as they arrive.
        Raises ``LookupError`` if the search returns zero hits so the caller
        can record the taxon as a failure.
        """
        count = self._client.search(taxonomy=taxon)
        if not count:
            raise LookupError(f"no BacDive records for taxon {taxon!r}")

        log.info("taxon %r -> %d record(s)", taxon, count)

        n = 0
        for record in self._client.retrieve():
            if not isinstance(record, dict):
                continue
            bid = _bacdive_id(record)
            if bid is not None:
                self._write_cache(bid, record)
            n += 1
            yield record
            # gentle pacing; retrieve() pages internally as it goes.
            if self.request_delay:
                time.sleep(self.request_delay)

        log.debug("taxon %r fully retrieved (%d records)", taxon, n)
