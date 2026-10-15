"""Thin wrapper around the ``bacdive`` PyPI client (BacDive API v2).

Searches by taxonomy, caches one JSON file per BacDive ID so re-runs skip the network,
and fetches uncached records in parallel batches.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import time
from pathlib import Path
from typing import Iterator

import bacdive  # the installed PyPI client (NOT this package)
from loguru import logger
from tqdm import tqdm

from src.bacdive import config


def _bacdive_id(record: dict) -> int | None:
    """Return the BacDive-ID from a record's ``General`` block, or ``None`` if absent or invalid."""
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
        show_progress: bool = True,
        max_workers: int = config.MAX_WORKERS,
        batch_size: int = config.FETCH_BATCH_SIZE,
    ) -> None:
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.request_delay = request_delay
        self.force = force
        self.show_progress = show_progress
        self.max_workers = max(1, max_workers)
        self.batch_size = max(1, batch_size)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # A single client instance is reused across taxa.
        self._client = bacdive.BacdiveClient()

    # -- caching helpers --

    def _cache_path(self, bacdive_id: int) -> Path:
        return self.cache_dir / f"{bacdive_id}.json"

    def _read_cache(self, bacdive_id: int) -> dict | None:
        try:
            return json.loads(self._cache_path(bacdive_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read cache for BacDive-ID {}: {}", bacdive_id, exc)
            return None

    def _write_cache(self, bacdive_id: int, record: dict) -> None:
        try:
            self._cache_path(bacdive_id).write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("could not cache BacDive-ID {}: {}", bacdive_id, exc)

    # -- fetch helpers --

    def _collect_ids(self) -> list[int]:
        """Return all ids for the current search by walking every ID-listing page.

        Pages are small (~100 ids), so paging sequentially is cheap compared with fetching
        the full records, which happens afterwards in parallel batches.
        """
        ids: list[int] = []
        result = self._client.result
        while True:
            ids.extend(int(i) for i in result.get("results", []))
            next_url = result.get("next")
            if not next_url:
                break
            result = self._client.do_api_call(next_url)
        return ids

    def _fetch_batch(self, ids: list[int]) -> list[dict]:
        """Fetch full records for a batch of ids; safe to call concurrently."""
        id_str = ";".join(str(i) for i in ids)
        response = self._client.do_api_call("fetch/" + id_str)
        if not isinstance(response, dict) or "results" not in response:
            raise RuntimeError(
                f"unexpected fetch response for {len(ids)} id(s): {response!r:.200}"
            )
        results = response["results"]
        if isinstance(results, dict):
            return [r for r in results.values() if isinstance(r, dict)]
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
        return []

    # -- public API --

    def fetch_taxon(self, taxon: str) -> Iterator[dict]:
        """Yield every BacDive record matching a taxonomy query.

        Cached records (``cache_dir/<bacdive_id>.json``) are read from disk and not
        re-fetched unless ``force=True``. Raises ``LookupError`` if the search returns
        zero hits so the caller can record the taxon as a failure.
        """
        count = self._client.search(taxonomy=taxon)
        if not count:
            raise LookupError(f"no BacDive records for taxon {taxon!r}")

        logger.info("taxon {!r} -> {} record(s)", taxon, count)

        all_ids = self._collect_ids()

        cached_ids: list[int] = []
        fresh_ids: list[int] = []
        for bid in all_ids:
            if not self.force and self._cache_path(bid).exists():
                cached_ids.append(bid)
            else:
                fresh_ids.append(bid)

        if cached_ids:
            logger.info(
                "taxon {!r}: {}/{} record(s) already cached, skipping re-fetch",
                taxon, len(cached_ids), len(all_ids),
            )

        pbar = tqdm(
            total=len(all_ids),
            desc=f"BacDive: {taxon}",
            unit="rec",
            leave=False,
            position=1,
            disable=not self.show_progress,
        )

        n = 0
        try:
            for bid in cached_ids:
                record = self._read_cache(bid)
                if record is not None:
                    n += 1
                    pbar.update(1)
                    yield record

            batches = [
                fresh_ids[i:i + self.batch_size]
                for i in range(0, len(fresh_ids), self.batch_size)
            ]
            if batches:
                with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures: dict[cf.Future, list[int]] = {}
                    for batch in batches:
                        futures[pool.submit(self._fetch_batch, batch)] = batch
                        # Stagger submission so we don't open every connection at once
                        if self.request_delay:
                            time.sleep(self.request_delay)

                    for future in cf.as_completed(futures):
                        batch = futures[future]
                        try:
                            records = future.result()
                        except Exception as exc:
                            logger.error(
                                "taxon {!r}: batch fetch failed ({} id(s)): {}",
                                taxon, len(batch), exc,
                            )
                            continue
                        for record in records:
                            bid = _bacdive_id(record)
                            if bid is not None:
                                self._write_cache(bid, record)
                            n += 1
                            pbar.update(1)
                            yield record
        finally:
            pbar.close()
        logger.debug("taxon {!r} fully retrieved ({} records)", taxon, n)
