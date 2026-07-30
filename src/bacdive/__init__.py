"""BacDive growth pipeline.

Downloads a taxonomy-scoped subset of the BacDive (DSMZ) database and emits a
tidy ``(strain x culture medium)`` table with a binary ``growth_class`` target
for the modeling pipeline.

Note: this package is imported as ``src.bacdive`` to avoid shadowing the
installed ``bacdive`` PyPI client, which it wraps internally.
"""

from src.bacdive.pipeline import run

__all__ = ["run"]
