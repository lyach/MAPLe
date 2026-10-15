"""BacDive growth pipeline.

Downloads a subset of the BacDive database and emits a ``(strain x culture medium)`` 
table with a binary ``growth_class`` target for the modeling pipeline.

Note: this package is imported as ``src.bacdive`` to avoid shadowing the
installed ``bacdive`` PyPI client, which it wraps internally.
"""

from src.bacdive.pipeline import run, run_from_cache

__all__ = ["run", "run_from_cache"]
