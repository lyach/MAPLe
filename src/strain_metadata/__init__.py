"""
strain_metadata package
========================
Internal package backing scripts/get_strain_metadata.py. Split into small,
single-purpose modules:

    config.py          paths and tunable constants
    credentials.py      resolving the NCBI email/API key
    align_loader.py      reading align_strains.csv + the manual name-override file
    ncbi_client.py        all Biopython Entrez calls, wrapped in NCBIAssemblyClient
    sequence_stats.py     GC%/N50/hash math on a downloaded FASTA
    pipeline.py            orchestrates the above into the final CSV

Nothing in this package talks directly to argparse or sys.argv -- that's
the entry-point script's job. This makes every piece here easy to import
and reuse from a notebook, a test, or a future script in this project.
"""

# Re-export the names other code is most likely to need, so callers can
# write `from strain_metadata import NCBIAssemblyClient` instead of
# reaching into the submodule directly.
from .align_loader import StrainInput, load_align_strains, load_name_overrides, normalize_strain_name
from .credentials import resolve_ncbi_credentials, clear_saved_credentials
from .ncbi_client import NCBIAssemblyClient, ResolvedAssembly
from .sequence_stats import summarize_fasta, calculate_n50, compute_gc_content_pct, compute_sequence_hash
from .pipeline import run, process_one_strain

__all__ = [
    "StrainInput",
    "load_align_strains",
    "load_name_overrides",
    "normalize_strain_name",
    "resolve_ncbi_credentials",
    "clear_saved_credentials",
    "NCBIAssemblyClient",
    "ResolvedAssembly",
    "summarize_fasta",
    "calculate_n50",
    "compute_gc_content_pct",
    "compute_sequence_hash",
    "run",
    "process_one_strain",
]
