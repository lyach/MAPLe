"""
sequence_stats.py
==================
Pure, NCBI-independent functions that compute statistics from a FASTA file
already sitting on disk. None of these functions make a network call --
that separation is what makes them trivial to unit-test and reuse (e.g. you
could point summarize_fasta() at any FASTA file from anywhere, not just
ones this pipeline downloaded).
"""

from __future__ import annotations

# hashlib: standard-library module providing SHA-256 and other hash functions.
import hashlib
# Path: filesystem paths.
from pathlib import Path

# Bio.SeqIO reads FASTA (and many other sequence formats) into Python
# objects with .id and .seq attributes.
from Bio import SeqIO


def calculate_n50(lengths: list[int]) -> int:
    """Standard N50 statistic: sort contig lengths largest-first, walk down
    the list adding up lengths, and return the length of the contig at the
    point where the running total first reaches half the genome size.
    Intuitively: "50% of the genome is contained in contigs at least this
    long." Returns 0 for an empty input rather than raising."""
    if not lengths:
        return 0
    # reverse=True sorts largest-first.
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    target = total / 2
    cumulative = 0
    for length in sorted_lengths:
        cumulative += length
        if cumulative >= target:
            return length
    # Unreachable in practice (the loop above always returns once
    # cumulative reaches the full total), but keeps type-checkers happy
    # and guards against floating-point edge cases.
    return sorted_lengths[-1]


def compute_gc_content_pct(sequences: list[str]) -> float:
    """GC content as a percentage (0-100), computed across all sequences
    concatenated together (so a genome with a short, GC-poor plasmid isn't
    skewed by treating each replicon as equally weighted)."""
    # "".join(...) concatenates every sequence into one long string;
    # .upper() makes the G/C count case-insensitive (some FASTA exports
    # use lowercase for soft-masked regions).
    full_sequence = "".join(sequences).upper()
    if not full_sequence:
        return 0.0
    gc_count = full_sequence.count("G") + full_sequence.count("C")
    return round((gc_count / len(full_sequence)) * 100, 2)


def compute_sequence_hash(records: list[tuple[str, str]]) -> str:
    """SHA-256 hash of the genome's sequence content, used to detect
    duplicates or changes on a future re-pull. `records` is a list of
    (accession_id, sequence) pairs.

    We sort by accession_id before concatenating so the hash reflects
    *content*, not whatever order NCBI happened to return the records in
    (which isn't strictly guaranteed to be identical across separate
    requests) -- otherwise two pulls of the literal same genome could
    produce different hashes purely due to ordering."""
    ordered = sorted(records, key=lambda pair: pair[0])
    joined_sequence = "".join(seq.upper() for _, seq in ordered)
    # .encode("utf-8") turns the string into bytes (hashlib needs bytes,
    # not str); .hexdigest() renders the hash as a readable hex string.
    return hashlib.sha256(joined_sequence.encode("utf-8")).hexdigest()


def summarize_fasta(fasta_path: Path) -> dict:
    """Read a downloaded FASTA file and compute every sequence-derived
    statistic in one pass: genome size, GC%, contig count, N50, and the
    content hash."""
    # SeqIO.parse streams through the file record-by-record; for each
    # record we only keep its id and its sequence as a plain string (we
    # don't need Biopython's richer Seq object beyond this point).
    records = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(str(fasta_path), "fasta")]
    lengths = [len(seq) for _, seq in records]

    return {
        "genome_size_bp": sum(lengths),
        "gc_content_pct": compute_gc_content_pct([seq for _, seq in records]),
        "num_contigs": len(records),
        "n50": calculate_n50(lengths),
        "sequence_hash": compute_sequence_hash(records),
    }
