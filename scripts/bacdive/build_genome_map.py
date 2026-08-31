#!/usr/bin/env python3
"""Build the bacdive_id -> genome mapping, and report what is on disk.

Two independent sources of accession, kept distinguishable:

* ``bacdive`` -- ``genome_accession`` as recorded by BacDive and extracted into
  the parquet. Restricted to ``genome_accession_source == "insdc"``; IMG and
  BV-BRC ids are real but not fetchable from NCBI.
* ``recovered`` -- inferred by scripts/bacdive/resolve_strain_genome_matches.py
  from strain-precise evidence, carrying its ``route`` and ``confidence``.

A strain appears once. A recorded accession always wins over a recovered one:
the recovered routes only ever ran on strains that had no accession, so the two
should not collide, and if they ever do the source of record is authoritative.

``confidence == "contested"`` means several strains resolved to one assembly,
so at most one of them is correctly assigned. Those rows are emitted, not
dropped -- filter them out for anything that assumes a genome is strain-specific.

Writes data/external/bacdive_genome_map.csv with, per strain, the accession,
its provenance, and the absolute FASTA paths where they exist on disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = ROOT / "data" / "external" / "bacdive_growth_full.parquet"
DEFAULT_MATCHES = ROOT / "data" / "external" / "genome_strain_matches.csv"
DEFAULT_GENOMES = ROOT / "data" / "external" / "bacdive_genomes"
DEFAULT_OUTPUT = ROOT / "data" / "external" / "bacdive_genome_map.csv"

log = logging.getLogger("build_genome_map")

COLUMNS = [
    "bacdive_id", "strain_name_raw", "genus", "accession", "accession_source",
    "route", "confidence", "assembly_level", "genus_agrees",
    "genome_fasta", "cds_fasta", "genome_on_disk", "cds_on_disk",
]


def _blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"None": ""})


def load_recorded(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=[
        "bacdive_id", "strain_name_raw", "genus",
        "genome_accession", "genome_accession_source",
    ])
    strains = df.drop_duplicates("bacdive_id").copy()
    for col in strains.columns:
        strains[col] = _blank(strains[col])
    recorded = strains[
        (strains.genome_accession != "")
        & (strains.genome_accession_source == "insdc")
    ].copy()
    recorded = recorded.rename(columns={"genome_accession": "accession"})
    recorded["accession_source"] = "bacdive"
    recorded["route"] = ""
    recorded["confidence"] = "recorded"
    recorded["assembly_level"] = ""
    recorded["genus_agrees"] = ""
    return recorded[[c for c in COLUMNS if c in recorded.columns]]


def load_recovered(matches_path: Path) -> pd.DataFrame:
    if not matches_path.exists():
        log.warning("no recovered matches at %s -- mapping recorded strains only",
                    matches_path)
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(matches_path, dtype=str).fillna("")
    df["accession_source"] = "recovered"
    return df[[c for c in COLUMNS if c in df.columns]]


def attach_paths(frame: pd.DataFrame, genomes_dir: Path) -> pd.DataFrame:
    """Resolve each accession to its FASTA paths, checking the disk once."""
    genome_paths: dict[str, tuple[str, str, bool, bool]] = {}
    for accession in frame.accession.unique():
        base = genomes_dir / accession
        genome = base / f"{accession}_genomic.fna"
        cds = base / f"{accession}_cds_from_genomic.fna"
        genome_paths[accession] = (
            str(genome), str(cds), genome.is_file(), cds.is_file(),
        )
    frame["genome_fasta"] = frame.accession.map(lambda a: genome_paths[a][0])
    frame["cds_fasta"] = frame.accession.map(lambda a: genome_paths[a][1])
    frame["genome_on_disk"] = frame.accession.map(lambda a: int(genome_paths[a][2]))
    frame["cds_on_disk"] = frame.accession.map(lambda a: int(genome_paths[a][3]))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--genomes-dir", type=Path, default=DEFAULT_GENOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    recorded = load_recorded(args.parquet)
    recovered = load_recovered(args.matches)
    log.info("recorded by BacDive (insdc): %d strains", len(recorded))
    log.info("recovered by resolver      : %d strains", len(recovered))

    overlap = set(recorded.bacdive_id) & set(recovered.bacdive_id)
    if overlap:
        log.warning("%d strain(s) in both sources -- keeping the BacDive record",
                    len(overlap))

    frame = pd.concat([recorded, recovered], ignore_index=True)
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    frame = frame.fillna("")
    frame = frame.drop_duplicates("bacdive_id", keep="first")
    frame = attach_paths(frame, args.genomes_dir)[COLUMNS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    log.info("\nwrote %s (%d strains, %d assemblies)",
             args.output, len(frame), frame.accession.nunique())

    log.info("\n=== provenance ===")
    for source, n in frame.accession_source.value_counts().items():
        log.info("  %-12s %6d", source, n)
    log.info("\n=== files present on disk ===")
    log.info("  genome FASTA  %6d of %d", int(frame.genome_on_disk.sum()), len(frame))
    log.info("  CDS FASTA     %6d of %d", int(frame.cds_on_disk.sum()), len(frame))
    log.info("\n=== usable (genome on disk), by provenance ===")
    usable = frame[frame.genome_on_disk == 1]
    for source, n in usable.accession_source.value_counts().items():
        log.info("  %-12s %6d", source, n)
    log.info("\n=== recovered rows by confidence ===")
    rec = frame[frame.accession_source == "recovered"]
    for name, n in rec.confidence.value_counts().items():
        log.info("  %-16s %6d", name, n)
    log.info("\nstrains still with no accession from either source: %d",
             _missing_count(args.parquet, frame))
    return 0


def _missing_count(parquet_path: Path, frame: pd.DataFrame) -> int:
    ids = pd.read_parquet(parquet_path, columns=["bacdive_id"])
    total = ids.bacdive_id.astype(str).str.strip().nunique()
    return total - len(frame)


if __name__ == "__main__":
    raise SystemExit(main())
