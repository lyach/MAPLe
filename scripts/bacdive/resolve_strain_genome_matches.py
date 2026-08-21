#!/usr/bin/env python3
"""Resolve BacDive strains that have no ``genome_accession`` to a genome
assembly, using only *strain-precise* evidence.

Reads data/external/bacdive_growth_api_107k.parquet, takes every strain with an empty
``genome_accession``, and matches the strain-identity columns extract.py emits
against NCBI's assembly_summary files plus the culture-collection -> BioSample
-> assembly map built by scratch/biosample_match.py.

Routes, in the order a strain's ``route`` is assigned. The first three are
exact-identifier matches; the last three are inferential and weaker:

  A_dsm_number         DSM deposit number in an assembly's infraspecific_name
                       / isolate / organism_name
  B_collection_id      any other culture-collection accession, same fields
  F_biosample          collection id -> BioSample -> assembly. Measured at 88%
                       agreement with the recorded accession on strains that
                       already have one, so it is trusted alongside A/B.
  C_strain_taxid       ncbi_taxid recorded at *strain* level
  E_strain_designation BacDive's strain designation inside an assembly's
                       infraspecific_name, constrained to the same species
  D_type_material      strain flagged is_type_strain and its species has an
                       assembly flagged relation_to_type_material

Deliberately excluded:

* CIP numbers as a match key -- NCBI's index drops the number, so a nonsense
  CIP number returns the same hits as a real one.
* subspecies-level taxids. The earlier audit accepted them under
  C_strain_taxid; 458 of its 481 strains were subspecies, and six of those
  resolved to *Salmonella enterica* subsp. *enterica* (taxid 59201) with
  18,039 assemblies. That is a surrogate, not this strain's genome.
* species-level anything. A species taxid identifies a set, not a strain.
* collection ids shared across genera. A first run matched seven strains from
  seven different genera to one assembly, because every one of those BacDive
  records carries ``DSM 104231`` alongside its own unique deposit number; and
  eight *Kitasatospora* isolates (DSM 41133-41140) to another, because all
  eight also list ``ATCC 10762`` and their own numbers match nothing. A
  deposit number claimed by strains of more than one genus is not a strain
  identifier, so it is dropped as a match key. A number shared within one
  genus is kept but reported as ``contested``, since duplicate BacDive entries
  for one deposit are legitimate and indistinguishable from this alone.

GenBank and RefSeq list the same assembly twice (GCA_000008765.1 /
GCF_000008765.1), which made 94% of the audit's "ambiguous" strains an
artifact. Candidates are collapsed on the shared numeric core before anything
is counted as ambiguous. Within a core the RefSeq accession is preferred as
the download handle because RefSeq assemblies are annotated, and the
downloader wants CDS FASTA as well as genomic; both accessions are recorded.

Writes data/external/genome_strain_matches.csv -- one row per matched strain.
**The parquet is not modified.** These accessions are inferred, not recorded by
BacDive, and merging them into ``genome_accession`` would erase that
distinction for every downstream consumer.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("BACDIVE_SCRATCH", "/home/pavan/scratch"))

DEFAULT_INPUT = ROOT / "data" / "external" / "bacdive_growth_api_107k.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "external" / "genome_strain_matches.csv"

SUMMARY_GENBANK = SCRATCH / "assembly_summary_genbank.txt"
SUMMARY_REFSEQ = SCRATCH / "assembly_summary_refseq.txt"
COLLECTION_MAP = SCRATCH / "collection_key_to_assembly.tsv"

log = logging.getLogger("resolve_strain_genome_matches")

# Culture-collection accession, e.g. "DSM 1313", "ATCC 700813". CIP is absent
# on purpose -- see the module docstring.
COLL_RE = re.compile(
    r"\b(DSM|DSMZ|ATCC|LMG|NCTC|JCM|KCTC|NBRC|CCUG|CECT|IAM|VKM|BCRC"
    r"|NRRL|CGMCC|NCIMB|KACC)\b[ ._-]?(\d[\d.]*)",
    re.I,
)

# Route order = confidence order. Exact-identifier routes first.
ROUTE_ORDER = [
    "A_dsm_number",
    "B_collection_id",
    "F_biosample",
    "C_strain_taxid",
    "E_strain_designation",
    "D_type_material",
]

# Better assembly first, used to break ties between distinct candidates.
# Same ordering as src/strain_metadata/ncbi_client.py:_rank_assembly_summaries,
# which also breaks remaining ties on recency -- mirrored below via seq_rel_date.
LEVEL_RANK = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}


def norm_coll(prefix: str, number: str) -> str:
    """'DSM 1313' / 'DSM1313' / 'DSM 1313.' -> 'DSM1313'."""
    prefix = prefix.upper()
    if prefix == "DSMZ":
        prefix = "DSM"
    return f"{prefix}{number.replace('.', '').rstrip('.')}"


def collection_keys(*texts: str) -> set[str]:
    keys: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in COLL_RE.finditer(text):
            keys.add(norm_coll(match.group(1), match.group(2)))
    return keys


def norm_designation(value: str) -> str:
    """Loose strain-designation key: alphanumerics only, uppercased."""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def accession_core(accession: str) -> str:
    """'GCA_000008765.1' / 'GCF_000008765.2' -> '000008765'.

    GenBank and RefSeq share the numeric part for a paired assembly, so this
    collapses the two listings of one genome into a single candidate.
    """
    body = accession.split(".", 1)[0]
    return body[4:] if body[:4] in ("GCA_", "GCF_") else body


# --------------------------------------------------------------------------
# strains
# --------------------------------------------------------------------------

STRAIN_COLS = [
    "bacdive_id", "strain_name_raw", "genus", "genome_accession",
    "dsm_number", "culture_collection_ids", "strain_designation",
    "is_type_strain", "ncbi_taxid", "ncbi_taxid_level",
]


def load_missing_strains(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=STRAIN_COLS)
    strains = df.drop_duplicates("bacdive_id").copy()
    for col in STRAIN_COLS:
        strains[col] = (
            strains[col].fillna("").astype(str).str.strip().replace({"None": ""})
        )
    return strains[strains.genome_accession == ""].reset_index(drop=True)


def collection_key_genera(strains: pd.DataFrame) -> dict[str, set[str]]:
    """collection key -> the set of BacDive genera that list it.

    A key naming more than one genus cannot identify a strain: no single
    culture-collection deposit is seven different genera.
    """
    genera: dict[str, set[str]] = defaultdict(set)
    for row in strains.itertuples():
        keys = collection_keys(
            f"DSM {row.dsm_number}" if row.dsm_number else "",
            row.culture_collection_ids,
        )
        for key in keys:
            genera[key].add(row.genus)
    return genera


def strain_lookup_keys(strains: pd.DataFrame, noisy_keys: set[str]) -> dict:
    """Everything we will look for, so the summary scan can discard the rest.

    Indexing all 3.8M assemblies costs tens of GB; we only ever query a few
    thousand keys, so the scan keeps a row only if it answers one of them.
    """
    wanted = {
        "coll": set(),
        "taxid": set(),        # strain-level only
        "species_type": set(),  # species taxid of a type strain
        "desig": set(),        # (species taxid, normalised designation)
    }
    for row in strains.itertuples():
        wanted["coll"] |= collection_keys(
            f"DSM {row.dsm_number}" if row.dsm_number else "",
            row.culture_collection_ids,
        ) - noisy_keys
        if not row.ncbi_taxid:
            continue
        if row.ncbi_taxid_level == "strain":
            wanted["taxid"].add(row.ncbi_taxid)
        elif row.ncbi_taxid_level == "species":
            # A species taxid is only usable through a strain-specific
            # qualifier: the type-material flag, or a designation match.
            if row.is_type_strain == "1":
                wanted["species_type"].add(row.ncbi_taxid)
            designation = norm_designation(row.strain_designation)
            if len(designation) >= 3:
                wanted["desig"].add((row.ncbi_taxid, designation))
    return wanted


# --------------------------------------------------------------------------
# assembly summaries
# --------------------------------------------------------------------------

class AssemblyIndex:
    """Assemblies relevant to our strains, indexed by each route's key."""

    def __init__(self) -> None:
        self.meta: dict[str, dict] = {}          # accession -> record
        self.by_coll: dict[str, set] = defaultdict(set)
        self.by_taxid: dict[str, set] = defaultdict(set)
        self.by_species_type: dict[str, set] = defaultdict(set)
        self.by_species_desig: dict[tuple, set] = defaultdict(set)
        self.by_core: dict[str, set] = defaultdict(set)

    def scan(self, path: Path, source: str, wanted: dict, *,
             keep_cores: set[str] | None = None,
             keep_accessions: set[str] | None = None) -> tuple[int, int]:
        """One pass over an assembly_summary file.

        ``keep_cores`` additionally retains rows that pair with an assembly
        already kept -- this is how the RefSeq listing of a GenBank hit gets
        picked up even though the collection id lives only on the GenBank row.

        ``keep_accessions`` retains named assemblies regardless of whether
        they answer a lookup key. The BioSample route names its accessions
        directly, so without this they would carry no organism name or
        assembly level and could be neither genus-checked nor ranked.
        """
        seen = kept = 0
        with open(path, encoding="utf-8", errors="replace") as handle:
            header = None
            for line in handle:
                if line.startswith("#"):
                    if line.startswith("#assembly_accession"):
                        header = line.lstrip("#").rstrip("\n").split("\t")
                    continue
                if header is None:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))
                if row.get("group") not in ("bacteria", "archaea"):
                    continue
                if row.get("version_status") != "latest":
                    continue
                seen += 1

                accession = row["assembly_accession"]
                taxid = row.get("taxid", "")
                species_taxid = row.get("species_taxid", "")
                infraspecific = row.get("infraspecific_name", "")
                isolate = row.get("isolate", "")
                organism = row.get("organism_name", "")
                relation = row.get("relation_to_type_material", "")
                relation = "" if relation == "na" else relation
                seq_rel_date = row.get("seq_rel_date", "")

                coll_hits = collection_keys(infraspecific, isolate, organism) & wanted["coll"]
                taxid_hit = taxid in wanted["taxid"]
                type_hit = bool(relation) and species_taxid in wanted["species_type"]
                designation = norm_designation(infraspecific.replace("strain=", ""))
                desig_key = (species_taxid, designation)
                desig_hit = len(designation) >= 3 and desig_key in wanted["desig"]
                core = accession_core(accession)
                pair_hit = keep_cores is not None and core in keep_cores
                named_hit = keep_accessions is not None and accession in keep_accessions

                if not (coll_hits or taxid_hit or type_hit or desig_hit
                        or pair_hit or named_hit):
                    continue

                kept += 1
                self.meta[accession] = {
                    "accession": accession,
                    "source": source,
                    "assembly_level": row.get("assembly_level", ""),
                    "relation_to_type_material": relation,
                    "organism_name": organism,
                    "seq_rel_date": seq_rel_date,
                }
                self.by_core[core].add(accession)
                for key in coll_hits:
                    self.by_coll[key].add(accession)
                if taxid_hit:
                    self.by_taxid[taxid].add(accession)
                if type_hit:
                    self.by_species_type[species_taxid].add(accession)
                if desig_hit:
                    self.by_species_desig[desig_key].add(accession)
        return seen, kept


def load_collection_map(path: Path, wanted_keys: set[str]) -> dict[str, set]:
    """collection key -> {assembly accession}, restricted to keys we query."""
    mapping: dict[str, set] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key, _, accessions = line.rstrip("\n").partition("\t")
            if key in wanted_keys and accessions:
                mapping[key] = set(accessions.split(","))
    return mapping


# --------------------------------------------------------------------------
# per-strain resolution
# --------------------------------------------------------------------------

def choose_within_core(accessions: set[str], index: AssemblyIndex) -> dict:
    """Pick the download handle for one assembly, preferring RefSeq.

    RefSeq assemblies carry annotation, so a GCF accession is what yields a
    CDS FASTA; the GenBank accession is kept alongside it.
    """
    genbank = sorted(a for a in accessions if a.startswith("GCA_"))
    refseq = sorted(a for a in accessions if a.startswith("GCF_"))
    chosen = (refseq or genbank)[0]
    meta = index.meta.get(chosen, {})
    # the type-material flag and level can be recorded on either listing
    level = ""
    relation = ""
    seq_rel_date = ""
    for accession in sorted(accessions):
        record = index.meta.get(accession, {})
        level = level or record.get("assembly_level", "")
        relation = relation or record.get("relation_to_type_material", "")
        seq_rel_date = max(seq_rel_date, record.get("seq_rel_date", ""))
    return {
        "accession": chosen,
        "genbank_accession": genbank[0] if genbank else "",
        "refseq_accession": refseq[0] if refseq else "",
        "assembly_level": level,
        "relation_to_type_material": relation,
        "organism_name": meta.get("organism_name", ""),
        "seq_rel_date": seq_rel_date,
    }


def assembly_genus(organism_name: str) -> str:
    """Leading token of an assembly's organism_name, e.g. 'Bacillus'."""
    return organism_name.split(" ", 1)[0] if organism_name else ""


def resolve_strain(row, index: AssemblyIndex, coll_map: dict,
                   noisy_keys: set[str]) -> dict | None:
    """All strain-precise candidates for one strain, collapsed and ranked."""
    dsm_keys = collection_keys(f"DSM {row.dsm_number}") if row.dsm_number else set()
    other_keys = collection_keys(row.culture_collection_ids) - dsm_keys
    # a deposit number claimed by strains of several genera identifies none
    dsm_keys -= noisy_keys
    other_keys -= noisy_keys

    hits_by_route: dict[str, set] = {}

    def add(route: str, accessions) -> None:
        accessions = set(accessions)
        if accessions:
            hits_by_route[route] = accessions

    add("A_dsm_number", {a for k in dsm_keys for a in index.by_coll.get(k, ())})
    add("B_collection_id", {a for k in other_keys for a in index.by_coll.get(k, ())})
    add("F_biosample",
        {a for k in dsm_keys | other_keys for a in coll_map.get(k, ())})
    if row.ncbi_taxid_level == "strain" and row.ncbi_taxid:
        add("C_strain_taxid", index.by_taxid.get(row.ncbi_taxid, set()))
    if row.ncbi_taxid_level == "species" and row.ncbi_taxid:
        designation = norm_designation(row.strain_designation)
        if len(designation) >= 3:
            add("E_strain_designation",
                index.by_species_desig.get((row.ncbi_taxid, designation), set()))
        if row.is_type_strain == "1":
            add("D_type_material", index.by_species_type.get(row.ncbi_taxid, set()))

    if not hits_by_route:
        return None

    # collapse GenBank/RefSeq listings of one assembly into a single candidate
    cores: dict[str, set] = defaultdict(set)
    routes_per_core: dict[str, set] = defaultdict(set)
    for route, accessions in hits_by_route.items():
        for accession in accessions:
            core = accession_core(accession)
            cores[core].add(accession)
            routes_per_core[core].add(route)
    # F_biosample yields accessions the summary scan may not have kept; fill in
    # their siblings so the RefSeq preference still applies where we know it
    for core, accessions in cores.items():
        accessions |= index.by_core.get(core, set())

    resolved = {core: choose_within_core(accs, index) for core, accs in cores.items()}

    for core, info in resolved.items():
        info["genus_agrees"] = (
            assembly_genus(info["organism_name"]).lower() == row.genus.lower()
            if info["organism_name"] and row.genus else None
        )
    agreeing_cores = {c for c, i in resolved.items() if i["genus_agrees"] is True}
    if agreeing_cores:
        # Only prune when there is something to fall back on. A lone candidate
        # whose genus disagrees is kept and reported, because BacDive and NCBI
        # disagree on genus for legitimately renamed taxa.
        cores = {c: a for c, a in cores.items() if c in agreeing_cores}
        routes_per_core = {c: r for c, r in routes_per_core.items() if c in agreeing_cores}
        resolved = {c: i for c, i in resolved.items() if c in agreeing_cores}

    def rank(core: str) -> tuple:
        info = resolved[core]
        best_route = min(routes_per_core[core], key=ROUTE_ORDER.index)
        return (
            ROUTE_ORDER.index(best_route),
            -len(routes_per_core[core]),                      # agreement first
            LEVEL_RANK.get(info["assembly_level"], 9),
            0 if info["genus_agrees"] is not False else 1,
            0 if info["relation_to_type_material"] else 1,
            0 if info["refseq_accession"] else 1,
            # seq_rel_date is zero-padded YYYY-MM-DD, so it sorts lexically;
            # negating each character reverses that to most-recent-first
            # (matches _rank_assembly_summaries). An unknown date must sort
            # last, not first -- an empty tuple would otherwise win outright.
            (0, tuple(-ord(c) for c in info["seq_rel_date"]))
            if info["seq_rel_date"] else (1, ()),
            info["accession"],
        )

    ordered = sorted(cores, key=rank)
    best_core = ordered[0]
    info = resolved[best_core]
    agreeing = routes_per_core[best_core]
    route = min(agreeing, key=ROUTE_ORDER.index)

    if len(cores) == 1:
        confidence = "cross_confirmed" if len(agreeing) > 1 else "unique"
    else:
        confidence = "ambiguous"

    return {
        "bacdive_id": row.bacdive_id,
        "strain_name_raw": row.strain_name_raw,
        "genus": row.genus,
        "accession": info["accession"],
        "genbank_accession": info["genbank_accession"],
        "refseq_accession": info["refseq_accession"],
        "assembly_level": info["assembly_level"],
        "seq_rel_date": info["seq_rel_date"],
        "relation_to_type_material": info["relation_to_type_material"],
        "assembly_organism_name": info["organism_name"],
        "genus_agrees": {True: "1", False: "0", None: ""}[info["genus_agrees"]],
        "route": route,
        "routes_agreeing": ";".join(sorted(agreeing, key=ROUTE_ORDER.index)),
        "n_routes_agreeing": len(agreeing),
        "routes_with_hits": ";".join(sorted(hits_by_route, key=ROUTE_ORDER.index)),
        "confidence": confidence,
        "n_candidate_assemblies": len(cores),
        "all_candidates": ";".join(resolved[c]["accession"] for c in ordered[:12]),
        "matched_via": ";".join(sorted(dsm_keys | other_keys)),
        "dsm_number": row.dsm_number,
        "culture_collection_ids": row.culture_collection_ids,
        "strain_designation": row.strain_designation,
        "is_type_strain": row.is_type_strain,
        "ncbi_taxid": row.ncbi_taxid,
        "ncbi_taxid_level": row.ncbi_taxid_level,
    }


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="only resolve the first N missing strains (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    for path in (args.input, SUMMARY_GENBANK, SUMMARY_REFSEQ, COLLECTION_MAP):
        if not path.exists():
            log.error("missing required input: %s", path)
            return 1

    strains = load_missing_strains(args.input)
    if args.limit:
        strains = strains.head(args.limit)
    log.info("strains with no genome_accession: %d", len(strains))

    key_genera = collection_key_genera(strains)
    noisy_keys = {k for k, g in key_genera.items() if len(g) > 1}
    log.info("collection keys seen: %d | dropped as cross-genus noise: %d",
             len(key_genera), len(noisy_keys))

    wanted = strain_lookup_keys(strains, noisy_keys)
    log.info("lookup keys -- collection %d | strain taxid %d | type species %d | designation %d",
             len(wanted["coll"]), len(wanted["taxid"]),
             len(wanted["species_type"]), len(wanted["desig"]))

    # loaded before the scan so BioSample-named assemblies keep their metadata
    coll_map = load_collection_map(COLLECTION_MAP, wanted["coll"])
    biosample_accessions = {a for accs in coll_map.values() for a in accs}
    log.info("biosample collection keys matching our strains: %d (%d assemblies)",
             len(coll_map), len(biosample_accessions))

    index = AssemblyIndex()
    seen, kept = index.scan(SUMMARY_GENBANK, "genbank", wanted,
                            keep_accessions=biosample_accessions)
    log.info("genbank: %d latest bacterial/archaeal assemblies, %d relevant", seen, kept)
    seen, kept = index.scan(SUMMARY_REFSEQ, "refseq", wanted,
                            keep_cores=set(index.by_core),
                            keep_accessions=biosample_accessions)
    log.info("refseq : %d latest bacterial/archaeal assemblies, %d relevant", seen, kept)

    rows = [r for r in (resolve_strain(s, index, coll_map, noisy_keys)
                        for s in strains.itertuples()) if r]
    if not rows:
        log.warning("no strain-precise match for any strain -- nothing written")
        return 1

    # One assembly claimed by several strains means at most one of them is
    # right (duplicate BacDive entries for a single deposit are the benign
    # case). Flag rather than drop -- the two are indistinguishable here.
    claims: dict[str, int] = defaultdict(int)
    for row in rows:
        claims[row["accession"]] += 1
    for row in rows:
        n = claims[row["accession"]]
        row["contested_by_n_strains"] = n
        if n > 1:
            row["confidence"] = "contested"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("\nwrote %s (%d strains)", args.output, len(rows))

    frame = pd.DataFrame(rows)
    log.info("\n=== strains matched, by route ===")
    for route in ROUTE_ORDER:
        n = int((frame.route == route).sum())
        if n:
            log.info("  %-22s %6d", route, n)
    log.info("\n=== confidence ===")
    for name in ("cross_confirmed", "unique", "ambiguous", "contested"):
        log.info("  %-22s %6d", name, int((frame.confidence == name).sum()))
    log.info("\n=== assembly level ===")
    for level, n in frame.assembly_level.replace("", "(unknown)").value_counts().items():
        log.info("  %-22s %6d", level, n)
    log.info("\ndistinct assemblies to download : %d", frame.accession.nunique())
    log.info("flagged as type material        : %d",
             int((frame.relation_to_type_material != "").sum()))
    log.info("genus agrees with NCBI          : %d agree / %d disagree / %d unknown",
             int((frame.genus_agrees == "1").sum()),
             int((frame.genus_agrees == "0").sum()),
             int((frame.genus_agrees == "").sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
