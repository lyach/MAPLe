"""
ncbi_client.py
==============
Every interaction with NCBI's Entrez API lives in this one module, wrapped
in a single NCBIAssemblyClient class. Keeping it in one place (instead of
scattering esearch/esummary/elink/efetch calls throughout the pipeline)
means:
  - it's the only place that needs to change if NCBI's API ever shifts,
  - it's reusable from other scripts in this project later on, and
  - it's easy to swap out for a fake/mock client in tests, since the rest
    of the pipeline only ever talks to this class, never to Bio.Entrez
    directly.
"""

from __future__ import annotations

# html: NCBI's assembly-summary "Meta" field comes back HTML-escaped
# (e.g. "&lt;Stat...&gt;"); we need to unescape it before reading it.
import html
# logging: structured console output.
import logging
# re: regex used both for cleaning ATCC numbers and for parsing the
# "Meta" stats blob.
import re
# time: short pauses between requests, to stay under NCBI's rate limits.
import time
# dataclass/field: lightweight containers for query results.
from dataclasses import dataclass, field
# datetime: parsing NCBI's submission/update date strings for ranking.
from datetime import datetime
# Path: filesystem paths (used for the FASTA/GBK download targets).
from pathlib import Path
# Optional: type hint meaning "this value or None".
from typing import Optional

# Bio.Entrez is Biopython's wrapper around NCBI's E-utilities web API.
# Bio.SeqIO isn't needed here -- sequence parsing lives in sequence_stats.py.
from Bio import Entrez

# Re-use the shared constants/config instead of redefining them here.
from .config import ASSEMBLY_LEVEL_MAP
from .align_loader import StrainInput, clean_strain_name_for_search

log = logging.getLogger("ncbi_client")


@dataclass
class ResolvedAssembly:
    """Everything we learned about a strain's best-matching NCBI assembly,
    before we've downloaded any actual sequence data."""
    assembly_uid: str                  # internal NCBI UID for this assembly record
    assembly_accession: str            # e.g. "GCF_000193495.2"
    refseq_accession: str              # RefSeq-specific accession, "" if none
    taxon_id: Optional[int]            # NCBI Taxonomy ID
    assembly_level: str                # one of: complete/chromosome/scaffold/contig
    # The base FTP directory for this assembly, taken directly from the
    # esummary response (FtpPath_RefSeq or FtpPath_GenBank). Looks like:
    #   https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/193/495/GCF_000193495.2_ASM19349v2
    # We use this to build the exact FASTA/GBK download URLs without any
    # additional FTP directory-listing step.
    ftp_path: str = ""
    # Stats NCBI itself reports for the assembly (total_length, contig_count,
    # contig_n50/scaffold_n50) -- used as a fallback when we can't or don't
    # download every sequence ourselves. Empty dict if NCBI didn't report any.
    meta_stats: dict = field(default_factory=dict)


def _clean_atcc_for_query(atcc_number: str) -> str:
    """Your align file's ATCC numbers are sometimes plain digits ("15973")
    and sometimes a prefixed catalog code ("BAA-1025"). Both forms are
    meaningful to NCBI's search (it indexes strain text like "ATCC
    BAA-1025" verbatim) so we only trim whitespace here -- we deliberately
    do NOT strip non-digit characters, since that would turn "BAA-1025"
    into the much less specific (and likely wrong) "1025"."""
    return atcc_number.strip()


def _parse_meta_stats(meta_str: str) -> dict:
    """NCBI's assembly esummary 'Meta' field is a loosely-formed XML blob
    (not a single well-formed document), so we pull out just the <Stat>
    tags we care about with a regex instead of a strict XML parser."""
    if not meta_str:
        return {}
    # html.unescape turns "&lt;Stat...&gt;" back into "<Stat...>".
    unescaped = html.unescape(meta_str)
    stats: dict = {}
    # Each match is (category, sequence_tag, value), e.g.
    # ("contig_count", "all", "5"). We only want the "all sequences"
    # totals, not per-replicon breakdowns.
    for category, tag_scope, value in re.findall(
        r'<Stat category="([^"]+)" sequence_tag="([^"]+)">([^<]+)</Stat>', unescaped
    ):
        if tag_scope != "all":
            continue
        try:
            stats[category] = int(value)
        except ValueError:
            # Non-numeric stat (rare) -- keep it as a string rather than
            # dropping it silently.
            stats[category] = value
    return stats


def _date_ordinal(date_str: str) -> int:
    """Convert an NCBI date string ("2022/01/15" or "2022-01-15") into a
    plain integer that's bigger for later dates, so we can sort by it.
    Unparseable or missing dates sort as "oldest" (ordinal 0) rather than
    raising an error."""
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).toordinal()
        except (ValueError, TypeError):
            continue
    return 0


def _expected_genus_species(strain_name: str) -> tuple[str, str]:
    """Extract (genus, species) in lowercase from a strain name, for
    cross-checking against what NCBI reports. Handles the project's
    'Genus species; substrain' convention and parenthetical authorities.
    Returns ("", "") if the name can't be parsed into two tokens."""
    import re
    # Drop everything from the first semicolon (substrain tag) onward.
    base = strain_name.split(";")[0]
    # Remove parenthetical authority like "(Schroeter)".
    base = re.sub(r"\([^)]*\)", " ", base)
    tokens = base.split()
    if len(tokens) >= 2:
        return tokens[0].lower(), tokens[1].lower()
    if len(tokens) == 1:
        return tokens[0].lower(), ""
    return "", ""


def _organism_matches(expected_strain: str, ncbi_organism: str) -> bool:
    """Check whether the organism NCBI reports for an assembly is consistent
    with the strain we were searching for. We compare at the genus+species
    level (ignoring substrain designations, which legitimately differ).

    This catches the case where a BioSample's ATCC number was mistyped or
    mislabeled by the submitter and points to a completely different organism
    -- exactly the kind of silent error that would corrupt training data."""
    exp_genus, exp_species = _expected_genus_species(expected_strain)
    ncbi_genus, ncbi_species = _expected_genus_species(ncbi_organism or "")

    if not exp_genus or not ncbi_genus:
        return True  # can't compare -- don't flag

    # Genus must match. Species must match IF both have one (some NCBI
    # records are genus-only, e.g. "Bacillus sp."), and we treat "sp." as
    # a wildcard that matches anything.
    if exp_genus != ncbi_genus:
        return False
    if exp_species and ncbi_species and ncbi_species != "sp." and exp_species != ncbi_species:
        return False
    return True


def _rank_assembly_summaries(summaries: list[dict]) -> list[dict]:
    """Sort candidate assemblies best-first: complete genomes before
    drafts, and -- within the same completeness level -- the most recently
    updated assembly first."""
    level_priority = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}

    def sort_key(s):
        level_rank = level_priority.get(s.get("AssemblyStatus", ""), 9)
        last_update = s.get("LastUpdateDate", "") or s.get("SubmissionDate", "")
        # Negate the date ordinal so that, for equal level_rank, sorting
        # ascending (Python's default) puts the *largest* ordinal --
        # i.e. the most recent date -- first.
        return (level_rank, -_date_ordinal(last_update))

    return sorted(summaries, key=sort_key)


class NCBIAssemblyClient:
    """A small, stateful wrapper around Biopython's Entrez module. One
    instance is created per script run (after credentials are resolved)
    and reused for every strain, so the email/API key/rate-limit settings
    only need to be configured once."""

    def __init__(self, email: str, api_key: Optional[str] = None):
        # Bio.Entrez.email/api_key are module-level globals inside
        # Biopython itself (not per-instance), so setting them here applies
        # to every Entrez call made afterwards, for the lifetime of the
        # process.
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        # Pause a little longer between requests if we don't have an API
        # key, since the rate limit without one is lower.
        self.request_delay = 0.11 if api_key else 0.34

    # ---- low-level helper -------------------------------------------------

    def _retry(self, fn, attempts: int = 3, **kwargs):
        """Call one Entrez function (esearch/esummary/elink/epost), parse
        its XML response, and retry with exponential backoff on transient
        failures (network blips, NCBI's occasional 429s). Always pauses
        afterwards to respect the rate limit, even on success."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                handle = fn(**kwargs)
                result = Entrez.read(handle)
                handle.close()
                time.sleep(self.request_delay)
                return result
            except Exception as exc:
                last_exc = exc
                wait_seconds = 2 ** attempt  # 2s, then 4s, then 8s
                log.debug(
                    "Entrez call failed (attempt %d/%d): %s -- retrying in %ds",
                    attempt, attempts, exc, wait_seconds,
                )
                time.sleep(wait_seconds)
        # All attempts exhausted -- re-raise the most recent error so the
        # caller can decide how to handle it (we don't want to silently
        # swallow a real failure).
        raise last_exc

    # ── helpers ────────────────────────────────────────────────────────────

    def _assembly_ids_from_term(self, term: str) -> list[str]:
        """Run esearch on the assembly database for one term. Returns a
        (possibly empty) list of NCBI UIDs on success, or raises on a
        network/API error (caller decides how to handle)."""
        result = self._retry(Entrez.esearch, db="assembly", term=term, retmax=20)
        return result.get("IdList", [])

    def _assembly_summaries_for_ids(self, uid_list: list[str]) -> list[dict]:
        """Fetch DocumentSummary dicts for a list of assembly UIDs."""
        if not uid_list:
            return []
        result = self._retry(Entrez.esummary, db="assembly", id=",".join(uid_list))
        return result.get("DocumentSummarySet", {}).get("DocumentSummary", [])

    # ── Strategy 1: direct ATCC search in the assembly database ────────────

    def _try_ncbi_atcc_assembly(
        self, strain_input: "StrainInput", report: "StrainResolutionReport"
    ) -> Optional[dict]:
        """Search NCBI's assembly database directly for the ATCC number.
        This is the fastest and most strain-specific approach when it works,
        but NCBI doesn't always index ATCC numbers in the assembly metadata."""
        from .resolution_report import Strategy

        atcc = _clean_atcc_for_query(strain_input.atcc_number or "")
        if not atcc:
            return None

        query = f'"ATCC {atcc}"'
        try:
            uid_list  = self._assembly_ids_from_term(query)
            summaries = self._assembly_summaries_for_ids(uid_list)
        except Exception as exc:
            report.add_attempt(Strategy.NCBI_ATCC_ASSEMBLY, query,
                               note=f"API error: {exc}")
            return None

        if summaries:
            chosen = _rank_assembly_summaries(summaries)[0]
            acc = chosen.get("AssemblyAccession", "?")
            report.add_attempt(Strategy.NCBI_ATCC_ASSEMBLY, query,
                               result=acc)
            log.info("Strategy 1 hit: %r via %r → %s (%s)",
                     strain_input.strain, query, acc, chosen.get("AssemblyStatus"))
            return chosen

        report.add_attempt(Strategy.NCBI_ATCC_ASSEMBLY, query,
                           note="0 results in NCBI assembly database")
        log.debug("Strategy 1 miss: %r", query)
        return None

    # ── Strategy 2: BioSample cross-reference ──────────────────────────────

    def _try_ncbi_biosample_link(
        self, strain_input: "StrainInput", report: "StrainResolutionReport"
    ) -> Optional[dict]:
        """Query NCBI's BioSample database for the ATCC number, then follow
        the BioSample → assembly elink to get the assembly record.

        BioSample stores a 'culture_collection' attribute like 'ATCC:15252'
        for every deposited genome -- even when that ATCC number is missing
        from the assembly database's own search index. This makes BioSample
        the most reliable ATCC cross-reference within NCBI's ecosystem."""
        from .resolution_report import Strategy

        atcc = _clean_atcc_for_query(strain_input.atcc_number or "")
        if not atcc:
            return None

        # Search BioSample for the ATCC number in the culture_collection field.
        # We try two forms: the colon-prefixed form NCBI uses internally
        # ("ATCC:15252") and the plain form ("ATCC 15252"), because different
        # submitters formatted their submissions differently.
        for bs_query in [f'"ATCC:{atcc}"', f'"ATCC {atcc}"']:
            try:
                bs_result = self._retry(
                    Entrez.esearch, db="biosample", term=bs_query, retmax=10
                )
            except Exception as exc:
                report.add_attempt(
                    Strategy.NCBI_BIOSAMPLE_LINK, bs_query,
                    note=f"BioSample esearch error: {exc}",
                )
                continue

            bs_ids = bs_result.get("IdList", [])
            if not bs_ids:
                report.add_attempt(
                    Strategy.NCBI_BIOSAMPLE_LINK, bs_query,
                    note="0 BioSample results",
                )
                continue

            # Found BioSample records -- now link them to assemblies.
            assembly_uid_set: set[str] = set()
            for bs_id in bs_ids:
                try:
                    link_result = self._retry(
                        Entrez.elink,
                        dbfrom="biosample", db="assembly",
                        id=bs_id, linkname="biosample_assembly",
                    )
                    for linkset in link_result[0].get("LinkSetDb", []):
                        for link in linkset.get("Link", []):
                            assembly_uid_set.add(link["Id"])
                except Exception:
                    pass  # link failed for this one BioSample record, try next

            if not assembly_uid_set:
                report.add_attempt(
                    Strategy.NCBI_BIOSAMPLE_LINK, bs_query,
                    note=f"{len(bs_ids)} BioSample record(s) found but "
                         "elink returned no assembly links",
                )
                continue

            # Fetch assembly summaries for all linked UIDs and pick the best.
            try:
                summaries = self._assembly_summaries_for_ids(list(assembly_uid_set))
            except Exception as exc:
                report.add_attempt(
                    Strategy.NCBI_BIOSAMPLE_LINK, bs_query,
                    note=f"esummary error for linked assemblies: {exc}",
                )
                continue

            if summaries:
                chosen = _rank_assembly_summaries(summaries)[0]
                acc = chosen.get("AssemblyAccession", "?")
                ncbi_organism = chosen.get("Organism", "") or chosen.get("SpeciesName", "")

                # Cross-check: does the organism NCBI reports actually match
                # the strain we searched for? If not, the ATCC number in this
                # BioSample was likely mislabeled -- flag it loudly but still
                # return it (with the flag) so the user can decide.
                organism_ok = _organism_matches(strain_input.strain, ncbi_organism)
                if not organism_ok:
                    report.organism_mismatch = True
                    log.warning(
                        "ORGANISM MISMATCH for %r: searched via BioSample %r, "
                        "but NCBI assembly %s is organism %r. The ATCC number "
                        "may be mislabeled in this BioSample record.",
                        strain_input.strain, bs_query, acc, ncbi_organism,
                    )
                    note = (f"{len(bs_ids)} BioSample -> {len(assembly_uid_set)} "
                            f"assembly link(s); ORGANISM MISMATCH (got {ncbi_organism!r})")
                else:
                    note = f"{len(bs_ids)} BioSample -> {len(assembly_uid_set)} assembly link(s)"

                report.add_attempt(
                    Strategy.NCBI_BIOSAMPLE_LINK, bs_query,
                    result=acc, note=note,
                )
                log.info(
                    "Strategy 2 hit: %r via BioSample %r -> %s (%s)%s",
                    strain_input.strain, bs_query, acc, chosen.get("AssemblyStatus"),
                    " [ORGANISM MISMATCH]" if not organism_ok else "",
                )
                return chosen

        return None

    # ── Strategy 3: organism-field search (broad) ──────────────────────────

    def _try_ncbi_organism(
        self, strain_input: "StrainInput", report: "StrainResolutionReport"
    ) -> Optional[dict]:
        """Search the assembly database by organism name (constrained to
        the [Organism] field, then as plain free text). Returns the best
        complete genome for the species -- may not be the exact strain,
        so results from this strategy are flagged as 'organism-level only'
        in the resolution report."""
        from .resolution_report import Strategy

        search_name = clean_strain_name_for_search(strain_input.strain)
        if not search_name:
            return None

        for strat, query in [
            (Strategy.NCBI_ORGANISM,  f'"{search_name}"[Organism]'),
            (Strategy.NCBI_FREETEXT,  search_name),
        ]:
            try:
                uid_list  = self._assembly_ids_from_term(query)
                summaries = self._assembly_summaries_for_ids(uid_list)
            except Exception as exc:
                report.add_attempt(strat, query, note=f"API error: {exc}")
                continue

            if summaries:
                chosen = _rank_assembly_summaries(summaries)[0]
                acc = chosen.get("AssemblyAccession", "?")
                report.add_attempt(strat, query, result=acc,
                                   note="organism-level match — not strain-confirmed")
                log.info(
                    "Strategy 3 hit: %r via %r → %s (%s) [organism-level]",
                    strain_input.strain, query, acc, chosen.get("AssemblyStatus"),
                )
                return chosen

            report.add_attempt(strat, query, note="0 results")

        return None

    # ── Master resolution entry point ──────────────────────────────────────

    def _parse_summary_to_resolved(self, summary: dict) -> "ResolvedAssembly":
        """Convert one raw NCBI DocumentSummary dict into a ResolvedAssembly.
        Shared by all resolution strategies so the output is always identical
        regardless of which strategy succeeded."""
        assembly_uid = str(summary.get("Uid", "") or summary.get("AssemblyId", ""))
        assembly_accession = str(summary.get("AssemblyAccession", ""))
        synonym = summary.get("Synonym", {}) or {}
        refseq_accession = str(synonym.get("RefSeq", "") or "")
        taxon_id_raw = summary.get("Taxid")
        try:
            taxon_id = int(taxon_id_raw) if taxon_id_raw else None
        except (TypeError, ValueError):
            taxon_id = None
        status = str(summary.get("AssemblyStatus", ""))
        assembly_level = ASSEMBLY_LEVEL_MAP.get(status.lower(), status.lower() or "unknown")
        meta_stats = _parse_meta_stats(str(summary.get("Meta", "")))
        ftp_path = str(summary.get("FtpPath_RefSeq", "") or "")
        if not ftp_path or ftp_path == "na":
            ftp_path = str(summary.get("FtpPath_GenBank", "") or "")
        if ftp_path == "na":
            ftp_path = ""
        return ResolvedAssembly(
            assembly_uid=assembly_uid,
            assembly_accession=assembly_accession,
            refseq_accession=refseq_accession,
            taxon_id=taxon_id,
            assembly_level=assembly_level,
            ftp_path=ftp_path,
            meta_stats=meta_stats,
        )

    def resolve_assembly(
        self,
        strain_input: "StrainInput",
        forced_accessions: dict = None,
        fallback_hints: dict = None,
        debug: bool = False,
    ) -> tuple[Optional["ResolvedAssembly"], "StrainResolutionReport"]:
        """Resolve a strain to a ResolvedAssembly using the full strategy
        chain. Always returns a (result, report) tuple -- result is None only
        when every strategy failed.

        Resolution order (most to least trusted):
          0.   Forced accession   -- forced_accessions.csv, bypasses all search
          1.   NCBI ATCC assembly -- "ATCC nnnnn" in the assembly database
          2.   NCBI BioSample     -- ATCC -> BioSample -> assembly cross-ref,
                                     with organism cross-validation
          3a.  NCBI organism      -- "[Organism]" field (broad, flags verify)
          3b.  NCBI free-text     -- plain text (broadest, flags verify)
          4.   Fallback hint      -- fallback_hints.csv, last resort before
                                     declaring failure (flags verify)
        """
        from .resolution_report import Strategy, StrainResolutionReport

        report = StrainResolutionReport(
            strain=strain_input.strain,
            atcc_number=strain_input.atcc_number,
        )

        # ── Strategy 0: forced accession (always wins) ───────────────────
        if forced_accessions and strain_input.strain in forced_accessions:
            pinned = forced_accessions[strain_input.strain]
            log.info("Using FORCED accession %s for %r", pinned, strain_input.strain)
            from .overrides import resolve_via_accession
            resolved = resolve_via_accession(pinned, self)
            if resolved:
                report.add_attempt(Strategy.FORCED_ACCESSION, pinned, result=pinned)
                report.record_success(Strategy.FORCED_ACCESSION, pinned)
                return resolved, report
            report.add_attempt(Strategy.FORCED_ACCESSION, pinned,
                               note="NCBI returned no record for this accession")

        # ── Strategy 1: ATCC → assembly ──────────────────────────────────
        summary = self._try_ncbi_atcc_assembly(strain_input, report)
        if summary:
            resolved = self._parse_summary_to_resolved(summary)
            report.record_success(Strategy.NCBI_ATCC_ASSEMBLY, resolved.assembly_accession)
            return resolved, report

        # ── Strategy 2: ATCC → BioSample → assembly (organism-validated) ─
        summary = self._try_ncbi_biosample_link(strain_input, report)
        if summary:
            resolved = self._parse_summary_to_resolved(summary)
            report.record_success(Strategy.NCBI_BIOSAMPLE_LINK, resolved.assembly_accession)
            return resolved, report

        # ── Strategy 3: organism name (broad) ────────────────────────────
        summary = self._try_ncbi_organism(strain_input, report)
        if summary:
            resolved = self._parse_summary_to_resolved(summary)
            # _try_ncbi_organism records which of ORGANISM/FREETEXT won via
            # the last attempt that has a result.
            winning = next(
                (a.strategy for a in reversed(report.attempts) if a.result),
                Strategy.NCBI_ORGANISM,
            )
            report.record_success(winning, resolved.assembly_accession)
            return resolved, report

        # ── Strategy 4: fallback hint (last resort) ──────────────────────
        if fallback_hints and strain_input.strain in fallback_hints:
            hint = fallback_hints[strain_input.strain]
            log.info("All searches failed; trying FALLBACK hint %s for %r",
                     hint, strain_input.strain)
            from .overrides import resolve_via_accession
            resolved = resolve_via_accession(hint, self)
            if resolved:
                report.add_attempt(Strategy.FALLBACK_HINT, hint, result=hint)
                report.record_success(Strategy.FALLBACK_HINT, hint)
                return resolved, report
            report.add_attempt(Strategy.FALLBACK_HINT, hint,
                               note="NCBI returned no record for this hint accession")

        log.warning(
            "All resolution strategies failed for strain=%r atcc=%r",
            strain_input.strain, strain_input.atcc_number,
        )
        return None, report

    # ---- downloading sequence data via NCBI FTP -----------------------

    def download_via_ftp(self, ftp_path: str, fasta_path: Path, gbk_path: Path) -> None:
        """Download the genome FASTA and GenBank files for an assembly using
        the FTP path we already got from esummary -- no elink step needed.

        NCBI's FTP layout is:
          <ftp_path>/<assembly_name>_genomic.fna.gz   <- FASTA (gzipped)
          <ftp_path>/<assembly_name>_genomic.gbff.gz  <- GenBank (gzipped)

        where <assembly_name> is the last component of ftp_path itself, e.g.
          .../GCF_000193495.2_ASM19349v2/GCF_000193495.2_ASM19349v2_genomic.fna.gz
        """
        # gzip and urllib.request are only needed in this method, so we
        # import them here rather than at the top of the file -- keeps the
        # module's top-level imports focused on what's always needed.
        import gzip
        import urllib.request

        # The assembly name is simply the last path component of ftp_path.
        # e.g. "GCF_000193495.2_ASM19349v2"
        assembly_name = ftp_path.rstrip("/").split("/")[-1]

        fasta_url = f"{ftp_path}/{assembly_name}_genomic.fna.gz"
        gbk_url   = f"{ftp_path}/{assembly_name}_genomic.gbff.gz"

        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        gbk_path.parent.mkdir(parents=True, exist_ok=True)

        for url, dest_path, label in [
            (fasta_url, fasta_path, "FASTA"),
            (gbk_url,   gbk_path,  "GBK"),
        ]:
            log.info("Downloading %s: %s", label, url)
            # NCBI's FTP server rejects urllib's default "Python-urllib/3.x"
            # user-agent string, so we set a plain but legitimate one.
            req = urllib.request.Request(url, headers={"User-Agent": "Biopython"})
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    # The files are gzip-compressed on NCBI's end. We
                    # decompress on the fly and write plain text to disk,
                    # so the rest of the pipeline (Bio.SeqIO etc.) can
                    # read them without needing to know about compression.
                    compressed = response.read()
                    text = gzip.decompress(compressed).decode("utf-8")
                    dest_path.write_text(text, encoding="utf-8")
                log.info("Saved %s -> %s (%d bytes)", label, dest_path.name, dest_path.stat().st_size)
            except Exception as exc:
                # Re-raise so pipeline.py can catch it and log it properly
                # against this strain's name rather than a raw URL.
                raise RuntimeError(f"Failed to download {label} from {url}: {exc}") from exc
            time.sleep(self.request_delay)  # stay polite even to FTP
