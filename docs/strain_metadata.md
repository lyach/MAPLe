# Using the strain-metadata pipeline

## Overview

Two entry points share the `src/strain_metadata/` package:

| Script | Input | Sequences | Output |
|---|---|---|---|
| [`scripts/external_tools/get_strain_metadata.py`](../scripts/external_tools/get_strain_metadata.py) | `align_strains.csv` | Downloads FASTA + GBK | `data/external/strains_metadata.csv` |
| [`scripts/external_tools/get_strain_metadata_bacdive.py`](../scripts/external_tools/get_strain_metadata_bacdive.py) | BacDive parquet | Metadata only (no FASTA/GBK) | sidecar CSV + `<input>_strain_meta.parquet` |

Both resolve an NCBI genome assembly per isolate, flag uncertain matches, and write a run log. The align path also computes sequence stats from downloaded FASTA. The BacDive path fills size / contig / N50 from NCBI assembly `Meta` stats and left-joins the new columns onto the original parquet.

```
python scripts/external_tools/get_strain_metadata.py
python scripts/external_tools/get_strain_metadata_bacdive.py --limit 20 --debug
```

---

## Align strains (`get_strain_metadata.py`)

For each strain in `data/metadata/ncbi/align_strains.csv` the script resolves the matching NCBI genome assembly, downloads the FASTA and GenBank files, computes derived sequence statistics, validates them, and writes everything to `data/external/strains_metadata.csv`.

### What it produces

| Output | Path | Contents |
|---|---|---|
| Metadata table | `data/external/strains_metadata.csv` | One row per resolved strain (18 columns) |
| Failure report | `data/external/failures.csv` | Diagnostics for any strain that could not be resolved |
| FASTA files | `data/external/fasta/<accession>.fna` | Genome sequence, one per assembly |
| GenBank files | `data/external/gbk/<accession>.gbk` | Annotated genome, one per assembly |
| Run log | `logs/run_<timestamp>.log` | Full record of every API call and decision |

### Output CSV columns

The standard schema (`strain`, `atcc_number`, `ncbi_refseq`, `ncbi_assembly_accession`, `ncbi_taxon_id`, `genus`, `family`, `order`, `phylum`, `assembly_level`, `genome_size_bp`, `gc_content_pct`, `num_contigs`, `n50`, `sequence_hash`, `date_retrieved`) plus two confidence columns:

- **`genus` / `family` / `order` / `phylum`** — NCBI Taxonomy lineage ranks for `ncbi_taxon_id`, filled in a batched lookup after assemblies are resolved. Blank if the taxon ID is missing or a rank is absent from the lineage.
- **`resolution_method`** — how the strain was matched (e.g. *NCBI assembly search by ATCC number*, *NCBI BioSample → assembly cross-reference*, *organism name*, *forced accession*).
- **`requires_verification`** — `yes` or `no`. Set to `yes` when the match was organism-level only, when a fallback hint was used, when the organism didn't cross-check, or when a genome statistic fell outside the expected range for the genus.

### Align resolution order

For each strain the script tries the following strategies in order and stops at the first success.

| Order | Strategy | Strain-confirmed? |
|---|---|---|
| 0 | **Forced accession** — pinned in `forced_accessions.csv` | Yes |
| 1 | **Recorded INSDC accession** — unused on the align CSV | Yes |
| 2 | **NCBI assembly search** by `"ATCC nnnnn"` | Yes |
| 3 | **NCBI BioSample** → assembly cross-reference (organism-validated) | Yes |
| 4 | **Strain-level taxid** — unused on the align CSV | Yes |
| 5 | **NCBI organism search** by species name | No → flags verify |
| 6 | **Fallback hint** — pinned in `fallback_hints.csv` | No → flags verify |
| — | If all fail → written to `failures.csv` | — |

Strategy 3 exists because NCBI's BioSample records store the ATCC number in a `culture_collection` field even when the assembly database itself does not index it.

```
python scripts/external_tools/get_strain_metadata.py --limit 3 --debug
python scripts/external_tools/get_strain_metadata.py
```

`--force` re-downloads FASTA/GBK even if cached. Metadata is always re-resolved (the align table is small).

---

## BacDive parquet (`get_strain_metadata_bacdive.py`)

Reads a BacDive table such as `data/external/bacdive_growth_full.parquet` (one row per strain × medium), collapses it to **unique `bacdive_id`**, resolves NCBI assembly metadata for each isolate, then left-joins the new columns back onto the **full** original parquet.

No FASTA/GBK download. `sequence_hash` and computed `gc_content_pct` are left blank; `genome_size_bp` / `num_contigs` / `n50` come from NCBI assembly stats when present.

### What it produces

| Output | Path | Contents |
|---|---|---|
| Isolate sidecar | `data/external/bacdive_strains_metadata.csv` | One row per resolved isolate (`bacdive_id` + standard schema) |
| Failure report | `data/external/bacdive_strain_meta_failures.csv` | Diagnostics for unresolved isolates this run |
| Merged parquet | `<input_stem>_strain_meta.parquet` | Original rows plus NCBI columns |
| Run log | `logs/run_<timestamp>.log` | Full record of every API call and decision |

Default input: `data/external/bacdive_growth_full.parquet` → `data/external/bacdive_growth_full_strain_meta.parquet`.

### Columns added to the parquet

Existing BacDive columns (`genus`, `family`, `order`, `phylum`, `gc_content`) are **not** overwritten. These NCBI fields are appended when absent:

`atcc_number`, `ncbi_refseq`, `ncbi_assembly_accession`, `ncbi_taxon_id`, `assembly_level`, `genome_size_bp`, `gc_content_pct`, `num_contigs`, `n50`, `sequence_hash`, `date_retrieved`, `resolution_method`, `requires_verification`.

Unresolved isolates keep empty metadata columns on their growth rows (and appear in the failures CSV).

### BacDive resolution order

Uses the isolate-identity columns already on the parquet. Organism-name search is **off** (a species hit is not this strain's genome). BV-BRC / IMG accessions and non-strain taxids are skipped. `n16s_accession` is not used.

| Order | Strategy | When |
|---|---|---|
| 0 | Forced accession | `forced_accessions.csv`, keyed by `bacdive_id` or `strain_name` |
| 1 | Recorded INSDC accession | `genome_accession_source == "insdc"` and `GCA_`/`GCF_` |
| 2 | Culture-collection assembly search | each id in `culture_collection_ids`, plus `dsm_number` / ATCC; queries `"DSM 1313"` and `"DSM:1313"` |
| 3 | BioSample → assembly | same collection ids |
| 4 | Strain-level taxid | `ncbi_taxid` set **and** `ncbi_taxid_level` is `strain` |
| 5 | Organism name | not run |
| 6 | Fallback hint | `fallback_hints.csv` |
| — | If all fail → failures CSV | — |

```
python scripts/external_tools/get_strain_metadata_bacdive.py --limit 20 --debug
python scripts/external_tools/get_strain_metadata_bacdive.py
python scripts/external_tools/get_strain_metadata_bacdive.py \
  --input data/external/bacdive_growth_full.parquet \
  --output data/external/bacdive_growth_full_strain_meta.parquet
```

`--force` re-resolves isolates even if they are already in the sidecar CSV. Without `--force`, a rerun skips `bacdive_id`s already in the sidecar (interrupted runs resume). `--limit` applies to unique isolates, not parquet rows.

---

## First-time setup

### 1. Install dependencies

```
pip install biopython pandas pyarrow
```

### 2. NCBI account

NCBI requires an email address to identify Entrez API users. On the first run the script will prompt you for one and save it locally to `.ncbi_credentials.json` (gitignored — never commit it). You will not be asked again on that machine.

To skip the prompt (e.g. on a shared server), set environment variables instead:

```
export NCBI_ENTREZ_EMAIL="you@lab.edu"
export NCBI_API_KEY="..."        # optional, raises the rate limit
```

---

## Accuracy safeguards

Two independent checks guard against wrong matches:

1. **Organism cross-check.** When a strain is matched via BioSample, the organism NCBI reports is compared against the expected genus and species. A mismatch (e.g. a mislabeled collection number) sets `requires_verification = yes` and is logged loudly.

2. **Genus range validation.** Each genome's size and GC content are checked against expected ranges for its genus. Values outside the range are flagged. Missing stats (BacDive metadata-only rows with no GC%) are skipped, not flagged.

A flag does **not** mean the strain is wrong — it means a human should glance at it.

---

## Manual overrides

Two optional files in `data/metadata/ncbi/` let you control resolution. Both share the format `strain,accession,note`. For BacDive, `strain` may be `bacdive_id` or `strain_name`.

### `forced_accessions.csv` — always wins

```
strain,accession,note
Caulobacter vibrioides,GCF_000006905.2,CB15 reference — verified on NCBI
```

### `fallback_hints.csv` — last resort

```
strain,accession,note
Komagataeibacter xylinus,GCF_009834365.1,best guess, not yet confirmed
```

---

## Fixing a flagged or failed strain

1. Open the failures CSV (for failures) or the run summary (for flagged strains).
2. Each failure row lists every strategy tried, why each failed, any rejected candidate accessions, and a ready-to-paste NCBI search URL.
3. Confirm the correct assembly on NCBI.
4. Add the accession to `forced_accessions.csv`.
5. Re-run. The override is picked up automatically.

---

## Command-line options

Shared flags:

| Flag | Effect |
|---|---|
| `--limit N` | Only process the first N isolates (quick test) |
| `--debug` | Show full DEBUG detail on the console (always in the log file) |
| `--quiet` | Console shows warnings/errors only |
| `--reset-credentials` | Forget the saved NCBI email/API key and prompt again |
| `--input PATH` | Different input CSV / parquet |
| `--output PATH` | Align: metadata CSV. BacDive: merged parquet |

Align-only: `--force` re-downloads FASTA/GBK.

BacDive-only: `--force` re-resolves despite the sidecar cache; `--metadata-csv` / `--failures-csv` override sidecar paths.

---

## Caching and re-runs

**Align.** Metadata is always re-resolved from NCBI. FASTA and GBK files already on disk are reused unless `--force`.

**BacDive.** Resolved `bacdive_id`s are appended to the sidecar as they succeed, so an interrupted run resumes. `--force` deletes that cache and starts over. The merged parquet is rewritten at the end of each run.

---

## Module layout

```
scripts/external_tools/
├── get_strain_metadata.py           align CSV entry point
└── get_strain_metadata_bacdive.py   BacDive parquet entry point
src/strain_metadata/
├── config.py
├── logging_setup.py
├── credentials.py
├── align_loader.py                  align_strains.csv + StrainInput
├── bacdive_loader.py                parquet → unique isolates + merge
├── overrides.py
├── ncbi_client.py
├── sequence_stats.py
├── validation.py
├── taxonomy.py
├── resolution_report.py
└── pipeline.py
```

---

## Known limitations

- **Different identifiers across databases.** When a culture-collection ID has no NCBI `culture_collection` cross-reference, the isolate can only be matched via a recorded INSDC accession, a strain-level taxid, or a manual pin.
- **Genus ranges are approximate.** The validation ranges are deliberately wide but may still flag legitimate strains at the boundary.
- **BacDive organism search is disabled.** Species-level NCBI hits are not this isolate's genome; pin those cases in `forced_accessions.csv`.
