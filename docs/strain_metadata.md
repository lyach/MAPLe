# Using `get_strain_metadata.py`

## Overview

This script builds the strain metadata table for the MicrobeNet dataset. For each strain in `data/metadata/ncbi/align_strains.csv` it resolves the matching NCBI genome assembly, downloads the FASTA and GenBank files, computes derived sequence statistics, validates them, and writes everything to `data/external/strains_metadata.csv`.

It is designed to run as a one-liner and to flag any strain whose match is uncertain rather than silently trusting it.

```
python scripts/get_strain_metadata.py
```

---

## What it produces

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

---

## First-time setup

### 1. Install dependencies

```
pip install biopython
```

### 2. NCBI account

NCBI requires an email address to identify Entrez API users. On the first run the script will prompt you for one and save it locally to `.ncbi_credentials.json` (gitignored — never commit it). You will not be asked again on that machine.

To skip the prompt (e.g. on a shared server), set environment variables instead:

```
export NCBI_ENTREZ_EMAIL="you@lab.edu"
export NCBI_API_KEY="..."        # optional, raises the rate limit
```

## How resolution works

For each strain the script tries the following strategies in order and stops at the first success. Each strategy is more permissive (and less certain) than the last.

| Order | Strategy | Strain-confirmed? |
|---|---|---|
| 0 | **Forced accession** — pinned in `forced_accessions.csv` | Yes |
| 1 | **NCBI assembly search** by `"ATCC nnnnn"` | Yes |
| 2 | **NCBI BioSample** → assembly cross-reference (organism-validated) | Yes |
| 3 | **NCBI organism search** by species name | No → flags verify |
| 4 | **Fallback hint** — pinned in `fallback_hints.csv` | No → flags verify |
| — | If all fail → written to `failures.csv` | — |

Strategy 2 exists because NCBI's BioSample records store the ATCC number in a `culture_collection` field even when the assembly database itself does not index it. This recovers strains that strategy 1 misses.

---

## Accuracy safeguards

Two independent checks guard against wrong matches:

1. **Organism cross-check.** When a strain is matched via BioSample, the organism NCBI reports is compared against the expected genus and species. A mismatch (e.g. a mislabeled ATCC number) sets `requires_verification = yes` and is logged loudly.

2. **Genus range validation.** Each genome's size and GC content are checked against expected ranges for its genus. Values outside the range are flagged. This catches a wrong assembly even if resolution thought it succeeded.

A flag does **not** mean the strain is wrong — it means a human should glance at it. The cost of checking a false positive is far lower than letting a wrong genome into the training data.

---

## Manual overrides

Two optional files in `data/metadata/` let you control resolution. Both share the format `strain,accession,note`.

### `forced_accessions.csv` — always wins

Use this when you have **verified** the correct assembly and want to guarantee it is used, bypassing all search.

```
strain,accession,note
Caulobacter vibrioides,GCF_000006905.2,CB15 reference — verified on NCBI
```

### `fallback_hints.csv` — last resort

Use this for an **unverified guess** you'd rather try than give up on. A strain resolved this way is flagged `requires_verification = yes`.

```
strain,accession,note
Komagataeibacter xylinus,GCF_009834365.1,best guess, not yet confirmed
```

The `strain` value must match the `Strain` column in `align_strains.csv` exactly (case-sensitive).

---

## Fixing a flagged or failed strain

1. Open `failures.csv` (for failures) or the run summary (for flagged strains).
2. For failures, each row lists every strategy tried, why each failed, any rejected candidate accessions, and a ready-to-paste NCBI search URL.
3. Click the URL, find the correct assembly on NCBI, and confirm it matches your physical strain (check the ATCC number on the assembly page).
4. Add the accession to `forced_accessions.csv`.
5. Re-run the script. The override is picked up automatically.

---

## Command-line options

| Flag | Effect |
|---|---|
| *(none)* | Process all strains using cached files where available |
| `--limit N` | Only process the first N strains (quick test) |
| `--force` | Re-download FASTA/GBK even if already cached |
| `--debug` | Show full DEBUG detail on the console (always in the log file) |
| `--quiet` | Console shows warnings/errors only |
| `--reset-credentials` | Forget the saved NCBI email/API key and prompt again |
| `--input PATH` | Use a different input CSV |
| `--output PATH` | Write to a different output CSV |

A typical first run on a new machine:

```
python scripts/get_strain_metadata.py --limit 3 --debug
```

Then the full run:

```
python scripts/get_strain_metadata.py
```

---

## Caching and re-runs

The script always re-resolves metadata from NCBI (cheap) but reuses FASTA and GBK files already on disk (expensive). This means edits to your override files take effect on the next run immediately, without re-downloading sequences. Use `--force` to re-download.

If a run is interrupted, simply run it again — completed downloads are reused and it picks up where it left off.

---

## Module layout

The entry-point script is thin; all logic lives in the `strain_metadata/` package.

```
scripts/
└── get_strain_metadata.py        entry point — CLI parsing and wiring only
src/
└── strain_metadata/
    ├── config.py                  paths, constants
    ├── logging_setup.py            console + timestamped-file logging
    ├── credentials.py               NCBI email/API key resolution
    ├── align_loader.py               reads align_strains.csv + name overrides
    ├── overrides.py                   forced + fallback accession loaders
    ├── ncbi_client.py                  NCBI Entrez/FTP calls + strategy chain
    ├── sequence_stats.py                GC% / N50 / hash from FASTA
    ├── validation.py                     genus-range sanity checks
    ├── taxonomy.py                        NCBI lineage (genus/family/order/phylum)
    ├── resolution_report.py               per-strain tracking + summaries
    └── pipeline.py                          orchestration
```

---

## Known limitations

- **Different identifiers across databases.** When ATCC and NCBI use entirely different strain names with no `culture_collection` cross-reference, the strain can only be matched at the organism level (flagged `requires_verification = yes`) or pinned manually. Full resolution of these cases requires ATCC Genome Portal API access.
- **Genus ranges are approximate.** The validation ranges are deliberately wide but may still flag legitimate strains at the boundary. 
