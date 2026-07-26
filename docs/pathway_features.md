# Building metabolic-pathway features with DRAM

## Overview

This pipeline derives a strain × KEGG-pathway feature matrix from the genome FASTA files in `data/external/fasta/` (see [`docs/plans/pathway_features.md`](plans/pathway_features.md) for the original design). Gene prediction, KO annotation, and pathway-completeness scoring are all done by **DRAM** (`DRAM.py annotate` + `DRAM.py distill`), parsed locally into a continuous strain × pathway completeness matrix — the canonical modeling input. No dimensionality reduction is precomputed here: PCA on this matrix must be fit inside the modeling pipeline's cross-validation loop, on training strains only, not globally in this script (see "What it produces" below).

**Confirmed working end-to-end on Nibi** (25 strains → `kegg_pathway_completeness.csv`, 25 strains × 52 pathways after dropping 46 constant columns). Getting there required patching four separate upstream bugs, all documented below and searchable by their `RERUN_WHEN_*`/`Known upstream bug` markers — none of them are issues in this repo's own code.

> **⚠ KEGG license pending — KOfam-only is a temporary fallback, not the final config.**
> The KEGG license needed for UniRef90 (and true KEGG gene data) hasn't arrived yet. Until it does, `setup_dram_env.sh` runs DRAM with `--skip_uniref` (KOfam-only), which is enough to unblock modeling-pipeline development this week but is **not the final annotation**. Search this repo for `RERUN_WHEN_KEGG_LICENSED` for the exact TODO marker — once the license lands, DRAM must be re-run without `--skip_uniref` and every downstream pathway-feature CSV regenerated. Do not treat KOfam-only output as final for any published result.
>
> Separately (unrelated to the license): dbCAN and VOGDB are also currently excluded via `--select_db`, but only because DRAM-bio 1.5.0's hardcoded download URLs for both are stale (`bcb.unl.edu` and `fileshare.csb.univie.ac.at` respectively — both hosts have since moved/redirected). MEROPS is **not** gated by anything and downloads normally. Search for `RERUN_WHEN_DBCAN_URL_FIXED` / `RERUN_WHEN_VOGDB_URL_FIXED`.

**This is the first HPC-dependent step in the repo.** DRAM's own docs state that even its lightest configuration (KOfam-only, skipping UniRef90) needs ~128GB RAM to build its databases and <50GB RAM to annotate — far more than a typical dev machine has. The pipeline therefore runs across **two environments**:

1. **Local** (this repo) — `prepare_dram_manifest.py` validates the input FASTA files and prints the commands to ship them out.
2. **Remote** (a DRAC — Digital Research Alliance of Canada — HPC cluster, Slurm-based) — you run DRAM by hand using the artifacts in `scripts/hpc/`.
3. **Local again** (this repo) — `build_pathway_features.py` parses DRAM's output back into the modeling-ready CSVs.

Nothing in this repo can execute steps 2 for you — a DRAC account/allocation and a manual `rsync`/`sbatch`/`rsync` round trip are required.

---

## What it produces

| Output | Path | Contents |
|---|---|---|
| FASTA manifest | `data/interim/dram/fasta_manifest.csv` | strain ↔ accession ↔ FASTA filename, cross-validated against `strains_metadata.csv` |
| DRAM distillate (copied back from DRAC) | `data/external/dram/distillate/product.tsv` | genome × module completeness percentages, DRAM's raw output |
| **Completeness matrix — canonical modeling input** | `data/external/kegg_pathway_completeness.csv` | strain × pathway, continuous 0.0-1.0 fractions. **This is what downstream feature pipelines should read.** Always written. |
| Presence/absence matrix — diagnostic only | `data/external/kegg_pathway_presence.csv` | strain × pathway, 0/1, thresholded from completeness. Manual inspection only, **not a modeling input**. Written only with `--emit-diagnostics`. |
| EDA PCA scores — exploratory only | `data/external/eda_pca_exploratory.csv` | strain × principal component, numeric. A single fit across *all* strains — **must never be joined into training data**, since that leaks test-strain information ahead of cross-validation. Written only with `--emit-diagnostics`. |
| EDA PCA loadings | `data/external/eda_pca_exploratory_loadings.csv` | component × pathway, for the exploratory fit above. Written only with `--emit-diagnostics`. |
| Gene × pathway membership (additive, optional) | `data/external/dram/gene_pathway_membership/gene_pathway_membership_{strain}.csv` | gene × KEGG-module-ID, binary, multi-label. One file per strain. Independent of the completeness matrix — see step 4 below. |
| KEGG module lookup | `data/external/dram/gene_pathway_membership/kegg_module_lookup.csv` | module ID → human-readable name, written once |
| Run log | `logs/run_<timestamp>.log` | full record of all three local scripts |

---

## Step-by-step workflow

### 1. Local: prepare the manifest

```
python scripts/prepare_dram_manifest.py
```

Cross-checks `data/external/strains_metadata.csv` against `data/external/fasta/`, writes `data/interim/dram/fasta_manifest.csv`, and prints the rsync + sbatch commands to ship the FASTA files and `scripts/hpc/` to your DRAC cluster. Any FASTA file without a metadata row (or vice versa) is excluded and logged rather than silently included.

The printed `sbatch` commands include your Slurm billing account, filled in automatically. **No lab's account is hardcoded anywhere in this repo** — the first time you run this script, you'll be prompted for your own (e.g. `def-yourpi` or `rrg-yourpi`; find yours on the cluster with `sacctmgr show associations user=$USER format=account%30`), and it's cached locally in `.drac_credentials.json` (gitignored, mirrors how `strain_metadata`'s NCBI email works — see `src/pathway_features/drac_credentials.py`). Set the `DRAC_ACCOUNT` environment variable instead to skip the prompt (e.g. in CI).

### 2. Remote: run DRAM on DRAC

`setup_dram_env.sh`'s database-processing step can need up to ~128GB RAM — too heavy to run directly on a shared login node (Alliance monitors and throttles/kills heavy processes there). Submit it through Slurm instead, as its own job (`prepare_dram_manifest.py` prints both commands below for you, with your account and paths already filled in — the paths are `$HOME`-relative so they work regardless of your current directory):

```
sbatch --account=<your-cpu-account> --mem=128G --cpus-per-task=8 --time=6:00:00 \
  --job-name=dram-setup --output=%x-%j.out \
  --wrap="bash $HOME/scratch/dram/hpc/setup_dram_env.sh"
```

This needs internet access for `pip install DRAM-bio`, the `barrnap` download, and the database download — confirmed available from Nibi's compute nodes specifically (not all Alliance clusters allow this; some restrict compute-node internet entirely, in which case the network-heavy parts would need to run from a login node instead and only the heavy processing submitted via Slurm).

Once that job completes (check with `squeue -u $USER`, inspect the `.out` log for errors), submit the actual annotation job (the second printed command):

```
sbatch --account=<your-cpu-account> $HOME/scratch/dram/hpc/run_dram_annotate.sbatch
```

Setup uses Alliance's module system plus a Python virtualenv rather than conda — Alliance's docs discourage building conda environments in `$HOME`, since the huge file count strains their shared filesystem. Check `module spider <name>` for `prodigal`, `hmmer`, `mmseqs2`, `trnascan-se`, and `bedtools` (a `barrnap` dependency) first; these are commonly available as Alliance modules (confirmed on Nibi). `barrnap` itself typically isn't, but is a pure-Perl script `setup_dram_env.sh` installs by hand — **pinned to tag `0.9`, not `master`**. `master` has grown into a much larger rewrite (barrnap 1.x) that checks for 12 external tools at startup (`cmsearch`, `cmscan`, `bedtools`, `aragorn`, `any2fasta`, `seqkit`, `make`, `transterm`, `ziggypep`, `pyrodigal`, `diamond`, `taxonkit`) — discovered the hard way, one missing dependency at a time. Tag `0.9` is the original, minimal version DRAM-bio 1.5.0 was actually built against: it only needs `nhmmer` (covered by the `hmmer` module) and `bedtools`. `pip install DRAM-bio` itself is confirmed working on Nibi — Alliance has it pre-built in their own wheelhouse (`+computecanada` suffix), so it (and nearly all its dependencies) install from local disk, not PyPI.

**`setup_dram_env.sh` also hand-installs the official Prodigal 2.6.3 static binary and puts it ahead of the module on `PATH`** — Nibi's `prodigal/2.6.3` module is a real module that loads fine, but its build is broken: it silently produces truncated GFF output missing the `ID=`/`partial=`/`start_type=`/score fields DRAM's own annotation code parses. Confirmed by running the identical input through the [official static v2.6.3 binary](https://github.com/hyattpd/Prodigal/releases/tag/v2.6.3) with the exact same flags — same Prodigal version, correct output — so this is specific to Alliance's build, not a real Prodigal/DRAM-bio compatibility gap. Marked `RERUN_WHEN_ALLIANCE_PRODIGAL_FIXED`, though there's no real downside to leaving the override in place indefinitely.

#### Known upstream bug: `DRAM.py annotate` crashes on missing gene IDs

If you ever see `AttributeError: 'NoneType' object has no attribute 'group'` in `mag_annotator/annotate_bins.py`'s `annotate_gff`, it means whatever `prodigal` binary is on `PATH` isn't producing `ID=`-tagged GFF output (the regex `ID=\d*_\d*;` won't match, and the code has no fallback). This is what the Prodigal override above prevents. If it recurs on a different cluster, verify with the same test: run `prodigal -i <fasta> -p meta -g 11 -f gff -o test.gff ...` (DRAM's exact invocation, see `run_prodigal()` in `annotate_bins.py`) and check the output has `ID=1_1;partial=...` in the attributes column.

#### Known upstream bug: `DRAM-setup.py` syntax error (DRAM-bio 1.5.0)

The `DRAM-setup.py` entry point installed by `pip install DRAM-bio==1.5.0` (still the latest release on PyPI) has a `SyntaxError` around its `--camper_tar_gz_loc` argument — two `add_argument()` calls got mis-transposed in that release. This is a known, still-unfixed upstream bug ([WrightonLabCSU/DRAM#330](https://github.com/WrightonLabCSU/DRAM/issues/330), [#360](https://github.com/WrightonLabCSU/DRAM/issues/360)), not anything specific to this repo or to DRAC.

**`setup_dram_env.sh` patches this automatically** — no manual step needed. Right after `pip install DRAM-bio`, it runs `python3 -m py_compile` on the installed file; if that fails, it applies an exact-string-match patch (moving the misplaced `help='mmseqs2 database file from ref seq viral gene collection')` line to close the `--viral_loc` call instead of trailing after the CAMPER block) and re-verifies with `py_compile`. If DRAM-bio ever ships a release with a *different* bug in this spot, the patch aborts loudly (rather than guessing) with a pointer back to this section — safe to rerun the whole script if that happens. Idempotent: a second run just sees the file already compiles and skips patching.

#### Known upstream issue: `barrnap`'s `master` branch outgrew what DRAM-bio needs

A naive `git clone` of `barrnap`'s `master` branch (no tag pinned) grabs version 1.10.x, which checks for 12 external tools at startup (`cmsearch`, `cmscan`, `bedtools`, `aragorn`, `any2fasta`, `seqkit`, `make`, `transterm`, `ziggypep`, `pyrodigal`, `diamond`, `taxonkit`) — discovered one missing dependency at a time before realizing the real fix was upstream, not chasing each one down. Tag `0.9` is the original, minimal barrnap DRAM-bio 1.5.0 was actually built against: it only needs `nhmmer` (from the `hmmer` module) and `bedtools`. `setup_dram_env.sh` clones `--branch 0.9` directly, not `master`.

The sbatch job runs `DRAM.py annotate` then `DRAM.py distill`, writing a `distillate/` directory (including `product.tsv`).

Copy the result back:

```
rsync -avz <cluster>:~/scratch/dram/output/distillate/ data/external/dram/distillate/
```

### 3. Local: build the feature CSVs

```
python scripts/build_pathway_features.py config/dataset/pathway_features.yaml
```

Parses `product.tsv`, joins genome IDs back to strain names via `strains_metadata.csv`, and writes `kegg_pathway_completeness.csv` — the canonical, continuous-valued modeling input. Add `--emit-diagnostics` to also write the binary presence/absence matrix and a global exploratory PCA — neither is a modeling input, see the table above. Refuses to overwrite existing outputs unless `--force` is passed (a DRAM run is expensive and remote, so accidental clobbering is guarded against). Fails loudly — not with a silently-shrunk matrix — if any strain from `strains_metadata.csv` is missing from the distillate.

### 4. Local (optional, additive): build the gene x pathway membership matrices

A separate, independent output from `kegg_pathway_completeness.csv` — does not read, write, or affect it in any way. For each strain, produces a binary matrix of every predicted gene (rows) against KEGG module IDs (columns), indicating which module(s) each gene's KO belongs to.

Requires the **raw** `annotation/` output rsynced back from DRAC — the `distillate/` sync alone isn't enough, since this needs gene-level KO calls (`annotations.tsv`), not the genome-level summary:

```
rsync -avz Nibi:scratch/dram/output/annotation/ data/external/dram/annotation/
```

Then:

```
python scripts/build_gene_pathway_matrix.py config/dataset/gene_pathway_membership.yaml
```

Writes one file per strain — `data/external/dram/gene_pathway_membership/gene_pathway_membership_{strain}.csv` — plus `kegg_module_lookup.csv` (module ID → human-readable name, once). Columns are bare KEGG module IDs (`M00001`), not names, so the matrix stays directly joinable with `kegg_pathway_completeness.csv` on the same module-ID axis; look up readable labels from `kegg_module_lookup.csv` separately rather than expecting them in the matrix header.

**Multi-label, not one-hot.** A gene's KO can legitimately belong to more than one KEGG module — e.g. malate dehydrogenase (K00024) belongs to 9 different modules in DRAM's own reference data, since it's a central-carbon enzyme feeding the TCA cycle, glyoxylate cycle, and several others at once. Every applicable module gets a 1 for that gene; this is not forced into a single "best" pathway assignment, because that would misrepresent real biology. See `src/pathway_features/gene_pathway_membership.py`'s docstring.

**Column scope — only 31 of `kegg_pathway_completeness.csv`'s 98 columns are covered.** Checked directly: only 32 of the 98 columns in DRAM's `product.tsv` (31 unique module IDs — one, M00153, legitimately appears under two different labels) correspond to a real, ID-able KEGG module, sourced from DRAM's own `module_step_form.tsv`/`etc_module_database.tsv` reference sheets. The other 66 (`CAZy: *`, `Nitrogen metabolism: *`, `SCFA and alcohol conversions: *`, `Other Reductases: *`, `Photosynthesis: *`, `Sulfur metabolism: *`, `Methanogenesis and methanotrophy: *`) are DRAM's own hand-curated marker-gene categories (from a different sheet, `function_heatmap_form.tsv`) that were never official KEGG modules with M-numbers at all — fabricating an ID for them would be actively wrong, so they're excluded from this matrix rather than faked. See `src/pathway_features/kegg_modules.py`'s docstring for the full accounting.

Same `--force` overwrite guard as `build_pathway_features.py`. Fails loudly and writes nothing (not a partial result set) if any strain's accession has zero rows in `annotations.tsv` — this is different from a strain's genes legitimately having no KO hits, which is normal and produces an all-zero-but-present row, not an error. `--refresh-mapping` re-downloads DRAM's reference sheets instead of using the local cache at `data/interim/dram/kegg_module_mapping/`.

---

## Tunables (`config/dataset/pathway_features.yaml`)

Both of these only affect the `--emit-diagnostics` extras — neither touches the canonical `kegg_pathway_completeness.csv` output, which has no tunables (it's just DRAM's completeness fractions, joined to strain names).

| Key | Default | Meaning |
|---|---|---|
| `completeness_threshold` | `0.8` | Fraction of a module's KOs that must be present to call it "present" in the diagnostic binary matrix. Set high to favor precision — with only ~25 strains and ~500 candidate pathways, false positives from partially-annotated genomes are riskier than missing a borderline pathway. |
| `pca_n_components` | `15` | Number of components in the exploratory PCA fit. Runs on raw completeness percentages, so the gradient between e.g. 40%- and 90%-complete pathways isn't lost. |
| `pca_random_state` | `42` | For reproducible exploratory PCA fits. |

---

## Module layout

```
scripts/
├── prepare_dram_manifest.py       entry point — local, pre-DRAM
├── build_pathway_features.py      entry point — local, post-DRAM
├── build_gene_pathway_matrix.py   entry point — local, post-DRAM, additive/optional
└── hpc/
    ├── setup_dram_env.sh          remote — one-time module/venv + database setup
    ├── run_dram_annotate.sbatch   remote — the annotation job
    └── README.md
src/
├── common/
│   └── logging_setup.py           shared console + timestamped-file logging
└── pathway_features/
    ├── config.py                  paths, tunable defaults
    ├── manifest.py                 strain ↔ accession ↔ FASTA-path table
    ├── drac_credentials.py          resolves your Slurm billing account (no lab hardcoded)
    ├── distillate.py                parses product.tsv, joins strain names (canonical output)
    ├── matrix.py                     completeness → presence/absence (--emit-diagnostics only)
    ├── pca.py                          StandardScaler + global PCA (--emit-diagnostics only, EDA use only)
    ├── validation.py                    sanity checks (missing strains, constant columns)
    ├── kegg_modules.py                   KO → KEGG-module-ID mapping (multi-label; sourced from DRAM's own reference sheets)
    └── gene_pathway_membership.py          per-strain gene × module-ID binary matrix (multi-label, additive output)
```

---

## Known limitations

- **Requires a DRAC account** — both `setup_dram_env.sh` and `run_dram_annotate.sbatch` are confirmed working end-to-end on Nibi, but re-check module versions/availability on other clusters, as they can differ.
- **DRAM-bio 1.5.0 ships a broken `DRAM-setup.py`** (see "Known upstream bug" above) — patched automatically by `setup_dram_env.sh`, unrelated to this repo otherwise.
- **Nibi's `prodigal/2.6.3` module produces broken output** (see "Known upstream bug" above) — overridden with an official static binary in `setup_dram_env.sh`/`run_dram_annotate.sbatch`. Worth re-checking `module spider prodigal` output quality on any other cluster before trusting it.
- **`barrnap`'s `master` branch has expanded far beyond what DRAM-bio needs** — pinned to tag `0.9` in `setup_dram_env.sh` instead. See "Known upstream bug" below.
- **KOfam-only annotation — license-pending fallback, not final.** `setup_dram_env.sh` skips UniRef90 (`--skip_uniref`) because the KEGG license needed for it hasn't arrived yet, not because it's the intended long-term config. Marked `RERUN_WHEN_KEGG_LICENSED` in `setup_dram_env.sh` — re-run without `--skip_uniref` once licensed, and regenerate every downstream pathway-feature CSV. Current footprint (~30GB disk / ~128GB RAM to build) was sized against DRAM's documented KOfam-only numbers, not against Nibi's actual available memory partitions — worth confirming with `sinfo` before the full-database rerun, since that job will need proportionally more.
- **dbCAN and VOGDB excluded — unrelated upstream bugs, not licensing.** `setup_dram_env.sh` excludes `dbcan`/`dbcan_fam_activities`/`dbcan_subfam_ec` and `vogdb`/`vog_annotations` via `--select_db`. Both are the same class of problem, different hosts: DRAM-bio 1.5.0 hardcodes download URLs at `bcb.unl.edu` (dbCAN) and `fileshare.csb.univie.ac.at` (VOGDB), and both hosts have since moved — `bcb.unl.edu` now redirects to a webpage instead of serving files, and `fileshare.csb.univie.ac.at` 301-redirects to `fileshare.lisc.univie.ac.at`, which silently produces a 0-byte file rather than erroring at download time (the failure only surfaces later, when `hmmpress` rejects the empty file). Neither database is needed for the KEGG/KOfam pathway-completeness scoring this pipeline actually produces, so both are safe to leave out for now. Marked `RERUN_WHEN_DBCAN_URL_FIXED` / `RERUN_WHEN_VOGDB_URL_FIXED`. MEROPS is unaffected by either issue and downloads normally.
- **Not yet wired into the modeling dataset.** `src/modeling/features.py` lives only on the unmerged `origin/modeling` branch, and its example config references the old `kegg_pathway_pca.csv` path — that file no longer exists (see above). Wiring this in now means pointing a `path:` feature spec at `kegg_pathway_completeness.csv` instead, **and** making sure `features.py` fits any PCA/dimensionality reduction on pathway columns inside its own cross-validation, not by re-reading a precomputed PCA file. Do not repoint that config at `eda_pca_exploratory.csv` as a shortcut — it's a global fit and leaks test-strain information into training.
