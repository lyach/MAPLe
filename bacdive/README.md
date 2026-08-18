# BacDive pipeline

Downloads a taxonomy-scoped subset of the [BacDive](https://bacdive.dsmz.de)
database and builds a tidy table with a binary `growth_class` target that plugs
straight into the modeling pipeline (`docs/modeling_pipeline.md`).

The public BacDive v2 API requires **no credentials**.

## Targets

The extraction **target** decides what one row is. Set it via the `target:` key
in [`bacdive/taxa.yaml`](taxa.yaml) or the `--target` CLI flag:

| Target | One row = | Label (`growth_class`) | Real negatives? | Output CSV |
| --- | --- | --- | --- | --- |
| `medium_growth` *(default)* | `(strain, culture medium)` | 1 = grows, 0 = no growth | **No** — positive-only (see PU notes below) | `data/external/bacdive_growth.csv` |
| `metabolite_utilization` | `(strain, metabolite, assay kind)` | 1 = utilizes (`+`), 0 = not (`-`) | **Yes** — balanced `+`/`-` | `data/external/bacdive_metabolite.csv` |

`metabolite_utilization` is the closest translatable analog to
"genome + nutrient -> phenotype" and, unlike `medium_growth`, has real,
trainable/evaluable negatives.

## Layout

| Path | Role |
| --- | --- |
| `bacdive/taxa.yaml` | User-editable taxonomic scope + target |
| `src/bacdive/` | Engine package (imported as `src.bacdive`) |
| `scripts/get_bacdive_dataset.py` | CLI entry point |
| `config/dataset/curves_dataset_bacdive.yaml` | Curves config (medium_growth, optional interim build) |
| `config/dataset/modeling_dataset_bacdive.yaml` | Modeling dataset config (medium_growth) |
| `config/experiments/class_bacdive.yaml` | Experiment config (medium_growth) |
| `config/dataset/curves_dataset_bacdive_metabolite.yaml` | Curves config (metabolite_utilization, optional interim build) |
| `config/dataset/modeling_dataset_bacdive_metabolite.yaml` | Modeling dataset config (metabolite_utilization) |
| `config/experiments/class_bacdive_metabolite.yaml` | Experiment config (metabolite_utilization) |

> The engine lives under `src/bacdive/` (not a top-level `bacdive/` package) so
> it does not shadow the installed `bacdive` PyPI client, which it wraps.

## Usage

1. Edit the scope + target in [`bacdive/taxa.yaml`](taxa.yaml):

```yaml
target: metabolite_utilization   # or: medium_growth
taxa:
  - Escherichia coli
  - Bacillus subtilis
  - Lactobacillus
```

2. Download and build the dataset:

```bash
python scripts/get_bacdive_dataset.py --taxa bacdive/taxa.yaml
# override the target from the CLI:
python scripts/get_bacdive_dataset.py --target metabolite_utilization
# quick smoke test on the first few records:
python scripts/get_bacdive_dataset.py --limit 5
```

Common outputs (target-dependent CSV name):
- `data/external/bacdive_growth.csv` / `bacdive_metabolite.csv`
- `data/external/bacdive_failures.csv` — taxa that returned nothing / errored
- `data/external/bacdive_cache/<id>.json` — raw per-record cache
- `logs/run_<timestamp>.log` — full run log

To rebuild a table from an existing cache (no API), edit
[`bacdive/build_from_cache.yaml`](build_from_cache.yaml) and run:

```bash
python scripts/build_bacdive_from_cache.py
python scripts/build_bacdive_from_cache.py --limit 100   # smoke test
```

3. Feed it into the modeling pipeline:

```bash
# medium_growth — experiment reads data/external/bacdive_growth_150k.parquet
python scripts/modeling/run_experiment.py config/experiments/class_bacdive.yaml

# metabolite_utilization
python scripts/modeling/run_experiment.py config/experiments/class_bacdive_metabolite.yaml

# optional: write an interim copy via the curves config
python scripts/modeling/create_modeling_dataset.py config/dataset/curves_dataset_bacdive.yaml
python scripts/modeling/create_modeling_dataset.py config/dataset/curves_dataset_bacdive_metabolite.yaml
```

## Output schemas

**`medium_growth`** — one row per `(strain, culture medium)`:

| Group | Columns |
| --- | --- |
| identifiers | `bacdive_id`, `strain_name`, `media_name`, `mediadive_id` |
| target | `growth_class` (1 = growth, 0 = no growth) |
| strain features | `genus`, `family`, `order`, `phylum`, `gram_stain`, `cell_shape`, `motility`, `oxygen_tolerance`, `nutrition_type`, `spore_formation`, `gc_content`, `genome_accession`, `n16s_accession` |
| condition (strain-level) | `temperature`, `ph` |
| media features | `n_ingredients`, `has_yeast_extract`, `has_peptone`, `has_agar`, `has_blood_components`, `has_nacl`, `complexity` |
| provenance | `composition_raw`, `refs` |

**`metabolite_utilization`** — one row per `(strain, metabolite, assay kind)`:

| Group | Columns |
| --- | --- |
| identifiers | `bacdive_id`, `strain_name`, `substrate_name`, `chebi_id`, `kind_of_utilization` |
| target | `growth_class` (1 = utilizes `+`, 0 = `-`) |
| strain features | `genus`, `family`, `order`, `phylum`, `gram_stain`, `cell_shape`, `motility`, `oxygen_tolerance`, `nutrition_type`, `spore_formation`, `gc_content`, `genome_accession`, `n16s_accession`, `temperature`, `ph` |
| provenance | `refs` |

The label is named `growth_class` in both schemas only so it plugs into the
existing classification `task` unchanged. For `metabolite_utilization`, the
substrate identifier is the explicit `substrate_name` (not `media_name`), so
`lomo` (leave-one-medium-out) does not apply — use `loso` / `grouped` /
`random`. `chebi_id` is emitted so substrate chemistry (ChEBI -> SMILES ->
fingerprint) can be joined later as a `path:` embedding.

## How it maps to the modeling pipeline

The output is a "curves-like" table consumed by `src/modeling/dataset.py`. It
provides `strain_name` and `growth_class` (plus `media_name` for
`medium_growth`), which is all the `growth_class` task needs. `loso` CV groups
on `strain_name`; `lomo` groups on `media_name` (medium_growth only).

## PU-aware handling (`medium_growth`)

**`medium_growth` is positive-only.** BacDive records
media on which a strain *was cultivated*; explicit `growth = "no"` is almost
never present. A missing `(strain, medium)` pair means
"untested", **not** "no growth". The pipeline logs a PU warning
when a `medium_growth` run comes back with zero negatives.


## Other caveats

- **Only `growth_class`-style binary targets are derivable.** BacDive has no
  growth-curve metrics, so `max_od` / `growth_rate` are not available here.
- **Conditions are strain-level, not per-datapoint.** `temperature`, `ph`, and
  `oxygen_tolerance` are recorded in blocks keyed by publication reference, so
  they are summarised per strain and attached to every row for that strain.
- **Large media vocabulary.** BacDive uses hundreds of distinct DSMZ/CIP media,
  so identity one-hot on `media_name` does not transfer under `lomo`; the
  `medium_growth` config relies on parsed composition features instead.
- Missing categorical values are written as `"unknown"` so downstream
  `OneHotEncoder` never sees NaN.
