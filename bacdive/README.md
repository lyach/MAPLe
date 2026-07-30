# BacDive growth pipeline

Downloads a taxonomy-scoped subset of the [BacDive](https://bacdive.dsmz.de)
database and builds a tidy **(strain x culture medium)** table with a binary
`growth_class` target that plugs straight into the modeling pipeline
(`docs/modeling_pipeline.md`).

The public BacDive v2 API requires **no credentials**.

## Layout

| Path | Role |
| --- | --- |
| `bacdive/taxa.yaml` | User-editable taxonomic scope (which strains to download) |
| `src/bacdive/` | Engine package (imported as `src.bacdive`) |
| `scripts/get_bacdive_dataset.py` | CLI entry point |
| `config/dataset/modeling_dataset_bacdive.yaml` | Modeling dataset config |
| `config/experiments/class_bacdive.yaml` | Experiment config (growth_class) |

> The engine lives under `src/bacdive/` (not a top-level `bacdive/` package) so
> it does not shadow the installed `bacdive` PyPI client, which it wraps.

## Usage

1. Edit the taxonomic scope in [`bacdive/taxa.yaml`](taxa.yaml):

```yaml
taxa:
  - Escherichia coli
  - Bacillus subtilis
  - Lactobacillus
```

2. Download and build the dataset:

```bash
python scripts/get_bacdive_dataset.py --taxa bacdive/taxa.yaml
# quick smoke test on the first few records:
python scripts/get_bacdive_dataset.py --limit 5
```

Outputs:
- `data/external/bacdive_growth.csv` — one row per (strain, culture medium)
- `data/external/bacdive_failures.csv` — taxa that returned nothing / errored
- `data/external/bacdive_cache/<id>.json` — raw per-record cache
- `logs/run_<timestamp>.log` — full run log

3. Feed it into the modeling pipeline:

```bash
python scripts/create_modeling_dataset.py config/dataset/modeling_dataset_bacdive.yaml
python scripts/run_experiment.py config/experiments/class_bacdive.yaml
```

## Output schema

One row per `(strain, culture medium)`:

| Group | Columns |
| --- | --- |
| identifiers | `bacdive_id`, `strain_name`, `media_name`, `mediadive_id` |
| target | `growth_class` (1 = growth, 0 = no growth) |
| strain features | `genus`, `family`, `order`, `phylum`, `gram_stain`, `cell_shape`, `motility`, `oxygen_tolerance`, `nutrition_type`, `spore_formation`, `gc_content`, `genome_accession`, `n16s_accession` |
| condition (strain-level) | `temperature`, `ph` |
| media features | `n_ingredients`, `has_yeast_extract`, `has_peptone`, `has_agar`, `has_blood_components`, `has_nacl`, `is_defined` |
| provenance | `composition_raw`, `refs` |

## How it maps to the modeling pipeline

The output is a "curves-like" table consumed unchanged by
`src/modeling/dataset.py`: it provides `strain_name`, `media_name`, and
`growth_class`, which is all the `growth_class` task needs. Leave-one-strain-out
(`loso`) CV groups on `strain_name`; leave-one-medium-out (`lomo`) groups on
`media_name`.

## Caveats

- **Only `growth_class` is derivable.** BacDive has no growth-curve metrics, so
  `max_od` / `growth_rate` are not available from this source.
- **Class imbalance.** BacDive records are dominated by `growth = "yes"`;
  explicit negatives are rare. Expect a positive-heavy dataset. A possible
  follow-up is mining negatives from `culture temp` / `pH` "negative" entries.
- **Conditions are strain-level, not per-datapoint.** `temperature`, `ph`, and
  `oxygen_tolerance` are recorded in blocks keyed by publication reference, not
  per medium, so they are summarised per strain and attached to every medium
  row for that strain. This differs from lab data where `atmosphere` /
  `temperature` vary per well.
- **Large media vocabulary.** BacDive uses hundreds of distinct DSMZ/CIP media,
  so identity one-hot on `media_name` does not transfer under `lomo`. The
  default config relies on the parsed composition features (`n_ingredients`,
  `has_*`, `is_defined`) instead, which do transfer across unseen media.
- Missing categorical values are written as `"unknown"` so downstream
  `OneHotEncoder` never sees NaN.
