# Plan: Retrieving Metabolic Pathway for ML Model Features

**Goal:** For each strain in the dataset, recover a feature vector of the metabolic pathways their genome can encode. This will hold information on metabolic capabilities that can be similar/different across strains:
```

                 map00010  map00020  map00190  ...  map01100
E. coli K12           1         1         1    ...       1
B. subtilis           1         1         1    ...       1
C. sporogenes         1         0         1    ...       0
...
```

## Overview

For each strain:
```
[FASTA seq]  -->  gene prediction  -->  [Expressed gene vector] --> map to KEGG IDs --> [KEGG ID vector] --> map to pathways --> [Metabolic Pathway Vector]
```

## Steps

**1. Start from genome FASTA**

These are already available under `data/external/fasta`

**2. Predict proteins** 

Metabolic pathways are defined in terms of functional genes, not raw nucleotide sequences. Predict these using (TOOL TBD).

**3. Annotate proteins with KEGG orthology (KO)**

Each protein can be mapped to a KEGG Orthology (KO) ID, e.g. K01689. This is an identifier used in the KEGG database to group genes and proteins from different species that share the same biological function.

The result per strain will look something like: 
```
gene_001 -> K01689
gene_002 -> K00850
```

**4. Map KOs to KEGG pathways**

Pathway mappings will look like:
```
K01689 -> map00010 (Glycolysis)
K00850 -> map00010, map00030, ...
```
A pathway is considered 'present' if the genome has enough of the required KOs. This can be classified in two ways:

**A. Simple presence/absence:**
Pathway = 1 if any gene in that pathway is detected.

**B. Completeness threshold:**
Pathway = 1 only if ≥ 50% or 80% of required KOs are present. e.g.:

- Glycolysis needs 30 KOs
   - Strain has 27/30 → completeness = 90% → present
   - Strain has 8/30 → completeness = 27% → absent

**5. Build the strain × pathway matrix**

```
# conceptual output
pathway_matrix.shape  # (25, ~500)
```

Each cell will be a binary `0` or `1`, or could also be a continuous value

**6. Reduce matrix with PCA**

- Reduce the vector of pathways (expected to be ~500 pathways), because:
   - Many features will lead to overfitting, especially for just 25 strains
   - Many pathways are highly correlated

- Use PCA (Principal Component Anaslysis) for this. This will find the combinations of pathways that vary most across strains. e.g.:
   `[PC1=-1.8, PC2=0.6, ...]`
   - PC1 might represent “aerobic respiration capacity”
   - PC2 might represent “fermentation metabolism”
- For an `n=25` 10-20 principal components should work

**In the pipeline**

```
FASTA
  -> annotation pipeline (from selected TOOL)
  -> strain × pathway matrix CSV
  -> optional PCA transform
  -> merged into modeling dataset / path: embedding file
  -> used in features.py
```

**Output files:**
```
data/external/
  kegg_pathway_presence.csv      # 25 x ~500 binary
  kegg_pathway_pca.csv           # 25 x 10-20 numeric PCs
  kegg_pca_loadings.csv          # for transforming new strains later
```

**In the model `config`**
```
features:
  genome_size_bp: "numeric"
  gc_content_pct: "numeric"
  strain_pathways: "path:data/external/kegg_pathway_pca.csv"
  media_name: "OH"
  atmosphere: "OH"
  temperature: "numeric"
```

## Tool (TBD)

Frameworks that could be used for this pipeline:

- `DRAM` (**preferred**): Takes as input FASTA seqs. Predicts proteins (using Prodigal), assigns functional annotations, and automatically processes pathway completeness (using function `DRAM.py distill`) to output the "distillate" and "liquor" files, which strain × pathway matrices.
   - [Repo](https://github.com/WrightonLabCSU/DRAM)
- `eggNOG-mapper:` Takes as input FASTA seqs. Predicts proteins (using Prodigal), and assigns functional annotations. 
   - Cons: would need to write a custom script to map those KOs to KEGG pathways, calculate the completeness thresholds (e.g., 27/30 genes present), and build the final matrix.
   - [Repo](https://github.com/eggnogdb/eggnog-mapper)

