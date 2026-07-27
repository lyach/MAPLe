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


## Analysis

4 questions to answer: 
1. Is the pathway matrix trustworthy? (sanity check)
2. Do pathways separate strains biologically?
3. Do pathways relate to growth? (MaxOD/growth class)
4. How many PCs are best for modeling?

Place plots under `src/modeling/plots.py`, visualize in a notebook called `04_pathway_matrix_HA.ipynb`:

### 1. Pathway count vs assembly quality scatter
**Is the pathway matrix trustworthy?**

| Aspect | Detail |
| --- | --- |
| Plot | Scatter: x = `num_contigs` (log scale), y = total pathways detected per strain. Color by `assembly_level` (complete/scaffold/contig). Label each point with strain name. |
| What to look for | If fragmented assemblies (right side of x-axis) have **fewer** pathways, annotation artifacts are present. A strain with 42 contigs but similar genome size to a complete genome should have a similar pathway count - if it's much lower, missing genes caused false absences. |
| How to compute | Sum each row of the binary matrix. Merge with `strains_metadata.csv` on strain. |

### 2. PCA biplot
**Do pathways separate strains biologically?**

| Aspect | Detail |
| --- | --- |
| Plot | Scatter of strains in PC1 vs PC2 space. Label each point with strain name. Color by phylum. Overlay the top 10 pathway loadings as arrows to show which pathways drive the separation. |
| What to look for | Related strains should cluster. If PC1 separates phylums (distance in the x axis), the matrix captures biological signal. If instead PC1 separates "complete" from "fragmented" assemblies, the matrix is encoding technical artifacts. |
| How to compute | `sklearn.decomposition.PCA` on the binary matrix. For the biplot arrows, use `pca.components_[0:2, :]` and select the top-10 pathways by loading magnitude. |

### 3. Pathway count vs mean max_od scatter
**Do pathways relate to growth?**

| Aspect | Detail |
| --- | --- |
| Plot | Scatter: x = total pathways per strain, y = mean `max_od` across all media (from the curves CSV). Color by phylum. Add Pearson r score. |
| What to look for | Strains with more pathways might grow better on average across diverse media. A positive correlation supports the hypothesis that pathway features carry signal for growth. A flat line suggests pathways and growth are orthogonal. |
| How to compute | Group curves CSV by `strain_name`, compute mean `max_od`. Merge with pathway row sums. |

### 4.1. Scree plot (cumulative explained variance)
**How many PCs are best for modeling?**

| Aspect | Detail |
| --- | --- |
| Plot | Line plot: x = number of PCA components (1 to 25), y = cumulative explained variance ratio. Horizontal reference lines at 80%, 90%, 95%. |
| What to look for | The elbow where additional components add diminishing returns. It tells which components will capture most variance and are worth keeping. |
| How to compute | `PCA(n_components=25).fit(X_pathway)`, then `np.cumsum(pca.explained_variance_ratio_)`. |

### 4.2. LOSO performance vs number of PCA components
**How many PCs are best for modeling?**

| Aspect | Detail |
| --- | --- |
| Plot | Line plot with error bars: x = number of PCA components used as strain features (1, 2, 3, 5, 7, 10, 15, 20), y = LOSO RMSE for `max_od` regression (mean +/- std across folds). Overlay a horizontal reference line for the LOSO performance with baseline genome features (eg `gc_content_pct`, `genome_size_bp`, etc). |
| What to look for | Performance should **increase then plateau** (or dip from overfitting). The plateau point is your answer. If pathway PCs never beat baselines, the annotation doesn't add value for LOSO. It's possible to see other overfitting happening, e.g. beating at 7 components but degrade at 15. |
| How to compute | For each value of k: fit `PCA(n_components=k)` on the **24 training strains** inside each LOSO fold (leakage-safe — do NOT fit PCA on all 25 strains then split). Transform the held-out strain. Concatenate PC features with media + condition features. Run XGBoost. Record metric. IMPORTANT: **PCA must be fit inside the fold**, otherwise there's leakage. |
| Repeat for | `growth_class` (but change RMSE to AUROC) to see if the optimal k differs by task. |

