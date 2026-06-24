# Plan: Adding strain metadata to dataset pipeline

## To do's

- Create the script `get_strain_metadata.py` that will use NCBI Entrez (via Biopython's Entrez module). Query by strain name or ATCC number, get the metadata, then download the FASTA and GBK:
    **From NCBI metadata (retrievable alongside the FASTA/GBK):**
    - `ncbi_refseq` 
    - `ncbi_assembly_accession` — e.g. `GCF_000123456.1`, the stable identifier for the assembly
    - `ncbi_taxon_id` — NCBI taxonomy ID, useful for phylogenetic grouping
    - `assembly_level` — `complete`, `chromosome`, `scaffold`, or `contig` — critical for ML because incomplete assemblies can introduce noise
    - `genome_size_bp` — total base pairs; helps flag outliers and normalize features
    - `gc_content_pct` — a basic but informative genomic feature
    - `sequence_hash` — a SHA-256 hash of the raw FASTA sequence, so you can detect duplicates or changes if you re-pull data
    - `date_retrieved` — NCBI records change; reproducibility requires knowing when you pulled
- **To do: phylogenetic distance, GO terms, CLEAN pipeline**
- The strating point of the script will be will be the `data/metadata/align_strains.csv`. Normalize names to match the strain names in the `_layout.csv` (this can be manual)
- The output will be a file called `strains_metadata.csv`. See schema below
- The whole process should run with a one liner like the other scripts (`python scripts/get_strain_metadata.py`)

## `/data` folder schema

```
project/
├── data/
    ├── external     ← to do!
        ├── strains_metadata.csv   
        ├── fasta/
        │   ├── GCF_000123456.1.fna
        │   └── ...
        └── gbk/
            ├── GCF_000123456.1.gbk
            └── ...
```


## CSV schema

| Column | Type | Source |
|---|---|---|
| `strain` | str | your data |
| `atcc_number` | str | your data |
| `ncbi_refseq` | str | NCBI (RefSeq accession) |
| `ncbi_assembly_accession` | str | NCBI |
| `ncbi_taxon_id` | int | NCBI |
| `assembly_level` | str | NCBI |
| `genome_size_bp` | int | NCBI / FASTA |
| `gc_content_pct` | float | computed from FASTA |
| `num_contigs` | int | NCBI / FASTA |
| `n50` | int | NCBI |
| `sequence_hash` | str | computed |
| `date_retrieved` | date | computed |


