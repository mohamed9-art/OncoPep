# OncoPep

OncoPep is a leakage-aware, property-conditioned generative framework for anticancer peptide-oriented sequence-space exploration and computational candidate prioritization.

The framework was developed using a curated peptide corpus compiled from dbAMP 3.0/dbAMPseq, APD3, CancerPPD2, and DCTPep/cancer-therapy peptide resources. OncoPep combines similarity-aware train-validation-test partitioning, train-derived descriptor conditioning, frozen sequence preprocessing, conditional sequence generation, memorization-risk auditing, descriptor-support analysis, and multi-component candidate prioritization.

This repository provides the processed data, figure-level source-data tables, configuration files, model code, benchmark outputs, candidate-prioritization files, and documentation required to reproduce the computational analyses reported in the associated OncoPep manuscript.

---

## Releases and archival

The previous public OncoPep release, v1.0.1, is archived in Zenodo under the version-specific DOI `10.5281/zenodo.21311557`. This record is retained as an immutable archive of the earlier release.

The corrected reproducibility-complete package is designated release v1.0.2. Its version-specific Zenodo DOI will be added after the GitHub release has been published and archived by Zenodo.

## Important scientific note

The final OncoPep sequences are computationally prioritized candidates. They are not experimentally validated anticancer peptides.

No claim is made here regarding experimental anticancer activity, selectivity, toxicity, serum stability, protease resistance, biological mechanism, receptor binding, therapeutic efficacy, or clinical utility. Experimental validation is required before any biological or therapeutic conclusion can be made.

---

## Repository structure

```text
OncoPep/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
├── data/
├── docs/
├── models/
├── results/
│   ├── audit/
│   ├── benchmarking/
│   ├── contextual/
│   ├── generation/
│   ├── prioritization/
│   ├── source_data/
│   │   └── all data/
│   └── tables/
└── src/
```

---

## Folder description

- `configs/`: configuration files for preprocessing, model training, sequence generation, benchmarking, and prioritization.
- `data/`: processed peptide tables, descriptor matrices, split assignments, condition labels, and tokenization metadata.
- `docs/`: reproducibility notes, workflow documentation, and manuscript-related supporting files.
- `models/`: trained OncoPep and baseline model checkpoints, where available.
- `results/audit/`: memorization-risk, descriptor-support, and quality-control audit outputs.
- `results/benchmarking/`: generator benchmarking summaries and comparison outputs.
- `results/contextual/`: contextual similarity and sequence-composition analysis outputs.
- `results/generation/`: generated peptide pools and generation-stage outputs.
- `results/prioritization/`: candidate-prioritization outputs and robustness-analysis files.
- `results/source_data/all data/`: figure-level source-data CSV files for main and supplementary figures.
- `results/tables/`: manuscript-related result tables and supporting tabular outputs.
- `src/`: source code for preprocessing, descriptor calculation, leakage-aware splitting, model training, sequence generation, benchmarking, auditing, prioritization, and figure generation.

---

## Source data

Figure-level source-data files are provided in:

```text
results/source_data/all data/
```

This folder contains the numerical values used to generate the main and supplementary figures, including source data for:

```text
Figure 1
Figure 2
Figure 3
Figure 4
Figure 5
Supplementary Figures S1–S14
```

These source-data files include corpus composition, descriptor distributions, split audits, condition-support summaries, generator benchmark metrics, generated-sequence quality-control summaries, candidate-prioritization stages, final-candidate descriptor-support audits, nearest-neighbor similarity summaries, prioritization-robustness outputs, and sequence-composition enrichment analyses.

The source-data files should be treated as the authoritative record for figure-level numerical values.

---

## Figures

Final main and supplementary figure files are submitted separately through the journal submission system.

High-resolution figures and complete step-wise figure-generation outputs are retained outside the lightweight GitHub repository when file-size constraints apply. The figure-level source-data files in this repository provide the numerical basis for the reported figures.

---

## Data sources

The OncoPep corpus was assembled from publicly available peptide resources, including dbAMP 3.0/dbAMPseq, APD3, CancerPPD2, and DCTPep/cancer-therapy peptide resources.

Raw third-party database records are not redistributed in this repository unless permitted by the original data-source terms. Users should obtain original database records from the corresponding source databases. This repository provides processed and derived analysis files required to reproduce the reported OncoPep analyses.

---

## Reproducibility workflow

The computational workflow consists of the following main stages:

1. Peptide corpus curation and sequence standardization.
2. Descriptor calculation and source-composition audit.
3. Similarity-aware train-validation-test partitioning.
4. Train-derived descriptor binning and condition-label construction.
5. Frozen tokenization and sequence preprocessing.
6. Conditional VAE training and baseline generator training.
7. Conditional sequence generation.
8. Generator benchmarking and memorization-risk audit.
9. Candidate quality control and descriptor-plausibility filtering.
10. Multi-component candidate prioritization.
11. Final-candidate descriptor-support and similarity-tail audit.
12. Figure source-data export and figure generation.

---

## Model checkpoints

The `models/` folder may include the following checkpoint files:

- `cvae_conditional_seed*.pt`: OncoPep conditional VAE checkpoints.
- `gru_conditional_seed*.pt`: conditional GRU baseline checkpoints.
- `gru_unconditional_seed*.pt`: unconditional GRU baseline checkpoints.
- `vae_unconditional_seed*.pt`: unconditional VAE baseline checkpoints.

Model checkpoints are provided for reproducibility where storage constraints permit. When checkpoints are not included, the repository provides the configuration files and training code required to regenerate them.

---

## Installation

Create the computational environment using either `environment.yml` or `requirements.txt`.

Using conda:

```bash
conda env create -f environment.yml
conda activate oncopep
```

Using pip:

```bash
pip install -r requirements.txt
```

---

## Data and code availability

The OncoPep repository is available at:

`https://github.com/mohamed9-art/OncoPep`

Release v1.0.2 contains the author-generated source code and configuration files, processed and derived peptide data, frozen sequence-hash partition assignments, conditioning metadata, generated-sequence outputs, generator-benchmarking and similarity-audit results, candidate-prioritization outputs, figure-level source data, computational-environment files, run manifests, checksums, and reproducibility documentation.

Author-generated software is distributed under the MIT License. Raw third-party peptide records are not redistributed where redistribution is restricted by the terms of the original databases.

## Final candidate panel

The final OncoPep candidate panel is provided in the results and prioritization outputs. Candidate-level files include descriptor values, nearest-neighbor similarity summaries, descriptor-support labels, prioritization scores, and final-panel metadata.

The final candidate panel is intended to guide future experimental testing. It should not be interpreted as a set of experimentally validated anticancer peptides.

---

## Citation

Please cite the associated OncoPep manuscript and the versioned Zenodo archive. The version-specific DOI for release v1.0.2 will be added after Zenodo archives the published GitHub release.

## License

The code is released under the license provided in the `LICENSE` file.

Processed data and source-data files are provided for academic reproducibility of the OncoPep manuscript, subject to the terms of the original peptide data sources where applicable.

---

## Contact

Mohamed Aldaw
Center of Bioinformatics
College of Life Sciences
Northwest A&F University
GitHub: https://github.com/mohamed9-art
