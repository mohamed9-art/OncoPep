# OncoPep

OncoPep is a leakage-aware, property-conditioned generative framework for anticancer peptide-oriented sequence-space exploration and computational candidate prioritization.

The framework was developed using a curated peptide corpus compiled from dbAMP 3.0/dbAMPseq, APD3, CancerPPD2, and DCTPep/cancer-therapy peptide resources. OncoPep combines similarity-aware train-validation-test partitioning, train-derived descriptor conditioning, frozen sequence preprocessing, conditional sequence generation, memorization-risk auditing, descriptor-support analysis, and multi-component candidate prioritization.

This repository provides the processed data, figure-level source-data tables, configuration files, model code, benchmark outputs, candidate-prioritization files, and documentation required to reproduce the computational analyses reported in the associated OncoPep manuscript.

---

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
