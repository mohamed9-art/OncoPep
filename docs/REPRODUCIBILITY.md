# OncoPep reproducibility package

## Model implementation

The recovered training and generation implementation is available as:

- `notebooks/OncoPep_training_generation.ipynb`
- `src/training/oncopep_training_generation_notebook_export.py`

The notebook is authoritative. Outputs and execution counters were removed
from the public copy; code and Markdown cells were preserved.

## Data partitions

The frozen partition assignments are available at:

- `data/splits/train_validation_test_split_assignments_sha256.csv`

Assignments use SHA-256 hashes of standardized peptide sequences, allowing
partition verification without automatically redistributing every original
third-party record.

## Conditioning metadata

Condition definitions, mappings, support tables, normalization parameters,
and preprocessing contracts are in:

- `data/conditioning/`

## Benchmarking and audits

Generator benchmarking results are in:

- `results/benchmarking/`

Novelty, filtering, and similarity audits are in:

- `results/audit/`

## Figure source data

Source-data tables for the main and supplementary figures are in:

- `results/source_data/`

## Runtime and execution metadata

Environment information is in:

- `environment/`

Stage-level run and artifact manifests are indexed in:

- `metadata/run_manifest_index.tsv`
