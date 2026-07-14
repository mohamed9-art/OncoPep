# OncoPep partition files

OncoPep distinguishes the complete standardized corpus from the population
used for model development and evaluation.

## Complete standardized corpus

`corpus_level_split_assignments_sha256.csv` contains the similarity-aware
partition assignments for all 51,005 standardized unique sequences:

- train: 24,485
- validation: 13,260
- test: 13,260

These assignments were established before conditioning and model-readiness
filtering.

## Model-development population

`train_validation_test_split_assignments_sha256.csv` contains the frozen
assignments for the 50,983 sequences used in the conditioned/model-ready
pipeline and reported in the manuscript:

- train: 24,485
- validation: 13,248
- test: 13,250

The Step-3 conditioned and Step-4 model-ready populations contain the same
50,983 sequence hashes, and their split assignments agree with the original
Step-2 assignments.

## Exclusions

`modeling_exclusions_from_corpus.tsv` lists the 22 sequence hashes that were
present after corpus standardization and Step-2 partitioning but absent from
the conditioned/model-ready population:

- train: 0
- validation: 12
- test: 10

The file records the observed stage transition without assigning an
unsupported biological or legal reason for exclusion.

## Privacy and third-party data handling

Partition files use SHA-256 hashes of standardized peptide sequences. They
allow users to verify assignments after independently obtaining the original
records from the relevant source databases.
