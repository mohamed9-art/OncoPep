# OncoPep v1.0.3 reproducibility release

This release provides the final reproducibility package underlying the
computational analyses reported in the OncoPep manuscript. It supersedes the
earlier v1.0.1 archive and the unpublished v1.0.2 release-candidate tag.

## Included materials

- recovered model-training and sequence-generation implementation;
- code-cell export of the original implementation notebook;
- figure-generation and candidate-prioritization code;
- model and analysis configuration files;
- processed and derived peptide tables;
- complete-corpus and model-ready sequence-hash partition assignments;
- documentation of the 22 sequences excluded between partitioning and
  model-readiness filtering;
- conditioning schemas, mappings, normalization parameters, and preprocessing
  contracts;
- generated-sequence outputs;
- generator-benchmarking results;
- exact-novelty, filtering, and similarity-audit results;
- candidate-prioritization and final-candidate outputs;
- source-data tables for Main Figures 1–5 and Supplementary Figures S1–S14;
- computational-environment and runtime metadata;
- stage-level run and artifact manifests;
- repository manifest and SHA-256 checksums;
- reproducibility and portability documentation.

## Data partitions

The standardized corpus contains 51,005 unique peptide sequences. After
conditioning and model-readiness filtering, 50,983 sequences were retained:

- training: 24,485;
- validation: 13,248;
- test: 13,250.

The release includes sequence-hash assignments for both the complete corpus
and the model-development population.

## Software license

Author-generated software is distributed under the MIT License.

## Third-party data

Original peptide records were obtained from dbAMP 3.0/dbAMPseq, APD3,
CancerPPD2, and DCTPep. Raw records are not redistributed where redistribution
is restricted by the applicable source-resource terms.

## Scientific scope

The final OncoPep sequences are computationally prioritized candidates and
have not undergone experimental anticancer-activity validation. Experimental
validation is required before biological or therapeutic conclusions can be
drawn.

## Archival

This GitHub release is intended for version-specific archival in Zenodo. The
corresponding DOI will be added to the manuscript and repository metadata after
the Zenodo record has been created and verified.
