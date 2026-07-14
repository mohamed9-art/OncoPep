# Portability notes

The recovered historical OncoPep notebooks preserve the code used in the
original analysis. Some cells retain absolute paths from the original Linux
workspace.

These paths are not credentials and do not affect the archived outputs, but
they must be adapted when rerunning the notebooks on another system.

The complete inventory is provided in:

- `metadata/absolute_path_inventory.txt`

For portable execution, map the original inputs to the corresponding
repository directories:

- processed peptide inputs: `data/processed/`
- frozen partitions: `data/splits/`
- conditioning metadata: `data/conditioning/`
- benchmark outputs: `results/benchmarking/`
- audit outputs: `results/audit/`
- figure source data: `results/source_data/`

The notebook code is preserved for provenance. Paths should be configured
before a full clean-environment rerun.
