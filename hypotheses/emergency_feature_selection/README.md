# Emergency feature selection

Independent experiment using `public.disque100_reports`. It does not reuse the
data products or code of other hypotheses.

| Location | Purpose |
| --- | --- |
| `emergency_feature_selection.ipynb` | Didactic, executable study with data audit, NCD matrix, DAMICORE tree, probability blocks, model metrics, and comparisons. |
| `pipeline/config.py` | Scope, target labels, fields, and search budget. |
| `pipeline/data.py` | Report-level aggregation and audit. |
| `pipeline/model.py` | Probabilistic model and quality metrics. |
| `pipeline/search.py` | Random, independent EDA, and DAMICORE-guided searches. |
| `pipeline/experiment.py` | Comparison, final evaluation, and aggregate outputs. |
| `tests/` | Synthetic data and search checks. |
| `artifacts/` | Generated aggregate results, per-cycle DAMICORE diagnostics, and figures; ignored by Git. |

Open the notebook from the project environment and run its cells in order. It
requires access to the existing PostgreSQL table; CSV input is not a fallback.
The notebook's invented three-field example only explains the two meanings of
binary values. All computed results come from the database. No report IDs or
individual predictions are exported. DAMICORE runs once per seed and generation;
each run saves the raw membership, labeled NCD matrix, tree, and diagnostic
summary alongside its input decision-series files.
