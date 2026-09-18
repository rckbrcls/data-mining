# Violence against women hypothesis

This hypothesis asks whether source categories related to violence against women show
recurring contextual profiles in Disque 100 data.

## Analytical scope

- Source table: `public.disque100_reports`.
- Registration period: January 2020 through June 2026.
- Population filter: `victim_gender = 'FEMININO'`.
- Unit of analysis for the normalized experiment: one eligible source category.
- Unit of analysis for the case experiments: one distinct `source_hash + category` in
  memory; report identifiers are not exported to derived artifacts.
- Context vocabulary: 20 dimensions shared by all three experiments and both category-set versions.
- Minimum category support: 50 distinct reports.
- Normalized-profile smoothing: `alpha = 20`.

The category catalog is versioned:

- `v1_14`: original 14-category aggregate baseline;
- `v2_30`: 30 specialized source-taxonomy paths, used by default.

Artifacts are isolated under `artifacts/versions/<category_set_version>/`. Set
`DAMICORE_CATEGORY_SET_VERSION` before running the notebooks to select a version.

The category scope is an exploratory rule based on the source taxonomy. The resulting
clusters are not legal classifications, causal explanations, individual predictions,
or confirmed social typologies.

## Notebook order

Run the notebooks in this order:

1. `notebooks/00_create_artifacts.ipynb` creates the shared corpora and manifest.
2. `notebooks/01_experiment_case_full.ipynb` runs the full case experiment.
3. `notebooks/02_experiment_case_balanced.ipynb` runs five balanced replicas and selects
   a deterministic representative result.
4. `notebooks/03_experiment_normalized_categories.ipynb` runs the normalized category
   experiment.
5. `notebooks/04_compare_experiments.ipynb` compares the three standardized result packs.
6. `notebooks/05_compare_category_sets.ipynb` checks both versioned result packs and
   reports exact category overlap without comparing incompatible trees directly.

The experiment notebooks do not query PostgreSQL. They consume the artifact manifest
and corpora created by the first notebook.

## Artifacts

Generated inputs and raw DAMICORE runs are under `artifacts/versions/<category_set_version>/work/`
and are ignored by Git. Aggregate tables, figures, trees, and comparison summaries are under
the corresponding versioned `results/` directory and do not export `source_hash`.

Each experiment produces the same primary result contract:

- `support.csv` and `support.png`;
- `distance-matrix.csv` and `ncd-heatmap.png`;
- `clusters.csv`;
- `bias-diagnostics.csv` and `bias-diagnostics.png`;
- `tree.nwk`, `tree.svg`, and `tree.html`;
- `run-summary.json`.

The balanced experiment additionally produces replica-level results, a median distance
matrix, a representative tree, and replica-stability summaries.

The plots use one fixed category order and one shared NCD color range. Cluster colors
are local labels within each tree and must not be compared across experiments as if the
numeric cluster identifiers were stable.

## Execution

The artifact notebook calls the same preparation functions as the standalone script.
Both require the PostgreSQL connection documented in the root README.
For the default Nitro tunnel:

```bash
ssh -N -L 5433:127.0.0.1:5432 nitro
DISQUE100_DATABASE_URL="postgresql://postgres@127.0.0.1:5433/disque100" jupyter lab
```

To prepare artifacts without Jupyter, run from the repository root with the project
environment active:

```bash
python -m hypotheses.violence_against_women.scripts.create_artifacts --workers 2
```

The script reads the repository's `.env` and the current `DISQUE100_DATABASE_URL`.
Its default category set is `DAMICORE_CATEGORY_SET_VERSION` or `v2_30`; pass
`--category-set-version v1_14` to prepare the baseline explicitly.
SQL remains in one connection. Only case serialization and per-category corpus
preparation run in up to two spawned processes. Sampling seeds, document order,
formats, and formulas are preserved. To use the sequential path instead:

```bash
python -m hypotheses.violence_against_women.scripts.create_artifacts --workers 1
```

Choose one preparation command. Both rebuild the selected version's work and results
from scratch, without resuming or reusing old experiments. Then execute notebooks `01`
through `04` with the same category-set version. Notebook `00` is an alternative to the
script and uses `ARTIFACT_WORKERS = 2`; set it to `1` for sequential preparation.

The notebooks can then be executed in order with `nbconvert`:

```bash
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/00_create_artifacts.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/01_experiment_case_full.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/02_experiment_case_balanced.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/03_experiment_normalized_categories.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/04_compare_experiments.ipynb
```

The notebooks default to `v2_30`. To rebuild the baseline, set
`DAMICORE_CATEGORY_SET_VERSION=v1_14` for the same five commands. After both versions have
complete result packs, run `notebooks/05_compare_category_sets.ipynb`.
