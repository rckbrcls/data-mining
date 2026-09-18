# Data Mining Research - DAMICORE and Disque 100

This repository contains the work for the current Data Mining course. It combines the Disque 100 datasets from 2020 through the first semester of 2026, a canonical PostgreSQL loader, and a DAMICORE experiment over complex public data.

The course project is intentionally separate from the future master's research. The professional learning roadmap and the provisional conflict-research architecture live in the public [`conflict-research`](https://github.com/rckbrcls/conflict-research) repository, under [`research/`](https://github.com/rckbrcls/conflict-research/tree/main/research).

## Course material

- [`AED Proposito Geral.pdf`](docs/AED%20Proposito%20Geral.pdf) - course material about general-purpose Estimation of Distribution Algorithms and complex-data mining.

## Prerequisites

- PostgreSQL server and the `psql` command-line tool.
- At least 30 GiB of free disk space for the historical import.
- All `data/disque100-*.csv` files in the repository root.
- An existing target database. The default database name is `disque100`.

## Installation

The hypothesis notebooks are self-contained. The project metadata is used only to install their dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install .
```

## Research hypotheses

The repository organizes analysis by hypothesis so that a shared Disque 100 dataset can
support multiple research questions without mixing their derived corpora or results.

- [`hypotheses/violence_against_women/`](hypotheses/violence_against_women/) - DAMICORE
  experiments over contextual profiles related to violence against women.

The raw CSVs, database loader, and migrations remain shared. Each hypothesis owns its
notebooks, generated inputs, experiment results, figures, and interpretation notes.

The current violence-against-women hypothesis uses three experiments and two frozen
category-set versions:

1. full case profiles with observed prevalence;
2. balanced case profiles across five deterministic replicas;
3. normalized category profiles.

The `v1_14` baseline preserves the original 14-category aggregation. The default `v2_30`
version uses 30 specialized source-taxonomy paths. Both versions use the same 20 contextual
dimensions and write their work and result artifacts under separate version directories.

All three experiments use the same 20 contextual dimensions and standardized figure
outputs. The balanced experiment preserves its five replica outputs and adds a median
distance matrix, a deterministic representative tree, and stability summaries.

## Artifact privacy split

Hypothesis-specific artifacts live under each hypothesis directory. For the current
analysis, `hypotheses/violence_against_women/artifacts/versions/<category_set_version>/work/`
contains generated corpora, metadata, and raw DAMICORE runs and is ignored by Git. Aggregate
tables, figures, trees, and comparison summaries live under the corresponding versioned
`results/` directory and do not export `source_hash`.

The DAMICORE tree is rendered with `toytree` for interactive HTML exploration and
publication-quality SVG output. `toytree` consumes the Newick tree already produced by
DAMICORE; it does not recalculate NCD distances, topology, or clusters.

## Execution

Open the hypothesis notebooks in order. The artifact notebook is the only database-facing
stage. The Nitro PostgreSQL container is bound to its loopback interface; create an SSH tunnel and point the notebook to it:

```bash
ssh -N -L 5433:127.0.0.1:5432 nitro
DISQUE100_DATABASE_URL="postgresql://postgres@127.0.0.1:5433/disque100" jupyter lab
```

```bash
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/00_create_artifacts.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/01_experiment_case_full.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/02_experiment_case_balanced.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/03_experiment_normalized_categories.ipynb
python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/04_compare_experiments.ipynb
```

The notebooks default to `v2_30`. To rebuild the baseline, set the version for the complete
five-notebook sequence:

```bash
DAMICORE_CATEGORY_SET_VERSION=v1_14 python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/00_create_artifacts.ipynb
DAMICORE_CATEGORY_SET_VERSION=v1_14 python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/01_experiment_case_full.ipynb
DAMICORE_CATEGORY_SET_VERSION=v1_14 python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/02_experiment_case_balanced.ipynb
DAMICORE_CATEGORY_SET_VERSION=v1_14 python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/03_experiment_normalized_categories.ipynb
DAMICORE_CATEGORY_SET_VERSION=v1_14 python -m nbconvert --execute --to notebook --inplace hypotheses/violence_against_women/notebooks/04_compare_experiments.ipynb
```

After both versions have complete result packs, run
`hypotheses/violence_against_women/notebooks/05_compare_category_sets.ipynb`. It reads both
versioned result packs and does not compare distance matrices whose category universes differ.

The full execution regenerates the input artifacts, all three experiment result packs,
and the final comparison. Existing generated outputs are not treated as authoritative
until this sequence completes again.

## Load the historical database

The Nitro loader reads the CSVs in place and streams canonical rows into the persistent PostgreSQL container without creating another local copy:

```bash
ssh nitro 'python3 /srv/storage/disque100/load_full_disque100.py /srv/storage/disque100/data'
```

The first-semester 2020 file uses a legacy 34-column layout and 32-character hashes; the loader maps it to the canonical schema. The other files use the 62-column layout, with minor header spelling differences handled by position.

## Run the 2026-only migration

From the repository root:

```bash
./scripts/migrate.sh
```

To use another database or connection string:

```bash
DISQUE100_DATABASE_URL="postgresql://user@host:5432/disque100" ./scripts/migrate.sh
```

The runner is forward-only and safe to rerun. It exits without importing again when the final table already satisfies every expected invariant. If the table exists with an incomplete schema or unexpected data, it fails without modifying the database.

The migration runs in one transaction. A failed first execution rolls back the replacement and preserves the existing `public.disque100_raw` table.

## Final schema

The final database contains one user table: `public.disque100_reports`.

- `id` is a generated `bigint` primary key.
- `source_hash` is a validated 32- or 64-character source identifier, but it is not unique per row.
- `registered_at` is a millisecond-precision timestamp without an inferred timezone.
- `victim_count` is a positive `smallint`.
- Remaining source fields use English `snake_case` identifiers and preserve the original Portuguese categorical values.
- Compound geographical values such as `BR | BRASIL` and `3304557 | RIO DE JANEIRO` remain unchanged.
- Literal `NULL` values and empty or whitespace-only text fields become SQL `NULL`.

## Row grain

One `source_hash` can occur in multiple denormalized rows, including rows with different violations or vulnerable groups. Use `count(distinct source_hash)` when the analysis intends to count source reports; use `count(*)` only when counting dataset rows.

```sql
select
    state,
    count(distinct source_hash) as report_count
from public.disque100_reports
where state is not null
group by state
order by report_count desc;
```

```sql
select
    vulnerable_group,
    count(distinct source_hash) as report_count
from public.disque100_reports
where vulnerable_group is not null
group by vulnerable_group
order by report_count desc;
```
