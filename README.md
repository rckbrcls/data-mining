# Disque 100 Research Data

This repository contains the first-semester 2026 Disque 100 dataset and a PostgreSQL migration that loads it into one analysis-ready table.

## Prerequisites

- PostgreSQL server and the `psql` command-line tool.
- At least 8 GiB of free disk space.
- `data/disque100-primeiro-semestre-2026.csv` in the repository root.
- An existing target database. The default database name is `disque100`.

## Installation

The notebook is self-contained. The project metadata is used only to install its dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install .
```

## Experiment

The repository has one analysis notebook:

- `notebooks/01_damicore_abuse_categories.ipynb`

It aggregates every female-victim report from January through June 2026 into normalized context profiles for an explicit violence-related subset of the source taxonomy. Profiles include demographic, reporting-process, relationship, setting, and monthly dimensions; DAMICORE checks whether support or document size still dominates NCD, then creates a colored tree plus two NetworkX views. There is no report sampling or later classification step.

## Artifact Privacy Split

Artifacts are stored under `artifacts/damicore_abuse_categories_2026_h1/`:

- `work/` contains the category corpus, category mapping, and DAMICORE run. It is ignored by Git.
- `results/` contains one aggregate CSV, two GraphML files, the cluster tree, and two NetworkX figures. No report hash or individual report is exported.

## Execution

The notebook is intentionally stored without outputs. Open it and run its cells from top to bottom. It defaults to `postgresql:///disque100`; set `DISQUE100_DATABASE_URL` only to override that connection.

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/01_damicore_abuse_categories.ipynb
```

The command is documented for reproducibility; it was not run during this refactor. Existing files under `results/` remain stale until the notebook is executed again.

## Run the migration

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
- `source_hash` is a validated 64-character source identifier, but it is not unique per row.
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
