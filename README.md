# Disque 100 Research Data

This repository contains Disque 100 datasets from 2020 through the first semester of 2026 and a canonical PostgreSQL loader for one analysis-ready table.

## Prerequisites

- PostgreSQL server and the `psql` command-line tool.
- At least 30 GiB of free disk space for the historical import.
- All `data/disque100-*.csv` files in the repository root.
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

It aggregates every female-victim report from January 2020 through June 2026 into normalized context profiles for an explicit violence-related subset of the source taxonomy. Profiles include demographic, reporting-process, relationship, setting, and monthly dimensions; DAMICORE checks whether support or document size still dominates NCD, then creates a colored tree plus two NetworkX views. The first-semester 2020 file has a legacy violation format and is preserved in PostgreSQL but excluded from the comparable hierarchy until it is harmonized. There is no report sampling or later classification step.

## Artifact Privacy Split

Artifacts are stored under `artifacts/damicore_abuse_categories_2020_2026/`:

- `work/` contains the category corpus, category mapping, and DAMICORE run. It is ignored by Git.
- `results/` contains one aggregate CSV, two GraphML files, the cluster tree, and two NetworkX figures. No report hash or individual report is exported.

## Execution

The notebook is intentionally stored without outputs. Open it and run its cells from top to bottom. The Nitro PostgreSQL container is bound to its loopback interface; create an SSH tunnel and point the notebook to it:

```bash
ssh -N -L 5433:127.0.0.1:5432 nitro
DISQUE100_DATABASE_URL="postgresql://postgres@127.0.0.1:5433/disque100" jupyter lab
```

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/01_damicore_abuse_categories.ipynb
```

The command is documented for reproducibility; it was not run during this refactor. Existing files under `results/` remain stale until the notebook is executed again.

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
