# Disque 100 Research Data

This repository contains the first-semester 2026 Disque 100 dataset and a PostgreSQL migration that loads it into one analysis-ready table.

## Prerequisites

- PostgreSQL server and the `psql` command-line tool.
- At least 8 GiB of free disk space.
- `data/disque100-primeiro-semestre-2026.csv` in the repository root.
- An existing target database. The default database name is `disque100`.

## Installation

This repository uses an editable Python package for shared infrastructure.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Notebook Order and Contracts

The analysis is divided into three notebooks. Execute them in order because they communicate through explicit artifacts:

1. `notebooks/01_damicore_structural_baseline.ipynb` creates the deterministic female-victim cohort, DAMICORE run, membership, tree, and structural summaries.
2. `notebooks/02_semantic_abuse_graphs.ipynb` reads the membership and creates observed cluster profiles and static semantic graphs.
3. `notebooks/03_temporal_abuse_graphs.ipynb` reads the aggregate category support and creates the six monthly graph snapshots and final interpretation.

## Artifact Privacy Split

Artifacts are stored under `artifacts/<experiment_id>`:

- `work/` contains private intermediate data, corpus files, memberships, and raw DAMICORE outputs. It is ignored by Git.
- `results/` contains aggregate, privacy-safe tables, GraphML files, and figures. Public text artifacts cannot contain `source_hash` or 64-character hexadecimal identifiers.

## Execution

Set the local database URL and execute the notebooks in place:

```bash
export DISQUE100_DATABASE_URL="postgresql:///disque100"
jupyter nbconvert --execute --to notebook --inplace notebooks/01_damicore_structural_baseline.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/02_semantic_abuse_graphs.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/03_temporal_abuse_graphs.ipynb
```

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
