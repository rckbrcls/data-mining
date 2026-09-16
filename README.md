# Data Mining Research - DAMICORE and Disque 100

This repository contains the work for the current Data Mining course. It combines the Disque 100 datasets from 2020 through the first semester of 2026, a canonical PostgreSQL loader, and a DAMICORE experiment over complex public data.

The course project is intentionally separate from the future master's research. The professional learning roadmap and the provisional conflict-research architecture live in the public [`conflict-research`](https://github.com/rckbrcls/conflict-research) repository, under [`research/`](https://github.com/rckbrcls/conflict-research/tree/main/research).

## Study guides

- [`DATA_MINING_STUDY_GUIDE.md`](DATA_MINING_STUDY_GUIDE.md) - general course topics, study order, and the bridge to professional skills.
- [`DAMICORE_STUDY_GUIDE.md`](DAMICORE_STUDY_GUIDE.md) - focused guide for the current DAMICORE experiment.
- [`AED Proposito Geral.pdf`](AED%20Proposito%20Geral.pdf) - course material about general-purpose Estimation of Distribution Algorithms and complex-data mining.

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

It aggregates every female-victim report from January 2020 through June 2026 into normalized context profiles for an explicit violence-related subset of the source taxonomy. The normalized profiles now use 20 dimensions: the original reporting, relationship, setting, monthly, victim, and suspect fields plus violation start period, motivation, victim race/color, victim education, victim income, victim ethnicity, suspect age group, and suspect education. DAMICORE checks whether support or document size still dominates NCD, then produces a distance-based tree and quantitative mining figures. The first-semester 2020 file has a legacy violation format and is preserved in PostgreSQL but excluded from the comparable hierarchy until it is harmonized. There is no report sampling or later classification step.

The same notebook now includes a second, case-level experiment. It builds one canonical line per distinct `source_hash + category` using the same 20 contextual dimensions, then runs DAMICORE over a `case-full` corpus and five uniformly sampled `case-balanced` replicas. `case-full` preserves category prevalence and volume; `case-balanced` gives every included category the same number of reports so the comparison can test whether co-occurring context remains stable when support is equal. `month` represents the registration period, while `violation_start_period` represents the reported onset period; both are retained as distinct dimensions.

## Artifact Privacy Split

Artifacts are stored under `artifacts/damicore_abuse_categories_2020_2026/`:

- `work/` contains the category corpus, category mapping, and DAMICORE run. It is ignored by Git.
- `work/case-corpus/` contains the case-level full corpus, balanced replicas, combination counts, and case-level DAMICORE runs. It is ignored by Git and does not export `source_hash`.
- `results/` contains aggregate CSV summaries and quantitative figures for category support, contextual counts, NCD distances, experiment agreement, replica stability, and the DAMICORE tree. No report hash or individual report is exported.
- The notebook writes `category-support.png`, `category-ncd-heatmap.png`, `category-tree.html`, `category-tree.svg`, `case-distance-agreement.png`, and `case-cluster-stability.png` when the corresponding experiments complete.

The DAMICORE tree is rendered with `toytree` for interactive HTML exploration and publication-quality SVG output. `toytree` consumes the Newick tree already produced by DAMICORE; it does not recalculate NCD distances, topology, or clusters. This research project accepts its GPL-3.0-only dependency for visualization.

## Execution

Open the notebook and run its cells from top to bottom. The Nitro PostgreSQL container is bound to its loopback interface; create an SSH tunnel and point the notebook to it:

```bash
ssh -N -L 5433:127.0.0.1:5432 nitro
DISQUE100_DATABASE_URL="postgresql://postgres@127.0.0.1:5433/disque100" jupyter lab
```

```bash
python -m nbconvert --execute --to notebook --inplace notebooks/01_damicore_abuse_categories.ipynb
```

The command regenerates both the existing normalized experiment and the case-level comparison. Existing files under `results/` remain stale until the notebook is executed again.

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
