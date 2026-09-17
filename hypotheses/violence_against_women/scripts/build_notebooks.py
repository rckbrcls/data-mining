from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = ROOT / "hypotheses" / "violence_against_women" / "notebooks"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def write_notebook(filename: str, cells: list):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbf.write(notebook, NOTEBOOK_DIR / filename)


COMMON_SETUP = r'''
from pathlib import Path
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "pyproject.toml").exists():
    raise RuntimeError("Run this notebook from the repository or one of its subdirectories.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypotheses.violence_against_women.scripts.experiment_common import (
    BIAS_WARNING_THRESHOLD,
    CASE_BALANCED_WORK_ROOT,
    CASE_FULL_WORK_ROOT,
    COMMON_WORK_ROOT,
    END_DATE,
    HYPOTHESIS_ROOT,
    MIN_REPORT_COUNT,
    NORMALIZED_WORK_ROOT,
    RESULTS_ROOT,
    START_DATE,
    VICTIM_GENDER,
    category_order_from_manifest,
    load_artifact_manifest,
    load_category_map,
    plot_bias_diagnostics,
    plot_cluster_stability,
    plot_ncd_heatmap,
    plot_support,
    render_tree_artifacts,
    same_cluster_pairs,
    write_json,
)

load_dotenv(PROJECT_ROOT / ".env")
'''


ARTIFACT_NOTEBOOK_CELLS = [
    markdown(
        """
        # DAMICORE artifacts: violence against women

        This notebook is the only database-facing preparation stage for this hypothesis.
        It creates the three corpora and the metadata contract consumed by the experiment
        notebooks. It does not execute DAMICORE or interpret clusters.

        **Scope:** distinct reports with a female victim registered from January 2020 through
        June 2026, restricted to the exploratory violence-related source taxonomy.
        """
    ),
    code(
        COMMON_SETUP
        + r'''
import json
import shutil
from math import log2

import numpy as np
import psycopg

from hypotheses.violence_against_women.scripts.damicore_case_experiment import (
    CASE_COLUMNS,
    CASE_FIELDS,
    build_case_corpora,
    compare_combination_distributions,
    load_case_records,
)
from hypotheses.violence_against_women.scripts.experiment_common import (
    ARTIFACT_ROOT,
    ARTIFACT_SCHEMA_VERSION,
    CASE_SEEDS,
    LOG_RATIO_LIMIT,
    RESULTS_ROOT,
    SMOOTHING_ALPHA,
    WORK_ROOT,
    ensure_artifact_directories,
    write_json,
)

DATABASE_URL = os.getenv(
    "DISQUE100_DATABASE_URL",
    "postgresql://postgres@127.0.0.1:5433/disque100",
)

# This hypothesis owns its generated workspace. Re-running this cell starts its work
# artifacts from zero without touching shared raw data or the database.
if ARTIFACT_ROOT.exists():
    shutil.rmtree(ARTIFACT_ROOT)
ensure_artifact_directories()
'''
    ),
    markdown(
        """
        ## 1. Query scope and shared context counts

        `source_hash` is used only inside PostgreSQL for distinct-report counts. It is not
        written to the derived artifacts.
        """
    ),
    code(
        r'''
CATEGORY_SQL = """
nullif(concat_ws(' > ',
    nullif(trim(split_part(violation, '>', 1)), ''),
    nullif(trim(split_part(violation, '>', 2)), '')
), '')
"""

CONTEXT_FIELDS = [
    ("faixa_etaria", "victim_age_group"),
    ("relacao_vitima_suspeito", "victim_suspect_relationship"),
    ("ambiente", "violation_setting"),
    ("mes", "registered_at"),
    ("inicio_violacoes", "violation_start_period"),
    ("canal_atendimento", "service_channel"),
    ("tipo_denunciante", "reporter_type"),
    ("frequencia", "frequency"),
    ("situacao_emergencia", "emergency_status"),
    ("motivacao", "motivation"),
    ("grupo_vulneravel", "vulnerable_group"),
    ("deficiencia_vitima", "victim_disability"),
    ("raca_cor_vitima", "victim_race_color"),
    ("escolaridade_vitima", "victim_education_level"),
    ("renda_vitima", "victim_income_range"),
    ("etnia_vitima", "victim_ethnicity"),
    ("faixa_etaria_suspeito", "suspect_age_group"),
    ("genero_suspeito", "suspect_gender"),
    ("escolaridade_suspeito", "suspect_education_level"),
    ("natureza_juridica_suspeito", "suspect_legal_nature"),
]
assert {label for label, _ in CONTEXT_FIELDS} == {label for _, label in CASE_FIELDS}
assert len(CONTEXT_FIELDS) == len(CASE_COLUMNS) == 20

value_selects = ["source_hash", f"{CATEGORY_SQL} AS category"]
for label, source_column in CONTEXT_FIELDS:
    if label == "mes":
        expression = "to_char(date_trunc('month', registered_at), 'YYYY-MM')"
    else:
        expression = f"coalesce(nullif(trim({source_column}), ''), 'DESCONHECIDO')"
    value_selects.append(f"{expression} AS {label}")

context_selects = [
    "SELECT category, source_hash, 'denuncia'::text AS dimension, 'TODAS'::text AS value\nFROM report_values WHERE category IS NOT NULL"
]
for label, _ in CONTEXT_FIELDS:
    context_selects.append(
        f"SELECT category, source_hash, '{label}' AS dimension, {label} AS value\n"
        "FROM report_values WHERE category IS NOT NULL"
    )

coverage_query = f"""
WITH reports AS (
    SELECT source_hash,
           bool_or(violation IS NOT NULL) AS has_violation,
           bool_or(({CATEGORY_SQL}) IS NOT NULL) AS has_category
    FROM public.disque100_reports
    WHERE registered_at >= %s AND registered_at < %s
      AND victim_gender = %s
    GROUP BY source_hash
)
SELECT count(*) AS female_reports,
       count(*) FILTER (WHERE has_category) AS reports_with_category,
       count(*) FILTER (WHERE NOT has_violation) AS reports_without_violation
FROM reports
"""

context_query = f"""
WITH report_values AS (
    SELECT {', '.join(value_selects)}
    FROM public.disque100_reports
    WHERE registered_at >= %s AND registered_at < %s
      AND victim_gender = %s
), contexts AS (
    {' UNION ALL '.join(context_selects)}
)
SELECT category, dimension, value, count(DISTINCT source_hash) AS report_count
FROM contexts
GROUP BY category, dimension, value
ORDER BY category, dimension, value
"""

parameters = (START_DATE, END_DATE, VICTIM_GENDER)
with psycopg.connect(DATABASE_URL) as connection:
    with connection.cursor() as cursor:
        cursor.execute(coverage_query, parameters)
        coverage = pd.DataFrame(
            cursor.fetchall(),
            columns=[column.name for column in cursor.description],
        )
        cursor.execute(context_query, parameters)
        context_counts = pd.DataFrame(
            cursor.fetchall(),
            columns=[column.name for column in cursor.description],
        )

assert set(context_counts.columns) == {"category", "dimension", "value", "report_count"}
assert set(context_counts["dimension"].unique()) == {"denuncia", *{label for label, _ in CASE_FIELDS}}
coverage.to_csv(COMMON_WORK_ROOT / "coverage.csv", index=False)
context_counts.to_csv(COMMON_WORK_ROOT / "context-counts.csv", index=False)
display(coverage)
display(context_counts.head())
print(f"Context rows: {len(context_counts):,}")
'''
    ),
    markdown(
        """
        ## 2. Select eligible categories and create the normalized corpus

        The selection rule follows the source taxonomy and the minimum support threshold used
        by the original experiment. Category order is fixed here and reused by every notebook.
        """
    ),
    code(
        r'''
category_summary = context_counts.loc[
    context_counts["dimension"] == "denuncia",
    ["category", "report_count"],
].rename(columns={"report_count": "support"})

scope_mask = (
    category_summary["category"].str.startswith(
        ("INTEGRIDADE", "VIDA", "VIOLÊNCIA INSTITUCIONAL")
    )
    | category_summary["category"].eq("LIBERDADE > SEXUAL")
    | category_summary["category"].str.contains(
        "VIOLÊNCIA POLITÍCA DE GÊNERO E CONTRA AS MULHERES",
        regex=False,
    )
)
category_support = category_summary.copy()
category_support["status"] = "included"
category_support.loc[~scope_mask, "status"] = "out_of_scope"
category_support.loc[
    scope_mask & category_support["support"].lt(MIN_REPORT_COUNT),
    "status",
] = "below_minimum_support"

included_support = category_support.loc[
    category_support["status"] == "included"
].copy()
included_support = included_support.sort_values("category").reset_index(drop=True)
category_order = included_support["category"].tolist()
assert category_order

profile_counts = context_counts.loc[
    context_counts["category"].isin(category_order)
    & context_counts["dimension"].ne("denuncia")
].copy()
vocabulary = (
    profile_counts[["dimension", "value"]]
    .drop_duplicates()
    .sort_values(["dimension", "value"])
)
expected_dimensions = {label for _, label in CASE_FIELDS}
assert set(vocabulary["dimension"]) == expected_dimensions

normalized_profiles = (
    included_support[["category", "support"]]
    .merge(vocabulary, how="cross")
    .merge(profile_counts, on=["category", "dimension", "value"], how="left")
    .sort_values(["category", "dimension", "value"])
)
normalized_profiles["report_count"] = normalized_profiles["report_count"].fillna(0).astype(int)
global_counts = (
    profile_counts.groupby(["dimension", "value"], as_index=False)["report_count"]
    .sum()
    .rename(columns={"report_count": "global_count"})
)
global_counts["global_prevalence"] = (
    global_counts["global_count"]
    / global_counts.groupby("dimension")["global_count"].transform("sum")
)
normalized_profiles = normalized_profiles.merge(
    global_counts[["dimension", "value", "global_prevalence"]],
    on=["dimension", "value"],
    how="left",
    validate="many_to_one",
)
normalized_profiles["smoothed_prevalence"] = (
    normalized_profiles["report_count"]
    + SMOOTHING_ALPHA * normalized_profiles["global_prevalence"]
) / (normalized_profiles["support"] + SMOOTHING_ALPHA)
normalized_profiles["relative_bps"] = (
    10_000
    * (
        normalized_profiles["smoothed_prevalence"].div(
            normalized_profiles["global_prevalence"]
        ).map(log2).clip(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
        + LOG_RATIO_LIMIT
    )
    / (2 * LOG_RATIO_LIMIT)
).round().astype(int)

category_rows = []
for number, category in enumerate(category_order, start=1):
    label = f"category-{number:03d}.txt"
    rows = normalized_profiles.loc[normalized_profiles["category"] == category]
    lines = [
        f"{row.dimension}|{row.value}|{row.relative_bps:05d}"
        for row in rows.itertuples(index=False)
    ]
    corpus_path = NORMALIZED_WORK_ROOT / "corpus" / label
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    category_rows.append(
        {"label": label, "category": category, "support": int(rows["support"].iloc[0])}
    )

category_map = pd.DataFrame(category_rows)
category_map.to_csv(COMMON_WORK_ROOT / "category-map.csv", index=False)
included_support.to_csv(COMMON_WORK_ROOT / "category-support.csv", index=False)
category_support.to_csv(COMMON_WORK_ROOT / "category-scope.csv", index=False)
context_counts.loc[
    context_counts["category"].isin(category_order)
].to_csv(COMMON_WORK_ROOT / "eligible-context-counts.csv", index=False)

corpus_sizes = pd.Series(
    {
        path.name: path.stat().st_size
        for path in (NORMALIZED_WORK_ROOT / "corpus").glob("*.txt")
    },
    name="bytes",
)
assert corpus_sizes.nunique() == 1
display(category_map)
print(f"Included categories: {len(category_order)}")
print(f"Normalized corpus bytes per category: {int(corpus_sizes.iloc[0]):,}")
'''
    ),
    markdown(
        """
        ## 3. Create the case-full and case-balanced corpora

        The case grain is `source_hash + category` in memory. Canonical case documents contain
        only the 20 contextual dimensions, so report identifiers are not exported.
        """
    ),
    code(
        r'''
case_records = load_case_records(
    database_url=DATABASE_URL,
    start_date=START_DATE,
    end_date=END_DATE,
    victim_gender=VICTIM_GENDER,
    included_categories=category_order,
    category_sql=CATEGORY_SQL,
)
case_support = (
    case_records.groupby("category", as_index=False)["source_hash"]
    .nunique()
    .rename(columns={"source_hash": "case_count"})
)
expected_support = included_support[["category", "support"]].rename(
    columns={"support": "expected_case_count"}
)
case_support_check = expected_support.merge(
    case_support, on="category", validate="one_to_one"
).set_index("category").loc[category_order]
assert case_support_check["expected_case_count"].equals(
    case_support_check["case_count"]
)

case_category_map, case_combination_counts, balanced_sample_size = build_case_corpora(
    case_records=case_records,
    category_map=category_map,
    work_dir=WORK_ROOT,
    seeds=CASE_SEEDS,
    metadata_dir=COMMON_WORK_ROOT,
)
case_combination_comparison = compare_combination_distributions(
    case_combination_counts,
    COMMON_WORK_ROOT / "case-combination-comparison.csv",
)
case_category_map.to_csv(COMMON_WORK_ROOT / "case-category-map.csv", index=False)
case_combination_counts.to_csv(
    COMMON_WORK_ROOT / "case-combination-counts.csv", index=False
)

assert not case_records.duplicated(["source_hash", "category"]).any()
canonical_keys = case_records["canonical_record"].map(lambda value: set(json.loads(value)))
assert canonical_keys.map(lambda keys: not {"source_hash", "category", "id"}.intersection(keys)).all()
assert canonical_keys.map(lambda keys: keys == expected_dimensions).all()
assert case_category_map["case_count"].gt(0).all()
assert set(case_category_map["category"]) == set(category_order)
assert case_category_map.loc[
    case_category_map["regime"] == "case-balanced", "case_count"
].eq(balanced_sample_size).all()

print(f"Case records in memory: {len(case_records):,}")
print(f"Balanced sample size per category and replica: {balanced_sample_size:,}")
print(f"Balanced replicas: {len(CASE_SEEDS)}")
display(case_category_map.head())

# Keep identifiers and large in-memory case records out of later notebook state.
del case_records, case_combination_counts, case_combination_comparison
'''
    ),
    markdown(
        """
        ## 4. Write the artifact manifest

        The manifest is the compatibility contract for the three experiment notebooks.
        """
    ),
    code(
        r'''
manifest = {
    "schema_version": ARTIFACT_SCHEMA_VERSION,
    "hypothesis": "violence_against_women",
    "source_table": "public.disque100_reports",
    "start_date": START_DATE,
    "end_date": END_DATE,
    "victim_gender": VICTIM_GENDER,
    "minimum_report_count": MIN_REPORT_COUNT,
    "smoothing_alpha": SMOOTHING_ALPHA,
    "log_ratio_limit": LOG_RATIO_LIMIT,
    "case_seeds": CASE_SEEDS,
    "dimensions": [label for _, label in CASE_FIELDS],
    "category_order": category_order,
    "category_count": len(category_order),
    "balanced_sample_size": balanced_sample_size,
    "paths": {
        "category_map": "common/category-map.csv",
        "category_support": "common/category-support.csv",
        "case_category_map": "common/case-category-map.csv",
        "normalized_corpus": "normalized_categories/corpus",
        "case_full_corpus": "case_full/corpus",
        "case_balanced_root": "case_balanced",
    },
}
write_json(COMMON_WORK_ROOT / "artifact-manifest.json", manifest)
write_json(HYPOTHESIS_ROOT / "manifest.json", {
    "hypothesis": "violence_against_women",
    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    "notebooks": [
        "00_create_artifacts.ipynb",
        "01_experiment_normalized_categories.ipynb",
        "02_experiment_case_full.ipynb",
        "03_experiment_case_balanced.ipynb",
        "04_compare_experiments.ipynb",
    ],
})
print("Artifact manifest written.")
'''
    ),
]


EXPERIMENT_SETUP = r'''
from pathlib import Path
from itertools import combinations
import sys

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "pyproject.toml").exists():
    raise RuntimeError("Run this notebook from the repository or one of its subdirectories.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypotheses.violence_against_women.scripts.experiment_common import (
    BIAS_WARNING_THRESHOLD,
    CASE_BALANCED_WORK_ROOT,
    CASE_FULL_WORK_ROOT,
    CASE_SEEDS,
    COMMON_WORK_ROOT,
    NCD_COLOR_VMAX,
    NORMALIZED_WORK_ROOT,
    RESULTS_ROOT,
    case_execution,
    category_order_from_manifest,
    ensure_artifact_directories,
    load_artifact_manifest,
    load_category_map,
    plot_bias_diagnostics,
    plot_cluster_stability,
    plot_ncd_heatmap,
    plot_support,
    render_tree_artifacts,
    run_damicore_experiment,
    same_cluster_pairs,
    write_json,
    write_common_result_artifacts,
)

ensure_artifact_directories()
manifest = load_artifact_manifest()
category_order = category_order_from_manifest(manifest)
category_map = load_category_map(COMMON_WORK_ROOT / "category-map.csv")
category_support = pd.read_csv(COMMON_WORK_ROOT / "category-support.csv")
category_support = category_support.rename(columns={"support": "support"})
assert category_map["category"].tolist() == category_order
assert set(category_support["category"]) == set(category_order)
'''


def normalized_notebook():
    return [
        markdown(
            """
            # Experiment 1: normalized category profiles

            One DAMICORE object represents one eligible source category. The input documents
            are fixed-width relative-context profiles created by `00_create_artifacts`.
            """
        ),
        code(EXPERIMENT_SETUP),
        markdown("## Run DAMICORE and record the common result contract"),
        code(
            r'''
result = run_damicore_experiment(
    experiment_name="normalized_categories",
    corpus_dir=NORMALIZED_WORK_ROOT / "corpus",
    raw_runs_dir=NORMALIZED_WORK_ROOT / "runs",
    execution=case_execution(),
)
assert result["status"] == "completed", result["preview"]

bytes_by_category = {
    row.category: int((NORMALIZED_WORK_ROOT / "corpus" / row.label).stat().st_size)
    for row in category_map.itertuples(index=False)
}
output_dir = RESULTS_ROOT / "normalized_categories"
normalized_result = write_common_result_artifacts(
    result=result,
    category_map=category_map,
    category_order=category_order,
    support=category_support,
    bytes_by_category=bytes_by_category,
    output_dir=output_dir,
    support_column="support",
    support_label="Distinct reports (log scale)",
    title_prefix="Normalized categories",
)
display(normalized_result["membership_by_category"])
display(normalized_result["distance"].round(3))
'''
        ),
        markdown(
            """
            ## Interpretation boundary

            These clusters describe similarity among normalized contextual profiles. They are
            exploratory and are not legal categories, causal mechanisms, or individual-level
            predictions.
            """
        ),
    ]


def case_full_notebook():
    return [
        markdown(
            """
            # Experiment 2: full case corpus

            One DAMICORE object represents one eligible category, while each document preserves
            the canonical contextual combinations observed in all distinct `source_hash + category`
            cases. Category prevalence is retained.
            """
        ),
        code(EXPERIMENT_SETUP),
        markdown("## Run DAMICORE and record the common result contract"),
        code(
            r'''
case_category_map = pd.read_csv(COMMON_WORK_ROOT / "case-category-map.csv")
case_full_support = (
    case_category_map.loc[case_category_map["regime"] == "case-full", ["category", "case_count"]]
    .rename(columns={"case_count": "support"})
    .sort_values("category")
)
bytes_by_category = (
    case_category_map.loc[case_category_map["regime"] == "case-full"]
    .set_index("category")["bytes"]
    .astype(int)
    .to_dict()
)
assert case_full_support["category"].tolist() == category_order

result = run_damicore_experiment(
    experiment_name="case_full",
    corpus_dir=CASE_FULL_WORK_ROOT / "corpus",
    raw_runs_dir=CASE_FULL_WORK_ROOT / "runs",
    execution=case_execution(),
)
assert result["status"] == "completed", result["preview"]
case_full_result = write_common_result_artifacts(
    result=result,
    category_map=category_map,
    category_order=category_order,
    support=case_full_support,
    bytes_by_category=bytes_by_category,
    output_dir=RESULTS_ROOT / "case_full",
    support_column="support",
    support_label="Distinct cases (log scale)",
    title_prefix="Case-full",
)
display(case_full_result["membership_by_category"])
display(case_full_result["distance"].round(3))
'''
        ),
        markdown(
            """
            ## Interpretation boundary

            `case-full` combines contextual similarity with observed category prevalence and
            corpus volume. It is not directly comparable to an equal-support regime without the
            balanced experiment.
            """
        ),
    ]


def case_balanced_notebook():
    return [
        markdown(
            """
            # Experiment 3: balanced case corpus

            Five deterministic replicas sample the same number of cases per category. The common
            result pack uses the median NCD matrix and the replicate closest to that median as the
            representative tree; all replica outputs remain available below the result directory.
            """
        ),
        code(EXPERIMENT_SETUP),
        markdown("## Run the five balanced replicas"),
        code(
            r'''
case_category_map = pd.read_csv(COMMON_WORK_ROOT / "case-category-map.csv")
balanced_root = RESULTS_ROOT / "case_balanced"
replicate_results = {}
replicate_tables = []

for index, _seed in enumerate(CASE_SEEDS, start=1):
    replicate = f"replicate-{index:03d}"
    replicate_map = case_category_map.loc[
        (case_category_map["regime"] == "case-balanced")
        & (case_category_map["replicate"] == replicate)
    ].sort_values("category")
    support = replicate_map[["category", "case_count"]].rename(
        columns={"case_count": "support"}
    )
    bytes_by_category = replicate_map.set_index("category")["bytes"].astype(int).to_dict()
    result = run_damicore_experiment(
        experiment_name=f"case_balanced_{replicate}",
        corpus_dir=CASE_BALANCED_WORK_ROOT / replicate,
        raw_runs_dir=CASE_BALANCED_WORK_ROOT / "runs",
        execution=case_execution(),
    )
    assert result["status"] == "completed", result["preview"]
    replicate_results[replicate] = result
    packaged = write_common_result_artifacts(
        result=result,
        category_map=category_map,
        category_order=category_order,
        support=support,
        bytes_by_category=bytes_by_category,
        output_dir=balanced_root / "replicates" / replicate,
        support_column="support",
        support_label="Distinct cases per category (log scale)",
        title_prefix=f"Case-balanced {replicate}",
    )
    replicate_tables.append(packaged)
'''
        ),
        markdown("## Build the representative balanced result pack"),
        code(
            r'''
distance_matrices = {
    replicate: packaged["distance"]
    for replicate, packaged in zip(
        replicate_results,
        replicate_tables,
    )
}
matrix_stack = np.stack([matrix.to_numpy() for matrix in distance_matrices.values()])
median_matrix = pd.DataFrame(
    np.median(matrix_stack, axis=0),
    index=category_order,
    columns=category_order,
)
median_vector = np.array([
    median_matrix.loc[left, right]
    for left, right in combinations(category_order, 2)
])
medoid_rows = []
for replicate, matrix in distance_matrices.items():
    vector = np.array([
        matrix.loc[left, right]
        for left, right in combinations(category_order, 2)
    ])
    medoid_rows.append({
        "replicate": replicate,
        "distance_to_median": float(np.linalg.norm(vector - median_vector)),
    })
medoid_table = pd.DataFrame(medoid_rows).sort_values(
    ["distance_to_median", "replicate"]
)
representative_replicate = medoid_table.iloc[0]["replicate"]
representative = replicate_tables[
    list(replicate_results).index(representative_replicate)
]

balanced_output = RESULTS_ROOT / "case_balanced"
balanced_output.mkdir(parents=True, exist_ok=True)
median_matrix.to_csv(balanced_output / "distance-matrix.csv")
representative["membership_by_category"].to_csv(
    balanced_output / "clusters.csv", index=False
)
(balanced_output / "tree.nwk").write_text(
    representative["tree_newick"], encoding="utf-8"
)
support = case_category_map.loc[
    case_category_map["regime"] == "case-balanced",
    ["category", "case_count"],
].drop_duplicates("category").rename(columns={"case_count": "support"})
support = support.sort_values("category")
support.to_csv(balanced_output / "support.csv", index=False)
plot_support(
    support,
    "support",
    "Distinct cases per category (log scale)",
    balanced_output / "support.png",
    "Case-balanced: category support",
)
plot_ncd_heatmap(
    median_matrix,
    balanced_output / "ncd-heatmap.png",
    "Case-balanced: median NCD distances",
)
replicate_diagnostics = pd.concat(
    [pd.read_csv(balanced_root / "replicates" / replicate / "bias-diagnostics.csv").assign(replicate=replicate)
     for replicate in distance_matrices],
    ignore_index=True,
)
median_diagnostics = (
    replicate_diagnostics.groupby("diagnostic", as_index=False)["spearman_correlation"]
    .median()
)
median_diagnostics.to_csv(balanced_output / "bias-diagnostics.csv", index=False)
plot_bias_diagnostics(
    median_diagnostics,
    balanced_output / "bias-diagnostics.png",
    "Case-balanced: median NCD bias diagnostics",
)
render_tree_artifacts(
    representative["tree_newick"],
    representative["membership_by_category"],
    category_map,
    balanced_output,
    f"Case-balanced representative ({representative_replicate})",
)

medoid_table.to_csv(balanced_output / "replicate-medoid-selection.csv", index=False)
'''
        ),
        markdown("## Replica stability"),
        code(
            r'''
all_category_pairs = set(combinations(category_order, 2))
balanced_memberships = {
    replicate: packaged["membership_by_category"]
    for replicate, packaged in zip(replicate_results, replicate_tables)
}
stability_rows = []
for left_name, right_name in combinations(balanced_memberships, 2):
    left_pairs = same_cluster_pairs(balanced_memberships[left_name], category_order)
    right_pairs = same_cluster_pairs(balanced_memberships[right_name], category_order)
    stability_rows.append({
        "left": left_name,
        "right": right_name,
        "pairwise_cluster_agreement": len(all_category_pairs - (left_pairs ^ right_pairs))
        / len(all_category_pairs),
    })
stability = pd.DataFrame(stability_rows)
stability.to_csv(balanced_output / "replicate-stability.csv", index=False)
plot_cluster_stability(stability, balanced_output / "replicate-stability.png")
write_json(
    balanced_output / "run-summary.json",
    {
        "experiment": "case_balanced",
        "replicas": list(replicate_results),
        "representative_replicate": representative_replicate,
        "status": "completed",
    },
)
display(medoid_table)
display(stability)
'''
        ),
    ]


def comparison_notebook():
    return [
        markdown(
            """
            # Comparison: the three DAMICORE experiments

            This notebook reads the standardized result packs only. It does not query the
            database and does not execute DAMICORE.
            """
        ),
        code(
            COMMON_SETUP
            + r'''
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from hypotheses.violence_against_women.scripts.experiment_common import (
    NCD_COLOR_VMAX,
    compare_distance_matrices,
    named_distance_matrix,
    plot_distance_agreement,
    plot_ncd_heatmap,
    plot_cluster_stability,
    same_cluster_pairs,
    wrapped_label,
    write_json,
)

manifest = load_artifact_manifest()
category_order = category_order_from_manifest(manifest)
category_map = load_category_map(COMMON_WORK_ROOT / "category-map.csv")
experiment_names = ["normalized_categories", "case_full", "case_balanced"]
result_dirs = {name: RESULTS_ROOT / name for name in experiment_names}
for name, directory in result_dirs.items():
    if not (directory / "distance-matrix.csv").exists():
        raise FileNotFoundError(f"Missing standardized result for {name}: {directory}")
'''
        ),
        markdown("## Load matrices, memberships and support"),
        code(
            r'''
distance_matrices = {
    name: pd.read_csv(
        directory / "distance-matrix.csv", index_col=0
    ).loc[category_order, category_order]
    for name, directory in result_dirs.items()
}
memberships = {
    name: pd.read_csv(directory / "clusters.csv")
    for name, directory in result_dirs.items()
}
supports = {
    name: pd.read_csv(directory / "support.csv")
    for name, directory in result_dirs.items()
}
diagnostics = {
    name: pd.read_csv(directory / "bias-diagnostics.csv")
    for name, directory in result_dirs.items()
}
assert all(np.allclose(np.diag(matrix.to_numpy()), 0.0) for matrix in distance_matrices.values())
'''
        ),
        markdown("## Distance agreement and common-scale heatmaps"),
        code(
            r'''
comparison_dir = RESULTS_ROOT / "comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)
distance_comparison = compare_distance_matrices(distance_matrices, category_order)
distance_comparison.to_csv(comparison_dir / "distance-agreement.csv", index=False)

agreement_names = list(distance_matrices)
agreement_values = np.eye(len(agreement_names))
for row in distance_comparison.itertuples(index=False):
    left = agreement_names.index(row.left)
    right = agreement_names.index(row.right)
    agreement_values[left, right] = row.spearman_correlation
    agreement_values[right, left] = row.spearman_correlation
agreement = pd.DataFrame(agreement_values, index=agreement_names, columns=agreement_names)
agreement.to_csv(comparison_dir / "distance-agreement-matrix.csv")
plot_distance_agreement(agreement, comparison_dir / "distance-agreement.png")

figure, axes = plt.subplots(1, 3, figsize=(21, 8), constrained_layout=True)
for axis, (name, matrix) in zip(axes, distance_matrices.items()):
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=NCD_COLOR_VMAX)
    labels = [wrapped_label(category, 20) for category in category_order]
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=6)
    axis.set_title(name)
figure.colorbar(image, ax=axes.tolist(), label="NCD", shrink=0.82)
figure.savefig(comparison_dir / "ncd-heatmaps-comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(figure)
'''
        ),
        markdown("## Label-invariant cluster agreement"),
        code(
            r'''
cluster_rows = []
pair_sets = {
    name: same_cluster_pairs(memberships[name], category_order)
    for name in experiment_names
}
all_pairs = set(combinations(category_order, 2))
for left_name, right_name in combinations(experiment_names, 2):
    left_pairs = pair_sets[left_name]
    right_pairs = pair_sets[right_name]
    cluster_rows.append({
        "left": left_name,
        "right": right_name,
        "pairwise_cluster_agreement": len(all_pairs - (left_pairs ^ right_pairs))
        / len(all_pairs),
    })
cluster_agreement = pd.DataFrame(cluster_rows)
cluster_agreement.to_csv(comparison_dir / "cluster-agreement.csv", index=False)
display(distance_comparison)
display(cluster_agreement)
'''
        ),
        markdown("## Support and size diagnostics"),
        code(
            r'''
diagnostic_rows = []
for name, table in diagnostics.items():
    for row in table.itertuples(index=False):
        diagnostic_rows.append({
            "experiment": name,
            "diagnostic": row.diagnostic,
            "spearman_correlation": row.spearman_correlation,
            "absolute_correlation": abs(row.spearman_correlation),
        })
diagnostics_comparison = pd.DataFrame(diagnostic_rows)
diagnostics_comparison.to_csv(
    comparison_dir / "bias-diagnostics-comparison.csv", index=False
)
figure, axis = plt.subplots(figsize=(12, 6))
for name, table in diagnostics.items():
    axis.plot(
        table["diagnostic"],
        table["spearman_correlation"].abs(),
        marker="o",
        label=name,
    )
axis.axhline(
    BIAS_WARNING_THRESHOLD,
    color="#C44536",
    linestyle="--",
    linewidth=1.2,
    label=f"screening threshold ({BIAS_WARNING_THRESHOLD:.2f})",
)
axis.set_ylim(0, 1)
axis.set_ylabel("Absolute Spearman correlation")
axis.set_title("Support and document-size diagnostics", loc="left", weight="bold")
axis.tick_params(axis="x", rotation=18)
axis.grid(axis="y", color="#E2E7E9", linewidth=0.7)
axis.set_axisbelow(True)
axis.legend(frameon=False)
figure.tight_layout()
figure.savefig(
    comparison_dir / "bias-diagnostics-comparison.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(figure)

balanced_stability_path = RESULTS_ROOT / "case_balanced" / "replicate-stability.csv"
if balanced_stability_path.exists():
    balanced_stability = pd.read_csv(balanced_stability_path)
    balanced_stability.to_csv(
        comparison_dir / "balanced-replica-stability.csv", index=False
    )
else:
    balanced_stability = pd.DataFrame()
'''
        ),
        markdown("## Common support comparison"),
        code(
            r'''
figure, axis = plt.subplots(figsize=(14, 7))
positions = np.arange(len(category_order))
width = 0.25
for index, name in enumerate(experiment_names):
    table = supports[name].set_index("category").loc[category_order]
    axis.bar(
        positions + (index - 1) * width,
        table["support"],
        width,
        label=name,
    )
axis.set_yscale("log")
axis.set_xticks(positions)
axis.set_xticklabels([wrapped_label(category, 22) for category in category_order], rotation=90, fontsize=7)
axis.set_ylabel("Support (log scale)")
axis.set_title("Support across the three experiments", loc="left", weight="bold")
axis.legend(frameon=False)
axis.grid(axis="y", color="#E2E7E9", linewidth=0.7)
axis.set_axisbelow(True)
figure.tight_layout()
figure.savefig(comparison_dir / "support-comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(figure)

summary = distance_comparison.copy()
summary["comparison_type"] = "distance"
cluster_summary = cluster_agreement.copy()
cluster_summary["comparison_type"] = "cluster"
pd.concat([summary, cluster_summary], ignore_index=True, sort=False).to_csv(
    comparison_dir / "comparison-summary.csv", index=False
)
write_json(
    comparison_dir / "comparison-manifest.json",
    {
        "experiments": experiment_names,
        "category_order": category_order,
        "ncd_color_range": [0.0, NCD_COLOR_VMAX],
        "source_artifact_manifest": str(COMMON_WORK_ROOT / "artifact-manifest.json"),
    },
)
'''
        ),
    ]


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook("00_create_artifacts.ipynb", ARTIFACT_NOTEBOOK_CELLS)
    write_notebook("01_experiment_normalized_categories.ipynb", normalized_notebook())
    write_notebook("02_experiment_case_full.ipynb", case_full_notebook())
    write_notebook("03_experiment_case_balanced.ipynb", case_balanced_notebook())
    write_notebook("04_compare_experiments.ipynb", comparison_notebook())


if __name__ == "__main__":
    main()
