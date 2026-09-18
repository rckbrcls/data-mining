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
    artifact_paths,
    load_artifact_manifest,
    load_category_map,
)
load_dotenv(PROJECT_ROOT / ".env")
CATEGORY_SET_VERSION = os.getenv("DAMICORE_CATEGORY_SET_VERSION", "v2_30")
PATHS = artifact_paths(CATEGORY_SET_VERSION)
COMMON_WORK_ROOT = PATHS.common
NORMALIZED_WORK_ROOT = PATHS.normalized
CASE_FULL_WORK_ROOT = PATHS.case_full
CASE_BALANCED_WORK_ROOT = PATHS.case_balanced
RESULTS_ROOT = PATHS.results
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
        r'''
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "pyproject.toml").exists():
    raise RuntimeError("Run this notebook from the repository or one of its subdirectories.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypotheses.violence_against_women.scripts.create_artifacts import (
    configured_database_url,
    initialize_preparation,
    prepare_case_artifacts,
    prepare_normalized_artifacts,
    prepare_source_data,
    write_artifact_manifests,
)
from hypotheses.violence_against_women.scripts.experiment_common import CASE_SEEDS

load_dotenv(PROJECT_ROOT / ".env")
CATEGORY_SET_VERSION = os.getenv("DAMICORE_CATEGORY_SET_VERSION", "v2_30")
ARTIFACT_WORKERS = 2
DATABASE_URL = configured_database_url()
PATHS = initialize_preparation(CATEGORY_SET_VERSION, ARTIFACT_WORKERS)
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
coverage, context_counts, prepared_case_records = prepare_source_data(DATABASE_URL, PATHS)
display(coverage)
display(context_counts.head())
print(f"Context rows: {len(context_counts):,}")
del coverage
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
category_order, included_support, category_map, normalized_corpus_bytes = prepare_normalized_artifacts(
    context_counts, PATHS
)
del context_counts
display(category_map)
print(f"Included categories: {len(category_order)}")
print(f"Normalized corpus bytes per category: {normalized_corpus_bytes:,}")
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
case_category_map, balanced_sample_size, case_count = prepare_case_artifacts(
    prepared_case_records,
    category_order,
    included_support,
    category_map,
    PATHS,
    workers=ARTIFACT_WORKERS,
)
del prepared_case_records, included_support, category_map
print(f"Case records in memory: {case_count:,}")
print(f"Balanced sample size per category and replica: {balanced_sample_size:,}")
print(f"Balanced replicas: {len(CASE_SEEDS)}")
display(case_category_map.head())
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
manifest = write_artifact_manifests(PATHS, category_order, balanced_sample_size)
print("Artifact manifest written.")
'''
    ),
]


EXPERIMENT_SETUP = r'''

from pathlib import Path
import os
import sys

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
    artifact_paths,
    case_execution,
    category_order_from_manifest,
    ensure_artifact_directories,
    load_artifact_manifest,
    load_category_map,
    run_damicore_experiment,
    write_common_result_artifacts,
)

CATEGORY_SET_VERSION = os.getenv("DAMICORE_CATEGORY_SET_VERSION", "v2_30")
PATHS = artifact_paths(CATEGORY_SET_VERSION)
COMMON_WORK_ROOT = PATHS.common
NORMALIZED_WORK_ROOT = PATHS.normalized
CASE_FULL_WORK_ROOT = PATHS.case_full
CASE_BALANCED_WORK_ROOT = PATHS.case_balanced
RESULTS_ROOT = PATHS.results
ensure_artifact_directories(PATHS)
manifest = load_artifact_manifest(
    COMMON_WORK_ROOT / "artifact-manifest.json",
    category_set_version=CATEGORY_SET_VERSION,
)
category_order = category_order_from_manifest(manifest)
category_map = load_category_map(COMMON_WORK_ROOT / "category-map.csv")
category_support = pd.read_csv(COMMON_WORK_ROOT / "category-support.csv")
assert category_map["category"].tolist() == category_order
assert set(category_support["category"]) == set(category_order)
assert manifest["dimension_count"] == 20
'''


BALANCED_SETUP = r'''
from itertools import combinations

import numpy as np

from hypotheses.violence_against_women.scripts.experiment_common import (
    CASE_SEEDS,
    plot_bias_diagnostics,
    plot_cluster_stability,
    plot_ncd_heatmap,
    plot_support,
    render_tree_artifacts,
    same_cluster_pairs,
    write_json,
)
'''


def normalized_notebook():
    return [
        markdown(
            """
            # Experiment 3: normalized category profiles

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
    manifest=manifest,
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
            # Experiment 1: full case corpus

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
    .set_index("category")
    .loc[category_order]
    .reset_index()
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
    manifest=manifest,
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
            # Experiment 2: balanced case corpus

            Five deterministic replicas sample the same number of cases per category. The common
            result pack uses the median NCD matrix and the replicate closest to that median as the
            representative tree; all replica outputs remain available below the result directory.
            """
        ),
        code(EXPERIMENT_SETUP + BALANCED_SETUP),
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
    ].set_index("category").loc[category_order].reset_index()
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
        manifest=manifest,
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
].drop_duplicates("category").set_index("category").loc[category_order].reset_index()
support = support.rename(columns={"case_count": "support"})
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
balanced_cluster_pairs = {
    replicate: same_cluster_pairs(membership, category_order)
    for replicate, membership in balanced_memberships.items()
}
stability_rows = []
for left_name, right_name in combinations(balanced_memberships, 2):
    left_pairs = balanced_cluster_pairs[left_name]
    right_pairs = balanced_cluster_pairs[right_name]
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
        "category_set_version": manifest["category_set_version"],
        "category_definition_hash": manifest["category_definition_hash"],
        "category_count": manifest["category_count"],
        "dimension_count": manifest["dimension_count"],
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

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from hypotheses.violence_against_women.scripts.experiment_common import (
    BIAS_WARNING_THRESHOLD,
    NCD_COLOR_VMAX,
    category_order_from_manifest,
    compare_distance_matrices,
    plot_distance_agreement,
    same_cluster_pairs,
    wrapped_label,
    write_json,
)

manifest = load_artifact_manifest(
    COMMON_WORK_ROOT / "artifact-manifest.json",
    category_set_version=CATEGORY_SET_VERSION,
)
category_order = category_order_from_manifest(manifest)
category_map = load_category_map(COMMON_WORK_ROOT / "category-map.csv")
experiment_names = ["case_full", "case_balanced", "normalized_categories"]
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

heatmap_labels = [wrapped_label(category, 20) for category in category_order]
figure, axes = plt.subplots(1, 3, figsize=(21, 8), constrained_layout=True)
for axis, (name, matrix) in zip(axes, distance_matrices.items()):
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=NCD_COLOR_VMAX)
    labels = heatmap_labels
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
        "category_set_version": manifest["category_set_version"],
        "category_definition_hash": manifest["category_definition_hash"],
        "category_count": manifest["category_count"],
        "dimension_count": manifest["dimension_count"],
        "category_order": category_order,
        "ncd_color_range": [0.0, NCD_COLOR_VMAX],
        "source_artifact_manifest": str(COMMON_WORK_ROOT / "artifact-manifest.json"),
    },
)
'''
        ),
    ]


def category_set_comparison_notebook():
    return [
        markdown(
            """
            # Comparison: category-set versions

            This notebook compares versioned result packs by metadata, support and artifact
            integrity. Trees with different category universes are not compared directly.
            """
        ),
        code(
            COMMON_SETUP
            + r'''
from hypotheses.violence_against_women.scripts.category_sets import CATEGORY_SETS
from hypotheses.violence_against_women.scripts.experiment_common import (
    ARTIFACT_ROOT,
    read_json,
)

VERSION_NAMES = ["v1_14", "v2_30"]
EXPERIMENT_NAMES = ["case_full", "case_balanced", "normalized_categories"]
version_paths = {version: artifact_paths(version) for version in VERSION_NAMES}
manifests = {
    version: load_artifact_manifest(
        paths.common / "artifact-manifest.json",
        category_set_version=version,
    )
    for version, paths in version_paths.items()
}
comparison_dir = ARTIFACT_ROOT / "results" / "category_set_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)
'''
        ),
        markdown("## Verify manifests, result packs and shared dimensions"),
        code(
            r'''
integrity_rows = []
metadata_rows = []
for version, paths in version_paths.items():
    manifest = manifests[version]
    category_order = manifest["category_order"]
    category_map = load_category_map(paths.common / "category-map.csv")
    assert category_map["category"].tolist() == category_order
    assert manifest["dimension_count"] == 20
    assert tuple(category_order) == CATEGORY_SETS[version]

    for experiment in EXPERIMENT_NAMES:
        result_dir = paths.results / experiment
        required = [
            "run-summary.json",
            "support.csv",
            "distance-matrix.csv",
            "clusters.csv",
            "tree.nwk",
        ]
        missing = [name for name in required if not (result_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing result files for {version}/{experiment}: {missing}"
            )
        summary = read_json(result_dir / "run-summary.json")
        support = pd.read_csv(result_dir / "support.csv")
        distances = pd.read_csv(result_dir / "distance-matrix.csv", index_col=0)
        clusters = pd.read_csv(result_dir / "clusters.csv")
        checks = {
            "manifest_version": summary.get("category_set_version") == version,
            "manifest_category_count": summary.get("category_count") == len(category_order),
            "manifest_dimension_count": summary.get("dimension_count") == 20,
            "support_categories": set(support["category"]) == set(category_order),
            "distance_categories": (
                distances.index.tolist() == category_order
                and distances.columns.tolist() == category_order
            ),
            "cluster_categories": set(clusters["category"]) == set(category_order),
            "tree_present": (result_dir / "tree.nwk").exists(),
        }
        if not all(checks.values()):
            raise ValueError(f"Incompatible result pack: {version}/{experiment}")
        integrity_rows.append({
            "category_set_version": version,
            "experiment": experiment,
            **checks,
            "integrity_ok": True,
        })
        metadata_rows.append({
            "category_set_version": version,
            "experiment": experiment,
            "category_count": len(category_order),
            "dimension_count": manifest["dimension_count"],
            "support_min": int(support["support"].min()),
            "support_max": int(support["support"].max()),
            "support_total": int(support["support"].sum()),
        })

integrity = pd.DataFrame(integrity_rows)
metadata = pd.DataFrame(metadata_rows)
integrity.to_csv(comparison_dir / "integrity.csv", index=False)
metadata.to_csv(comparison_dir / "metadata-comparison.csv", index=False)
display(metadata)
display(integrity)
'''
        ),
        markdown("## Exact category overlap"),
        code(
            r'''
shared_categories = sorted(
    set(manifests["v1_14"]["category_order"])
    & set(manifests["v2_30"]["category_order"])
)
pd.DataFrame({"category": shared_categories}).to_csv(
    comparison_dir / "shared-categories.csv", index=False
)
write_json(
    comparison_dir / "comparison-manifest.json",
    {
        "versions": VERSION_NAMES,
        "experiments": EXPERIMENT_NAMES,
        "shared_category_count": len(shared_categories),
        "direct_tree_distance_comparison": False,
    },
)
print(f"Shared exact categories: {len(shared_categories)}")
'''
        ),
    ]


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook("00_create_artifacts.ipynb", ARTIFACT_NOTEBOOK_CELLS)
    write_notebook("01_experiment_case_full.ipynb", case_full_notebook())
    write_notebook("02_experiment_case_balanced.ipynb", case_balanced_notebook())
    write_notebook("03_experiment_normalized_categories.ipynb", normalized_notebook())
    write_notebook("04_compare_experiments.ipynb", comparison_notebook())
    write_notebook("05_compare_category_sets.ipynb", category_set_comparison_notebook())


if __name__ == "__main__":
    main()
