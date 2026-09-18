from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from itertools import combinations
from math import log10
from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toytree
import toyplot
from damicore import estimate, run
from damicore.config import ExecutionConfig, ResourceLimits

from .category_sets import category_definition_hash, get_category_set


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HYPOTHESIS_ROOT = PROJECT_ROOT / "hypotheses" / "violence_against_women"
ARTIFACT_ROOT = HYPOTHESIS_ROOT / "artifacts"
VERSIONED_ARTIFACT_ROOT = ARTIFACT_ROOT / "versions"
DEFAULT_CATEGORY_SET_VERSION = "v2_30"

START_DATE = "2020-01-01"
END_DATE = "2026-07-01"
VICTIM_GENDER = "FEMININO"
MIN_REPORT_COUNT = 50
SMOOTHING_ALPHA = 20
LOG_RATIO_LIMIT = 4.0
BIAS_WARNING_THRESHOLD = 0.70
CASE_SEEDS = [101, 202, 303, 404, 505]
ARTIFACT_SCHEMA_VERSION = "1.0"

# This range is shared by all three experiment figures for direct comparison.
NCD_COLOR_VMIN = 0.0
NCD_COLOR_VMAX = 1.05

VISUAL_STYLE = {
    "font_size": 7,
    "text_color": "#263238",
    "grid_color": "#E2E7E9",
    "accent_color": "#2A6F97",
    "accent_light": "#CFE4EE",
}

CLUSTER_COLORS = [
    "#2A6F97",
    "#D95F02",
    "#1B9E77",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
]


@dataclass(frozen=True)
class ArtifactPaths:
    category_set_version: str
    root: Path
    work: Path
    results: Path
    common: Path
    normalized: Path
    case_full: Path
    case_balanced: Path


def artifact_paths(category_set_version: str = DEFAULT_CATEGORY_SET_VERSION) -> ArtifactPaths:
    get_category_set(category_set_version)
    root = VERSIONED_ARTIFACT_ROOT / category_set_version
    work = root / "work"
    return ArtifactPaths(
        category_set_version=category_set_version,
        root=root,
        work=work,
        results=root / "results",
        common=work / "common",
        normalized=work / "normalized_categories",
        case_full=work / "case_full",
        case_balanced=work / "case_balanced",
    )


def wrapped_label(value: object, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))


def ensure_artifact_directories(paths: ArtifactPaths) -> None:
    for directory in (
        paths.common,
        paths.normalized,
        paths.case_full,
        paths.case_balanced,
        paths.results,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_manifest_path(
    category_set_version: str = DEFAULT_CATEGORY_SET_VERSION,
) -> Path:
    return artifact_paths(category_set_version).common / "artifact-manifest.json"


def load_artifact_manifest(
    manifest_path: Path | None = None,
    category_set_version: str | None = None,
) -> dict[str, Any]:
    requested_version = category_set_version or DEFAULT_CATEGORY_SET_VERSION
    manifest = read_json(manifest_path or artifact_manifest_path(requested_version))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema: {manifest.get('schema_version')}"
        )
    selected_version = category_set_version or manifest.get(
        "category_set_version", requested_version
    )
    expected = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "victim_gender": VICTIM_GENDER,
        "smoothing_alpha": SMOOTHING_ALPHA,
        "log_ratio_limit": LOG_RATIO_LIMIT,
        "case_seeds": CASE_SEEDS,
        "category_set_version": selected_version,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Artifact manifest mismatch for {key}: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )
    categories = get_category_set(selected_version)
    if manifest.get("category_count") != len(categories):
        raise ValueError(
            "Artifact manifest category count does not match the versioned catalog."
        )
    if manifest.get("category_order") != list(categories):
        raise ValueError(
            "Artifact manifest category order does not match the versioned catalog."
        )
    if manifest.get("category_definition_hash") != category_definition_hash(categories):
        raise ValueError(
            "Artifact manifest category definition hash does not match the catalog."
        )
    if manifest.get("dimension_count") != 20:
        raise ValueError("Artifact manifest must describe exactly 20 dimensions.")
    return manifest


def load_category_map(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"label", "category"}
    if not required.issubset(table.columns):
        raise ValueError(f"Category map is missing columns: {required - set(table.columns)}")
    if table["category"].duplicated().any() or table["label"].duplicated().any():
        raise ValueError("Category map labels and categories must be unique.")
    return table


def category_order_from_manifest(manifest: dict[str, Any]) -> list[str]:
    categories = manifest.get("category_order")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Artifact manifest must contain a non-empty category_order list.")
    return categories


def case_execution() -> ExecutionConfig:
    return ExecutionConfig(
        workers=1,
        limits=ResourceLimits(max_working_memory_bytes=6 * 1024**3),
    )


def run_damicore_experiment(
    experiment_name: str,
    corpus_dir: Path,
    raw_runs_dir: Path,
    execution: ExecutionConfig | None = None,
) -> dict[str, Any]:
    raw_runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = raw_runs_dir / f"{experiment_name}-run-{run_id}"

    estimate_kwargs: dict[str, Any] = {
        "source_kind": "files",
    }
    if execution is not None:
        estimate_kwargs["execution"] = execution
    preview = estimate(corpus_dir, **estimate_kwargs)
    preview_data = preview.model_dump()
    preview_data["experiment"] = experiment_name
    preview_data["run_id"] = run_id

    if not preview.within_limits:
        preview_data["status"] = "resource_limit"
        return {
            "name": experiment_name,
            "status": "resource_limit",
            "preview": preview_data,
            "corpus_dir": str(corpus_dir),
            "output_dir": str(output_dir),
        }

    run_kwargs: dict[str, Any] = {
        "source_kind": "files",
        "output_dir": output_dir,
    }
    if execution is not None:
        run_kwargs["execution"] = execution
    result = run(corpus_dir, **run_kwargs)
    try:
        membership = result.membership.copy()
        distance_matrix = result.distance_matrix.to_pandas()
        tree_newick = result.tree_newick
    finally:
        result.close()

    preview_data["status"] = "completed"
    return {
        "name": experiment_name,
        "status": "completed",
        "preview": preview_data,
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "membership": membership,
        "distance_matrix": distance_matrix,
        "tree_newick": tree_newick,
    }


def membership_by_category(
    membership: pd.DataFrame,
    category_map: pd.DataFrame,
    experiment_name: str,
) -> pd.DataFrame:
    table = membership[["object_id", "label", "cluster"]].merge(
        category_map[["label", "category"]],
        on="label",
        validate="one_to_one",
    )
    table["experiment"] = experiment_name
    return table[["experiment", "object_id", "label", "category", "cluster"]].sort_values(
        "category"
    )


def named_distance_matrix(
    distance_matrix: pd.DataFrame,
    category_map: pd.DataFrame,
    category_order: list[str],
) -> pd.DataFrame:
    label_to_category = category_map.set_index("label")["category"].to_dict()
    named = distance_matrix.rename(index=label_to_category, columns=label_to_category)
    named = named.loc[category_order, category_order]
    diagonal = np.diag(named.to_numpy())
    if not np.allclose(diagonal, 0.0):
        raise ValueError("Every NCD distance matrix must have a zero diagonal.")
    if float(np.nanmax(named.to_numpy())) > NCD_COLOR_VMAX:
        raise ValueError(
            f"NCD exceeds the shared visualization range of {NCD_COLOR_VMAX}."
        )
    return named


def write_common_result_artifacts(
    result: dict[str, Any],
    category_map: pd.DataFrame,
    category_order: list[str],
    support: pd.DataFrame,
    bytes_by_category: dict[str, int],
    output_dir: Path,
    support_column: str,
    support_label: str,
    title_prefix: str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if result["status"] != "completed":
        summary = {
            **result["preview"],
            **_manifest_run_metadata(manifest),
        }
        write_json(output_dir / "run-summary.json", summary)
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    membership = membership_by_category(
        result["membership"], category_map, result["name"]
    )
    distance = named_distance_matrix(
        result["distance_matrix"], category_map, category_order
    )
    diagnostics = ncd_diagnostics(
        distance,
        support.set_index("category")[support_column].to_dict(),
        bytes_by_category,
        category_order,
    )

    membership.to_csv(output_dir / "clusters.csv", index=False)
    distance.to_csv(output_dir / "distance-matrix.csv")
    support.to_csv(output_dir / "support.csv", index=False)
    diagnostics.to_csv(output_dir / "bias-diagnostics.csv", index=False)
    (output_dir / "tree.nwk").write_text(result["tree_newick"], encoding="utf-8")

    plot_support(
        support,
        support_column,
        support_label,
        output_dir / "support.png",
        f"{title_prefix}: category support",
    )
    plot_ncd_heatmap(
        distance,
        output_dir / "ncd-heatmap.png",
        f"{title_prefix}: NCD distances",
    )
    plot_bias_diagnostics(
        diagnostics,
        output_dir / "bias-diagnostics.png",
        f"{title_prefix}: NCD bias diagnostics",
    )
    render_tree_artifacts(
        result["tree_newick"],
        membership,
        category_map,
        output_dir,
        title_prefix,
    )

    summary = {
        **result["preview"],
        **_manifest_run_metadata(manifest),
        "experiment": result["name"],
        "status": result["status"],
        "corpus_dir": result["corpus_dir"],
        "raw_output_dir": result["output_dir"],
        "result_dir": str(output_dir),
    }
    write_json(output_dir / "run-summary.json", summary)
    return {**result, "membership_by_category": membership, "distance": distance}


def _manifest_run_metadata(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    return {
        "category_set_version": manifest["category_set_version"],
        "category_definition_hash": manifest["category_definition_hash"],
        "category_count": manifest["category_count"],
        "dimension_count": manifest["dimension_count"],
    }


def ncd_diagnostics(
    distance: pd.DataFrame,
    support_by_category: dict[str, int],
    bytes_by_category: dict[str, int],
    category_order: list[str],
) -> pd.DataFrame:
    rows = []
    for left, right in combinations(category_order, 2):
        rows.append(
            {
                "ncd": float(distance.loc[left, right]),
                "support_gap": abs(
                    log10(support_by_category[left])
                    - log10(support_by_category[right])
                ),
                "byte_gap": abs(bytes_by_category[left] - bytes_by_category[right]),
            }
        )
    pairwise = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "diagnostic": "NCD versus log support gap",
                "spearman_correlation": pairwise["ncd"].rank().corr(
                    pairwise["support_gap"].rank()
                ),
            },
            {
                "diagnostic": "NCD versus byte gap",
                "spearman_correlation": pairwise["ncd"].rank().corr(
                    pairwise["byte_gap"].rank()
                ),
            },
        ]
    )


def plot_support(
    support: pd.DataFrame,
    support_column: str,
    support_label: str,
    output_path: Path,
    title: str,
) -> None:
    table = support.sort_values(support_column)
    figure, axis = plt.subplots(figsize=(12, max(6, 0.42 * len(table))))
    axis.barh(
        [wrapped_label(value, 38) for value in table["category"]],
        table[support_column],
        color=VISUAL_STYLE["accent_color"],
    )
    axis.set_xscale("log")
    axis.set_xlabel(support_label)
    axis.set_title(title, loc="left", weight="bold")
    axis.grid(axis="x", color=VISUAL_STYLE["grid_color"], linewidth=0.7)
    axis.set_axisbelow(True)
    maximum = float(table[support_column].max())
    axis.set_xlim(1, maximum * 2)
    for position, value in enumerate(table[support_column]):
        axis.text(
            float(value) * 1.08,
            position,
            f"{int(value):,}",
            va="center",
            fontsize=8,
            color=VISUAL_STYLE["text_color"],
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_ncd_heatmap(
    distance: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    order = list(distance.index)
    figure, axis = plt.subplots(
        figsize=(max(10, 0.68 * len(order)), max(8, 0.58 * len(order)))
    )
    image = axis.imshow(
        distance.to_numpy(),
        cmap="Blues",
        aspect="equal",
        vmin=NCD_COLOR_VMIN,
        vmax=NCD_COLOR_VMAX,
    )
    labels = [wrapped_label(category, 24) for category in order]
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel("Fixed category order")
    axis.set_ylabel("Fixed category order")
    axis.set_title(title, loc="left", weight="bold")
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="NCD", shrink=0.82)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_bias_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    table = diagnostics.copy()
    table["absolute_correlation"] = table["spearman_correlation"].abs()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        table["diagnostic"],
        table["absolute_correlation"],
        color=VISUAL_STYLE["accent_color"],
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
    axis.set_title(title, loc="left", weight="bold")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", color=VISUAL_STYLE["grid_color"], linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_tree_artifacts(
    tree_newick: str,
    membership: pd.DataFrame,
    category_map: pd.DataFrame,
    output_dir: Path,
    title: str,
) -> None:
    tree = toytree.tree(tree_newick)
    by_object_id = membership.set_index("object_id")
    by_label = membership.set_index("label")
    category_by_label = category_map.set_index("label")["category"].to_dict()
    tip_labels = []
    tip_colors = []

    for raw_tip_name in tree.get_tip_labels():
        tip_name = str(raw_tip_name).strip()
        if len(tip_name) >= 2 and tip_name[0] == tip_name[-1] and tip_name[0] in {"'", '"'}:
            tip_name = tip_name[1:-1]
        if tip_name in by_object_id.index:
            row = by_object_id.loc[tip_name]
        elif tip_name in by_label.index:
            row = by_label.loc[tip_name]
        else:
            raise KeyError(f"Unknown DAMICORE tree leaf label: {tip_name}")
        cluster = int(row["cluster"])
        tip_labels.append(category_by_label[row["label"]])
        tip_colors.append(CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)])

    if len(tip_labels) != len(membership):
        raise ValueError("Tree tips and membership rows do not match.")

    canvas, axes, _ = tree.draw(
        tree_style="d",
        layout="r",
        width=1500,
        height=max(480, len(tip_labels) * 42),
        tip_labels=tip_labels,
        tip_labels_colors=tip_colors,
        tip_labels_align=True,
        tip_labels_style={"font-size": "12px", "font-family": "Helvetica"},
        node_labels=False,
        node_mask=False,
        edge_style={"stroke": "#718090", "stroke-width": 1.8, "stroke-opacity": 0.95},
        edge_align_style={
            "stroke": "#5D6C7A",
            "stroke-width": 1.8,
            "stroke-opacity": 0.95,
            "stroke-dasharray": "2,4",
        },
        scale_bar=True,
    )
    canvas.style = {"background-color": "#121820"}
    axes.x.spine.style = {"stroke": "#556474", "stroke-width": 1.2}
    axes.x.ticks.style = {"stroke": "#556474", "stroke-width": 1.0}
    axes.x.ticks.labels.style = {"fill": "#AAB7C4", "font-family": "Helvetica"}
    canvas.text(
        35,
        24,
        f"DAMICORE tree: {title}",
        style={"font-size": "19px", "font-weight": "bold", "fill": "#F4F7FA"},
    )
    canvas.text(
        35,
        46,
        "Contextual profiles shown by category; colors are local cluster labels.",
        style={"font-size": "12px", "fill": "#AAB7C4"},
    )
    legend_markers = [
        (
            f"Cluster {cluster}",
            toyplot.marker.create(
                shape="s",
                size=12,
                mstyle={
                    "fill": CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                    "stroke": CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                },
            ),
        )
        for cluster in sorted(set(membership["cluster"]))
    ]
    legend = canvas.legend(
        legend_markers,
        corner=("top-right", 24, 210, max(72, 28 * len(legend_markers) + 24)),
        label="Local DAMICORE cluster",
    )
    legend.label.style = {"fill": "#F4F7FA", "font-family": "Helvetica"}
    legend.cells.column[1].lstyle = {"fill": "#D4DCE4", "font-family": "Helvetica"}
    toytree.save(canvas, output_dir / "tree.html")
    toytree.save(canvas, output_dir / "tree.svg")


def upper_triangle_values(matrix: pd.DataFrame, category_order: list[str]) -> pd.Series:
    return pd.Series(
        [matrix.loc[left, right] for left, right in combinations(category_order, 2)]
    )


def compare_distance_matrices(
    matrices: dict[str, pd.DataFrame], category_order: list[str]
) -> pd.DataFrame:
    rows = []
    for left_name, right_name in combinations(matrices, 2):
        left = upper_triangle_values(matrices[left_name], category_order)
        right = upper_triangle_values(matrices[right_name], category_order)
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "spearman_correlation": left.rank().corr(right.rank()),
                "mean_absolute_difference": (left - right).abs().mean(),
            }
        )
    return pd.DataFrame(rows)


def same_cluster_pairs(
    membership: pd.DataFrame, category_order: list[str]
) -> set[tuple[str, str]]:
    cluster_by_category = membership.set_index("category")["cluster"].to_dict()
    return {
        tuple(sorted((left, right)))
        for left, right in combinations(category_order, 2)
        if cluster_by_category[left] == cluster_by_category[right]
    }


def plot_distance_agreement(
    agreement: pd.DataFrame,
    output_path: Path,
) -> None:
    names = list(agreement.index)
    figure, axis = plt.subplots(
        figsize=(max(8, 0.82 * len(names)), max(7, 0.72 * len(names)))
    )
    image = axis.imshow(agreement.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(names)))
    axis.set_yticklabels(names)
    axis.set_title("Agreement between experiment NCD matrices", loc="left", weight="bold")
    axis.set_xlabel("Spearman correlation of pairwise NCD rankings")
    axis.set_ylabel("Experiment")
    figure.colorbar(image, ax=axis, label="Spearman correlation (self = 1.0)", shrink=0.82)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_cluster_stability(stability: pd.DataFrame, output_path: Path) -> None:
    table = stability.copy()
    table["comparison"] = table["left"] + " vs " + table["right"]
    table = table.sort_values("pairwise_cluster_agreement")
    figure, axis = plt.subplots(figsize=(12, max(5, 0.42 * len(table))))
    axis.barh(
        table["comparison"],
        table["pairwise_cluster_agreement"],
        color=VISUAL_STYLE["accent_color"],
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Pairwise cluster agreement")
    axis.set_title("Stability of balanced-replica cluster assignments", loc="left", weight="bold")
    axis.grid(axis="x", color=VISUAL_STYLE["grid_color"], linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
