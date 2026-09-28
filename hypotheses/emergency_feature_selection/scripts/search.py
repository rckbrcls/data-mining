"""DAMICORE-guided multiobjective AED over binary field masks.

Following the "variables as samples" construction: after each generation the promising
masks are written as one file per field (that field's inclusion decisions, in Pareto order),
DAMICORE groups the fields through NCD, Neighbor Joining, and FastGreedy, and each group
receives a Laplace-smoothed joint probability table from which the next masks are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import json
from pathlib import Path

from damicore import run
from damicore.config import ExecutionConfig
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, FEATURES, ExperimentConfig
from .fitness import FEATURE_COUNT, Mask, MaskEvaluator, mask_string
from .pareto import pareto_order


@dataclass
class SearchResult:
    seed: int
    history: list[dict]
    probability_tables: list[dict]
    blocks: list[dict]
    diagnostics: list[dict]


def random_mask(rng: np.random.Generator) -> Mask:
    while True:
        candidate = tuple(int(bit) for bit in rng.integers(0, 2, FEATURE_COUNT))
        if any(candidate):
            return candidate


def initial_population(seed: int, config: ExperimentConfig = DEFAULT_CONFIG) -> list[Mask]:
    """Distinct masks with sizes cycling through 1..15, so every cost level starts covered."""
    rng = np.random.default_rng(seed)
    population: list[Mask] = []
    seen: set[Mask] = set()
    index = 0
    while len(population) < config.population_size:
        size = 1 + index % FEATURE_COUNT
        chosen = rng.choice(FEATURE_COUNT, size=size, replace=False)
        candidate = tuple(int(position in chosen) for position in range(FEATURE_COUNT))
        if candidate not in seen:
            population.append(candidate)
            seen.add(candidate)
        index += 1
    return population


def promising_masks(
    evaluated: list[Mask], evaluator: MaskEvaluator, config: ExperimentConfig
) -> list[Mask]:
    """Best share of everything evaluated so far, in Pareto order (front, then crowding)."""
    points = [(evaluator.quality(mask), sum(mask)) for mask in evaluated]
    ordered = [evaluated[index] for index in pareto_order(points)]
    keep = min(config.archive_size, max(2, ceil(config.selection_fraction * len(evaluated))))
    return ordered[:keep]


def validate_partition(groups: list[tuple[int, ...]]) -> None:
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(FEATURE_COUNT)):
        raise ValueError("Probability blocks must partition all features once.")


def write_feature_series(promising: list[Mask], series_dir: Path) -> int:
    """One file per field holding its inclusion bits across the promising masks.

    No separators: every file is one byte per mask, all in the same (Pareto) order, so the
    compressor can only find shared structure where two fields co-occur mask by mask.
    """
    series_dir.mkdir(parents=True, exist_ok=True)
    for feature_index in range(FEATURE_COUNT):
        payload = bytes(ord("0") + mask[feature_index] for mask in promising)
        (series_dir / f"feature-{feature_index:02d}.txt").write_bytes(payload)
    return len(promising)


def damicore_groups(
    promising: list[Mask],
    generation_dir: Path,
) -> tuple[list[tuple[int, ...]], list[dict], dict]:
    """Run DAMICORE over the field series; each of its clusters becomes one probability block."""
    if len(promising) < 2:
        raise ValueError("DAMICORE needs at least two promising masks.")
    series_bytes = write_feature_series(promising, generation_dir / "feature_series")

    result = run(
        generation_dir / "feature_series",
        source_kind="files",
        output_dir=generation_dir / "damicore_run",
        progress=False,
        execution=ExecutionConfig(workers=1),
    )
    try:
        membership = result.membership.copy()
        distances = result.distance_matrix.to_pandas()
        tree_newick = result.tree_newick
    finally:
        result.close()

    label_to_index = {f"feature-{index:02d}.txt": index for index in range(FEATURE_COUNT)}
    if set(membership["label"]) != set(label_to_index):
        raise ValueError("DAMICORE membership does not cover every feature.")
    names = {label: FEATURES[index] for label, index in label_to_index.items()}
    distances.rename(index=names, columns=names).to_csv(generation_dir / "ncd_matrix.csv")
    (generation_dir / "tree.nwk").write_text(tree_newick, encoding="utf-8")

    groups: list[tuple[int, ...]] = []
    block_rows: list[dict] = []
    for cluster_id, rows in membership.groupby("cluster", sort=True):
        group = tuple(sorted(label_to_index[label] for label in rows["label"]))
        groups.append(group)
        block_rows.append(
            {
                "damicore_cluster": str(cluster_id),
                "block_id": len(block_rows) + 1,
                "features": ", ".join(FEATURES[index] for index in group),
                "feature_count": len(group),
            }
        )
    validate_partition(groups)

    off_diagonal = distances.to_numpy()[~np.eye(FEATURE_COUNT, dtype=bool)]
    diagnostics = {
        "bytes_per_series": series_bytes,
        "block_count": len(groups),
        "largest_block": max(len(group) for group in groups),
        "ncd_min": float(off_diagonal.min()),
        "ncd_median": float(np.median(off_diagonal)),
        "ncd_max": float(off_diagonal.max()),
        "blocks": [[FEATURES[index] for index in group] for group in groups],
    }
    (generation_dir / "damicore_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return groups, block_rows, diagnostics


def probability_tables(
    promising: list[Mask],
    groups: list[tuple[int, ...]],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[list[np.ndarray], list[dict]]:
    """Laplace-smoothed joint inclusion probabilities inside each block."""
    validate_partition(groups)
    distributions: list[np.ndarray] = []
    records: list[dict] = []

    for block_index, group in enumerate(groups, start=1):
        observed_counts = np.zeros(2 ** len(group), dtype=int)
        for mask in promising:
            pattern_index = sum(
                mask[feature_index] << position
                for position, feature_index in enumerate(group)
            )
            observed_counts[pattern_index] += 1

        smoothed_counts = observed_counts.astype(float) + config.laplace_alpha
        probabilities = smoothed_counts / smoothed_counts.sum()
        distributions.append(probabilities)

        feature_names = ", ".join(FEATURES[index] for index in group)
        for pattern_index, probability in enumerate(probabilities):
            records.append(
                {
                    "block_id": block_index,
                    "features": feature_names,
                    # Bit i of the pattern is the i-th listed feature.
                    "pattern": "".join(
                        str((pattern_index >> position) & 1) for position in range(len(group))
                    ),
                    "promising_count": int(observed_counts[pattern_index]),
                    "laplace_alpha": config.laplace_alpha,
                    "smoothed_probability": float(probability),
                }
            )
    return distributions, records


def sample_mask(
    rng: np.random.Generator,
    groups: list[tuple[int, ...]],
    distributions: list[np.ndarray],
) -> Mask:
    decisions = [0] * FEATURE_COUNT
    for group, probabilities in zip(groups, distributions, strict=True):
        pattern_index = int(rng.choice(len(probabilities), p=probabilities))
        for position, feature_index in enumerate(group):
            decisions[feature_index] = (pattern_index >> position) & 1
    return tuple(decisions)


def _new_candidate(
    rng: np.random.Generator,
    seen: set[Mask],
    groups: list[tuple[int, ...]],
    distributions: list[np.ndarray],
) -> tuple[Mask, str]:
    """A not-yet-evaluated mask sampled from the tables (random only if they are exhausted)."""
    for _ in range(1_000):
        candidate = sample_mask(rng, groups, distributions)
        if any(candidate) and candidate not in seen:
            return candidate, "model_sample"
    while True:
        candidate = random_mask(rng)
        if candidate not in seen:
            return candidate, "random_fallback"


def run_search(
    seed: int,
    evaluator: MaskEvaluator,
    artifact_root: Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> SearchResult:
    """One AED run: initial population, then generations of DAMICORE-modelled offspring."""
    rng = np.random.default_rng(seed + 20_000)
    evaluated: list[Mask] = []
    seen: set[Mask] = set()
    history: list[dict] = []
    probability_records: list[dict] = []
    block_records: list[dict] = []
    diagnostics_records: list[dict] = []

    def record(mask: Mask, generation: int, origin: str) -> None:
        evaluated.append(mask)
        seen.add(mask)
        history.append(
            {
                "seed": seed,
                "generation": generation,
                "evaluation_index": len(evaluated),
                "mask": mask_string(mask),
                "feature_count": sum(mask),
                "origin": origin,
                "mean_gain_bits": evaluator.quality(mask),
                "worst_gain_bits": evaluator.worst_fold(mask),
            }
        )

    for mask in initial_population(seed, config):
        record(mask, 0, "initial_population")

    for generation in range(1, config.generations + 1):
        # 1. Promising solutions -> 2. one file per field -> 3. DAMICORE groups.
        promising = promising_masks(evaluated, evaluator, config)
        generation_dir = artifact_root / f"seed-{seed}" / f"generation-{generation:02d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        groups, blocks, diagnostics = damicore_groups(promising, generation_dir)
        diagnostics_records.append({"seed": seed, "generation": generation, **diagnostics})
        block_records.extend({"seed": seed, "generation": generation, **row} for row in blocks)

        # 4. Joint probability table per group -> 5. sample new masks -> 6. evaluate.
        distributions, tables = probability_tables(promising, groups, config)
        pd.DataFrame(tables).to_csv(generation_dir / "probability_tables.csv", index=False)
        probability_records.extend(
            {"seed": seed, "generation": generation, **row} for row in tables
        )
        for _ in range(config.offspring_per_generation):
            candidate, origin = _new_candidate(rng, seen, groups, distributions)
            record(candidate, generation, origin)

    if len(history) != config.evaluations_per_seed:
        raise AssertionError("Every run must use exactly the configured budget.")
    return SearchResult(
        seed=seed,
        history=history,
        probability_tables=probability_records,
        blocks=block_records,
        diagnostics=diagnostics_records,
    )
