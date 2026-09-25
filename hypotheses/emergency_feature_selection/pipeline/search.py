"""Equal-budget random, independent EDA, and DAMICORE-guided EDA searches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

from damicore import estimate, run
import numpy as np

from .config import DEFAULT_CONFIG, FEATURES, ExperimentConfig
from .model import FULL_MASK, Mask, SolutionEvaluator


SearchMethod = Literal["random", "independent", "damicore"]
GroupBuilder = Callable[[list[Mask], Path, int, int, ExperimentConfig], list[tuple[int, ...]]]


@dataclass
class SearchResult:
    method: SearchMethod
    seed: int
    best_mask: Mask
    evaluated_masks: list[Mask]
    history: list[dict]
    clusters: list[dict]
    probabilities: list[dict]
    candidates: list[dict]


def random_mask(rng: np.random.Generator) -> Mask:
    while True:
        mask = tuple(int(bit) for bit in rng.integers(0, 2, size=len(FEATURES)))
        if any(mask):
            return mask


def initial_pool(seed: int, config: ExperimentConfig = DEFAULT_CONFIG) -> list[Mask]:
    minimum = len(FEATURES) + 1
    if config.initial_pool_size < minimum:
        raise ValueError(f"The initial pool needs at least {minimum} masks.")
    rng = np.random.default_rng(seed)
    masks = [FULL_MASK]
    masks.extend(
        tuple(int(index == selected) for index in range(len(FEATURES)))
        for selected in range(len(FEATURES))
    )
    seen = set(masks)
    while len(masks) < config.initial_pool_size:
        candidate = random_mask(rng)
        if candidate not in seen:
            masks.append(candidate)
            seen.add(candidate)
    return masks


def _split_large_group(
    members: list[int],
    distances,
    config: ExperimentConfig,
) -> list[tuple[int, ...]]:
    """Split DAMICORE communities by their NCD distances when joint tables grow sparse."""
    remaining = set(members)
    groups: list[tuple[int, ...]] = []
    while remaining:
        group = [min(remaining)]
        remaining.remove(group[0])
        while remaining and len(group) < config.max_block_size:
            nearest = min(
                remaining,
                key=lambda candidate: (
                    sum(
                        float(
                            distances.loc[
                                f"feature-{candidate:02d}.txt",
                                f"feature-{member:02d}.txt",
                            ]
                        )
                        for member in group
                    ) / len(group),
                    candidate,
                ),
            )
            group.append(nearest)
            remaining.remove(nearest)
        groups.append(tuple(sorted(group)))
    return groups


def damicore_groups(
    elite: list[Mask],
    artifact_root: Path,
    seed: int,
    generation: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> list[tuple[int, ...]]:
    """Cluster one binary elite-solution series per decision variable."""
    if len(elite) < 2:
        raise ValueError("DAMICORE needs at least two elite solutions.")
    rng = np.random.default_rng(seed * 1000 + generation)
    ordered_elite = [elite[index] for index in rng.permutation(len(elite))]
    corpus_dir = artifact_root / f"seed-{seed}" / f"generation-{generation:02d}" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for feature_index in range(len(FEATURES)):
        label = f"feature-{feature_index:02d}.txt"
        content = "\n".join(str(mask[feature_index]) for mask in ordered_elite) + "\n"
        (corpus_dir / label).write_text(content, encoding="ascii")

    preview = estimate(corpus_dir, source_kind="files")
    if not preview.within_limits:
        raise RuntimeError("DAMICORE resource limits rejected an elite-solution corpus.")
    output_dir = corpus_dir.parent / "damicore-run"
    result = run(corpus_dir, source_kind="files", output_dir=output_dir)
    try:
        membership = result.membership.copy()
        distances = result.distance_matrix.to_pandas()
        tree_newick = result.tree_newick
        run_report = result.report.model_dump(mode="json")
    finally:
        result.close()

    label_to_index = {
        f"feature-{index:02d}.txt": index for index in range(len(FEATURES))
    }
    if set(membership["label"]) != set(label_to_index):
        raise ValueError("DAMICORE membership does not cover every decision variable.")
    if set(distances.index) != set(label_to_index) or set(distances.columns) != set(label_to_index):
        raise ValueError("DAMICORE distances do not cover every decision variable.")
    if len(membership) != len(FEATURES) or not membership["object_id"].is_unique:
        raise ValueError("DAMICORE tree leaves cannot be mapped uniquely to fields.")
    membership["feature"] = membership["label"].map(
        lambda label: FEATURES[label_to_index[label]]
    )
    membership.to_csv(corpus_dir.parent / "raw_membership.csv", index=False)
    distances.rename(
        index={label: FEATURES[index] for label, index in label_to_index.items()},
        columns={label: FEATURES[index] for label, index in label_to_index.items()},
    ).to_csv(corpus_dir.parent / "ncd_matrix.csv")
    (corpus_dir.parent / "tree.nwk").write_text(tree_newick, encoding="utf-8")
    file_sizes = [
        (corpus_dir / f"feature-{index:02d}.txt").stat().st_size
        for index in range(len(FEATURES))
    ]
    if len(set(file_sizes)) != 1:
        raise ValueError("Decision-series files must have the same number of bytes.")
    run_report.update({
        "seed": seed,
        "generation": generation,
        "elite_size": len(elite),
        "file_count": len(FEATURES),
        "bytes_per_file": file_sizes[0],
    })
    (corpus_dir.parent / "damicore_summary.json").write_text(
        json.dumps(run_report, indent=2) + "\n", encoding="utf-8"
    )
    groups: list[tuple[int, ...]] = []
    for _, rows in membership.groupby("cluster", sort=True):
        members = sorted(label_to_index[label] for label in rows["label"])
        groups.extend(_split_large_group(members, distances, config))
    validate_partition(groups, config)
    return groups


def validate_partition(
    groups: list[tuple[int, ...]], config: ExperimentConfig = DEFAULT_CONFIG
) -> None:
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(FEATURES))):
        raise ValueError("Probability blocks must partition all decision variables exactly once.")
    if any(not group or len(group) > config.max_block_size for group in groups):
        raise ValueError("A probability block exceeds its size limit.")


def block_distributions(
    elite: list[Mask],
    groups: list[tuple[int, ...]],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> list[np.ndarray]:
    """Laplace-smoothed joint probabilities for each DAMICORE block."""
    validate_partition(groups, config)
    distributions: list[np.ndarray] = []
    for group in groups:
        counts = np.ones(2 ** len(group), dtype=float)
        for mask in elite:
            pattern = sum(mask[index] << position for position, index in enumerate(group))
            counts[pattern] += 1
        distributions.append(counts / counts.sum())
    return distributions


def sample_grouped(
    rng: np.random.Generator,
    groups: list[tuple[int, ...]],
    distributions: list[np.ndarray],
) -> Mask:
    decisions = [0] * len(FEATURES)
    for group, probabilities in zip(groups, distributions, strict=True):
        pattern = int(rng.choice(len(probabilities), p=probabilities))
        for position, index in enumerate(group):
            decisions[index] = (pattern >> position) & 1
    return tuple(decisions)


def sample_independent(rng: np.random.Generator, elite: list[Mask]) -> Mask:
    counts = np.asarray(elite, dtype=int).sum(axis=0)
    probabilities = (counts + 1) / (len(elite) + 2)
    return tuple(int(bit) for bit in (rng.random(len(FEATURES)) < probabilities))


def _next_unique_mask(
    rng: np.random.Generator,
    method: SearchMethod,
    seen: set[Mask],
    elite: list[Mask],
    groups: list[tuple[int, ...]],
    distributions: list[np.ndarray],
    config: ExperimentConfig,
) -> tuple[Mask, str]:
    for attempt in range(20_000):
        explore = method == "random" or rng.random() < config.random_exploration
        if explore or attempt > 1_000:
            candidate = random_mask(rng)
            origin = (
                "random" if method == "random"
                else "fallback" if attempt > 1_000
                else "exploration"
            )
        elif method == "independent":
            candidate = sample_independent(rng, elite)
            origin = "independent"
        else:
            candidate = sample_grouped(rng, groups, distributions)
            origin = "grouped"
        if any(candidate) and candidate not in seen:
            return candidate, origin
    raise RuntimeError("The search could not find a new binary solution.")


def run_search(
    method: SearchMethod,
    seed: int,
    evaluator: SolutionEvaluator,
    artifact_root: Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
    group_builder: GroupBuilder = damicore_groups,
) -> SearchResult:
    if method not in ("random", "independent", "damicore"):
        raise ValueError(f"Unknown search method: {method}")
    if config.elite_size > config.initial_pool_size:
        raise ValueError("Elite size cannot exceed the common initial pool.")
    rng = np.random.default_rng(seed + {"random": 0, "independent": 10_000, "damicore": 20_000}[method])
    evaluated = initial_pool(seed, config)
    seen = set(evaluated)
    history: list[dict] = []
    clusters: list[dict] = []
    probabilities: list[dict] = []
    candidates: list[dict] = []
    for mask in evaluated:
        evaluator.evaluate(mask)

    def record(generation: int) -> None:
        best = min(evaluated, key=evaluator.rank)
        metrics = evaluator.evaluate(best)
        history.append(
            {
                "method": method,
                "seed": seed,
                "generation": generation,
                "evaluations": len(evaluated),
                "best_mask": "".join(map(str, best)),
                "feature_count": sum(best),
                "acceptable": evaluator.is_acceptable(best),
                **metrics.as_dict(),
            }
        )

    record(0)
    for generation in range(1, config.generations + 1):
        elite = sorted(evaluated, key=evaluator.rank)[: config.elite_size]
        groups: list[tuple[int, ...]] = []
        distributions: list[np.ndarray] = []
        if method == "damicore":
            groups = group_builder(elite, artifact_root, seed, generation, config)
            validate_partition(groups, config)
            distributions = block_distributions(elite, groups, config)
            for group_number, (group, distribution) in enumerate(
                zip(groups, distributions, strict=True), start=1
            ):
                for index in group:
                    clusters.append(
                        {
                            "seed": seed,
                            "generation": generation,
                            "block": group_number,
                            "feature": FEATURES[index],
                        }
                    )
                for pattern, probability in enumerate(distribution):
                    probabilities.append({
                        "seed": seed,
                        "generation": generation,
                        "block": group_number,
                        "features": ", ".join(FEATURES[index] for index in group),
                        "pattern": "".join(
                            str((pattern >> position) & 1)
                            for position in range(len(group))
                        ),
                        "probability": float(probability),
                    })
        for candidate_number in range(1, config.offspring_per_generation + 1):
            mask, origin = _next_unique_mask(
                rng, method, seen, elite, groups, distributions, config
            )
            evaluator.evaluate(mask)
            evaluated.append(mask)
            seen.add(mask)
            if method == "damicore":
                candidates.append({
                    "seed": seed,
                    "generation": generation,
                    "candidate": candidate_number,
                    "mask": "".join(map(str, mask)),
                    "feature_count": sum(mask),
                    "origin": origin,
                })
        record(generation)
    if len(evaluated) != config.evaluations_per_method:
        raise AssertionError("Search methods must spend the same solution budget.")
    return SearchResult(
        method=method,
        seed=seed,
        best_mask=min(evaluated, key=evaluator.rank),
        evaluated_masks=evaluated,
        history=history,
        clusters=clusters,
        probabilities=probabilities,
        candidates=candidates,
    )


def greedy_forward(evaluator: SolutionEvaluator) -> Mask:
    """Simple deterministic reference, outside the equal-budget stochastic comparison."""
    selected: Mask | None = None
    remaining = set(range(len(FEATURES)))
    best = FULL_MASK
    while remaining:
        candidates = []
        for index in sorted(remaining):
            mask = tuple(
                int(position == index or (selected is not None and selected[position]))
                for position in range(len(FEATURES))
            )
            candidates.append(mask)
        selected = min(candidates, key=evaluator.rank)
        remaining -= {index for index, enabled in enumerate(selected) if enabled}
        if evaluator.rank(selected) < evaluator.rank(best):
            best = selected
    return best
