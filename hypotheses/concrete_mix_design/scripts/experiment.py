"""Orchestrate the hypothesis: the DAMICORE AED on every seed, then the ground-truth comparison."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import binomtest

from .config import (
    AGE_COLUMN,
    ARTIFACT_ROOT,
    COMPONENTS,
    DATA_PATH,
    DEFAULT_CONFIG,
    PRICES_USD_PER_KG,
    TARGET_COLUMN,
    ExperimentConfig,
)
from .landscape import (
    Landscape,
    cached_landscape,
    load_landscape,
    load_tests,
    surrogate_validation,
    unique_mixes,
)
from .pareto import front_mask, hypervolume, pareto_front
from .search import SearchResult, run_search, solution_from_string


@dataclass
class GroundTruth:
    """What only full enumeration knows: the true front and its hypervolume."""

    reference: tuple[float, float]
    hypervolume: float
    front: pd.DataFrame
    cheapest: dict[float, float]


@dataclass
class ExperimentResults:
    output_dir: Path
    audit: dict
    landscape: Landscape
    search_history: pd.DataFrame
    convergence: pd.DataFrame
    run_metrics: pd.DataFrame
    summary: pd.DataFrame
    origins: pd.DataFrame
    probability_tables: pd.DataFrame
    blocks: pd.DataFrame
    niches: pd.DataFrame
    damicore_diagnostics: pd.DataFrame
    block_cooccurrence: pd.DataFrame
    pair_tests: pd.DataFrame
    front: pd.DataFrame
    true_front: pd.DataFrame
    recommendations: pd.DataFrame


def _all_solutions(landscape: Landscape, positions: np.ndarray) -> list[str]:
    coordinates = np.unravel_index(positions, landscape.shape)
    return ["".join(str(axis[i]) for axis in coordinates) for i in range(len(positions))]


def ground_truth(landscape: Landscape, config: ExperimentConfig) -> GroundTruth:
    feasible = np.flatnonzero(landscape.feasible)
    strength = landscape.strength[feasible].astype(float)
    cost = landscape.cost[feasible].astype(float)
    reference = (float(strength.min()), float(cost.max()))
    keep = front_mask(strength, cost)
    front = pd.DataFrame(
        {
            "solution": _all_solutions(landscape, feasible[keep]),
            "strength_mpa": strength[keep],
            "cost_usd_m3": cost[keep],
        }
    ).sort_values("cost_usd_m3", ignore_index=True)
    cheapest = {
        target: float(cost[strength >= target].min()) for target in config.strength_targets
    }
    return GroundTruth(reference, hypervolume(strength, cost, reference), front, cheapest)


# Runs are independent; each worker process loads the scored landscape once.
_WORKER_LANDSCAPE: Landscape | None = None


def _initialize_worker(landscape_path: Path, config: ExperimentConfig) -> None:
    global _WORKER_LANDSCAPE
    _WORKER_LANDSCAPE = load_landscape(landscape_path, load_tests(DATA_PATH), config)


def _search_task(task: tuple[int, Path, ExperimentConfig]) -> SearchResult:
    seed, artifact_root, config = task
    assert _WORKER_LANDSCAPE is not None
    return run_search(seed, _WORKER_LANDSCAPE, artifact_root, config)


def run_searches(
    landscape_path: Path, artifact_root: Path, config: ExperimentConfig
) -> list[SearchResult]:
    tasks = [(seed, artifact_root, config) for seed in config.seeds]
    workers = min(len(tasks), config.workers)
    if workers <= 1:
        _initialize_worker(landscape_path, config)
        return [_search_task(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(landscape_path, config),
    ) as pool:
        return list(pool.map(_search_task, tasks))


def _feasible_arrays(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feasible = history.loc[history["violation"] == 0]
    return feasible["strength_mpa"].to_numpy(), feasible["cost_usd_m3"].to_numpy()


def _convergence(history: pd.DataFrame, truth: GroundTruth, config: ExperimentConfig) -> pd.DataFrame:
    """Share of the true hypervolume reached after every ``checkpoint_every`` evaluations."""
    rows = []
    checkpoints = range(config.checkpoint_every, config.evaluations_per_run + 1, config.checkpoint_every)
    for seed, run_history in history.groupby("seed"):
        for evaluations in checkpoints:
            strength, cost = _feasible_arrays(run_history.loc[run_history["evaluation_index"] <= evaluations])
            rows.append(
                {
                    "seed": seed,
                    "evaluations": evaluations,
                    "hypervolume_ratio": hypervolume(strength, cost, truth.reference) / truth.hypervolume,
                }
            )
    return pd.DataFrame(rows)


def _run_metrics(history: pd.DataFrame, truth: GroundTruth, config: ExperimentConfig) -> pd.DataFrame:
    """Per run: hypervolume ratio, feasible share, and extra cost over the true cheapest mix."""
    rows = []
    for seed, run_history in history.groupby("seed"):
        strength, cost = _feasible_arrays(run_history)
        row = {
            "seed": seed,
            "hypervolume_ratio": hypervolume(strength, cost, truth.reference) / truth.hypervolume,
            "feasible_share": float((run_history["violation"] == 0).mean()),
            "true_front_points_found": int(
                truth.front["solution"].isin(run_history["solution"]).sum()
            ),
        }
        for target in config.strength_targets:
            reached = strength >= target
            found = float(cost[reached].min()) if reached.any() else np.nan
            row[f"cost_gap_{target:g}mpa"] = found - truth.cheapest[target]
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean, spread, and range of each run metric over the seeds."""
    values = metrics.drop(columns="seed")
    return values.agg(["mean", "std", "min", "max"]).T.rename_axis("metric").reset_index()


def _origins(history: pd.DataFrame) -> pd.DataFrame:
    """Which model produced the offspring (initial population excluded)."""
    offspring = history.loc[history["generation"] > 0]
    counts = offspring["origin"].value_counts()
    return pd.DataFrame({"offspring": counts, "share": counts / len(offspring)}).rename_axis(
        "origin"
    ).reset_index()


def _block_cooccurrence(blocks: pd.DataFrame, scope: str = "archive") -> pd.DataFrame:
    """Share of generations (all seeds) in which two components share a DAMICORE block."""
    position = {name: index for index, name in enumerate(COMPONENTS)}
    counts = np.zeros((len(COMPONENTS), len(COMPONENTS)))
    generations = blocks.loc[blocks["scope"] == scope].groupby(["seed", "generation"])
    for _, generation_blocks in generations:
        for components in generation_blocks["components"]:
            members = [position[name] for name in components.split(", ")]
            for left, right in combinations(members, 2):
                counts[left, right] += 1
                counts[right, left] += 1
    counts /= max(1, generations.ngroups)
    np.fill_diagonal(counts, 1.0)
    return pd.DataFrame(counts, index=list(COMPONENTS), columns=list(COMPONENTS))


def _chance_same_block(blocks: pd.DataFrame, scope: str = "archive") -> float:
    """Share expected for any pair if blocks of the same sizes took components at random."""
    scoped = blocks.loc[blocks["scope"] == scope]
    pairs = len(COMPONENTS) * (len(COMPONENTS) - 1) / 2
    within = scoped["component_count"] * (scoped["component_count"] - 1) / 2
    return float(within.groupby([scoped["seed"], scoped["generation"]]).sum().mean() / pairs)


def _pair_tests(blocks: pd.DataFrame, cooccurrence: pd.DataFrame) -> pd.DataFrame:
    """Each pair against chance: one-sided binomial test over the archive-level generations.

    Generations of one seed are not independent, so the p-values are a screen, not proof;
    the Bonferroni column corrects for the 21 pairs.
    """
    chance = _chance_same_block(blocks)
    generations = blocks.loc[blocks["scope"] == "archive", ["seed", "generation"]].drop_duplicates()
    total = len(generations)
    rows = []
    for left, right in combinations(COMPONENTS, 2):
        share = float(cooccurrence.loc[left, right])
        together = int(round(share * total))
        p_value = float(binomtest(together, total, chance, alternative="greater").pvalue)
        rows.append(
            {
                "pair": f"{left} + {right}",
                "same_block_share": share,
                "chance_share": chance,
                "generations": total,
                "binomial_p": p_value,
                "bonferroni_p": min(1.0, p_value * 21),
            }
        )
    return pd.DataFrame(rows).sort_values("same_block_share", ascending=False, ignore_index=True)


def _niche_separation(niches: pd.DataFrame) -> float:
    """Mean share of the promising mixes' strength range that niches split apart.

    Per generation with two or more niches: 1 - (mean range within a niche) / (overall range).
    0 means every niche spans the whole front; 1 means the niches do not overlap at all.
    """
    shares = []
    for _, generation in niches.loc[niches["niche_count"] > 1].groupby(["seed", "generation"]):
        overall = generation["strength_max"].max() - generation["strength_min"].min()
        within = (generation["strength_max"] - generation["strength_min"]).mean()
        if overall > 0:
            shares.append(1 - within / overall)
    return float(np.mean(shares)) if shares else float("nan")


def _with_mix(table: pd.DataFrame, landscape: Landscape) -> pd.DataFrame:
    mixes = pd.DataFrame(
        [landscape.mix(solution_from_string(value)) for value in table["solution"]],
        index=table.index,
    )
    binder = mixes[["cement", "slag", "fly_ash"]].sum(axis=1)
    return table.join(mixes).assign(
        water_cement_ratio=mixes["water"] / mixes["cement"],
        water_binder_ratio=mixes["water"] / binder,
    )


def _recommendations(
    history: pd.DataFrame,
    truth: GroundTruth,
    landscape: Landscape,
    tests: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Cheapest mix reaching each strength target, by the AED (all seeds pooled) and the truth.

    Each mix also shows the closest real laboratory mix and its measured strength at the
    design age, as a sanity check on the surrogate.
    """
    candidates = [
        ("aed_damicore", history.loc[history["violation"] == 0]),
        ("ground_truth", truth.front),
    ]
    rows = []
    for target in config.strength_targets:
        for source, table in candidates:
            reached = table.loc[table["strength_mpa"] >= target]
            best = reached.sort_values(["cost_usd_m3", "solution"]).iloc[0]
            rows.append(
                {
                    "strength_target_mpa": target,
                    "source": source,
                    "solution": best["solution"],
                    "strength_mpa": float(best["strength_mpa"]),
                    "cost_usd_m3": float(best["cost_usd_m3"]),
                    "cost_gap_usd_m3": float(best["cost_usd_m3"]) - truth.cheapest[target],
                }
            )
    table = _with_mix(pd.DataFrame(rows), landscape)

    real = unique_mixes(tests)
    center, scale = real.mean(axis=0), real.std(axis=0)
    _, nearest = cKDTree((real - center) / scale).query(
        (table[list(COMPONENTS)].to_numpy() - center) / scale
    )
    at_age = tests.loc[tests[AGE_COLUMN] == config.design_age_days]
    measured = at_age.groupby(list(COMPONENTS))[TARGET_COLUMN].mean()
    table["nearest_real_mix"] = [
        ", ".join(f"{name}={value:g}" for name, value in zip(COMPONENTS, real[i], strict=True))
        for i in nearest
    ]
    table["nearest_real_strength_at_design_age"] = [
        measured.get(tuple(real[i]), np.nan) for i in nearest
    ]
    return table


def execute_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG, output_root: Path = ARTIFACT_ROOT
) -> ExperimentResults:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    # 1. Data, surrogate check, and the scored search space.
    tests = load_tests(DATA_PATH)
    surrogate = surrogate_validation(tests)
    landscape, landscape_path, from_cache = cached_landscape(tests, config, output_root / "cache")
    truth = ground_truth(landscape, config)

    # 2. The DAMICORE AED, every seed.
    searches = run_searches(landscape_path, output_dir / "searches", config)
    history = pd.DataFrame(row for search in searches for row in search.history)
    probability_tables = pd.DataFrame(row for s in searches for row in s.probability_tables)
    blocks = pd.DataFrame(row for s in searches for row in s.blocks)
    niches = pd.DataFrame(row for s in searches for row in s.niches)
    diagnostics = pd.DataFrame(row for s in searches for row in s.diagnostics)

    # 3. Comparison with the ground truth, and what DAMICORE found.
    convergence = _convergence(history, truth, config)
    metrics = _run_metrics(history, truth, config)
    summary = _summary(metrics)
    origins = _origins(history)
    cooccurrence = _block_cooccurrence(blocks)
    pair_tests = _pair_tests(blocks, cooccurrence)
    front = pareto_front(history)
    niche_generations = niches.drop_duplicates(["seed", "generation"])
    recommendations = _recommendations(history, truth, landscape, tests, config)

    audit = {
        "data_path": str(DATA_PATH.relative_to(DATA_PATH.parents[3])),
        "laboratory_tests": len(tests),
        "unique_mixes": len(unique_mixes(tests)),
        "surrogate": surrogate,
        "config": asdict(config),
        "prices_usd_per_kg": PRICES_USD_PER_KG,
        "search_space_size": int(landscape.strength.size),
        "feasible_mixes": int(landscape.feasible.sum()),
        "evaluations_per_run": config.evaluations_per_run,
        "share_of_space_per_run": config.evaluations_per_run / landscape.strength.size,
        "hypervolume_reference": truth.reference,
        "true_hypervolume": truth.hypervolume,
        "true_front_size": len(truth.front),
        "true_cheapest_by_target": {f"{k:g}": v for k, v in truth.cheapest.items()},
        "landscape_from_cache": from_cache,
        "hypervolume_ratio_mean": float(metrics["hypervolume_ratio"].mean()),
        "uniform_fallback_offspring": int((history["origin"] == "uniform_fallback").sum()),
        "cement_water_same_block_share": float(cooccurrence.loc["cement", "water"]),
        "same_block_share_by_chance": _chance_same_block(blocks),
        "damicore_component_runs": len(diagnostics),
        "damicore_component_runs_without_tree": int((diagnostics["damicore_status"] != "ok").sum()),
        "niche_count_share": {
            str(count): float(share)
            for count, share in niche_generations["niche_count"].value_counts(normalize=True).sort_index().items()
        },
        "niche_status_share": {
            str(status): float(share)
            for status, share in niche_generations["niche_status"].value_counts(normalize=True).items()
        },
        "niche_strength_separation": _niche_separation(niches),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for name, table in {
        "search_history": history,
        "convergence": convergence,
        "run_metrics": metrics,
        "summary": summary,
        "origins": origins,
        "probability_tables": probability_tables,
        "blocks": blocks,
        "niches": niches,
        "damicore_diagnostics": diagnostics,
        "pair_tests": pair_tests,
        "aed_front": front,
        "true_pareto_front": truth.front,
        "recommendations": recommendations,
    }.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    cooccurrence.to_csv(output_dir / "block_cooccurrence.csv")

    return ExperimentResults(
        output_dir=output_dir,
        audit=audit,
        landscape=landscape,
        search_history=history,
        convergence=convergence,
        run_metrics=metrics,
        summary=summary,
        origins=origins,
        probability_tables=probability_tables,
        blocks=blocks,
        niches=niches,
        damicore_diagnostics=diagnostics,
        block_cooccurrence=cooccurrence,
        pair_tests=pair_tests,
        front=front,
        true_front=truth.front,
        recommendations=recommendations,
    )
