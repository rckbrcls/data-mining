"""Orchestrate the hypothesis: every problem size, population size, and seed, then the summary."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_ROOT, DEFAULT_CONFIG, ExperimentConfig
from .search import RunResult, run_search
from .trap import make_instance


@dataclass
class ExperimentResults:
    output_dir: Path
    audit: dict
    runs: pd.DataFrame
    generations: pd.DataFrame
    blocks: pd.DataFrame
    summary: pd.DataFrame
    population_needed: pd.DataFrame


def _task(task: tuple[int, int, int, Path, ExperimentConfig]) -> RunResult:
    block_count, population_size, seed, output_dir, config = task
    instance = make_instance(block_count, config.trap_size, seed)
    keep_dir = None
    if seed == config.example_seed:
        keep_dir = output_dir / "examples" / f"blocks-{block_count:02d}_population-{population_size}"
    return run_search(instance, population_size, seed, config, keep_dir)


def run_all(output_dir: Path, config: ExperimentConfig) -> list[RunResult]:
    # Largest problems first, so the slowest runs do not end up alone at the end.
    tasks = [
        (block_count, population_size, seed, output_dir, config)
        for block_count in sorted(config.block_counts, reverse=True)
        for population_size in sorted(config.population_sizes, reverse=True)
        for seed in config.seeds
    ]
    if config.workers <= 1:
        return [_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        return list(pool.map(_task, tasks))


def _summary(runs: pd.DataFrame, generations: pd.DataFrame) -> pd.DataFrame:
    """Per problem and population size: success, cost, and how well DAMICORE found the traps."""
    keys = ["block_count", "population_size"]
    last = generations.sort_values("generation").groupby([*keys, "seed"]).tail(1)
    first = generations.loc[generations["generation"] == 1]
    summary = runs.groupby(keys).agg(
        variables=("variables", "first"),
        seeds=("seed", "size"),
        success_rate=("success", "mean"),
        mean_solved_share=("solved_share", "mean"),
        mean_generations=("generations", "mean"),
    )
    summary["mean_evaluations_when_solved"] = runs.loc[runs["success"]].groupby(keys)["evaluations"].mean()
    summary["ari_generation_1"] = first.groupby(keys)["ari"].mean()
    summary["ari_best"] = generations.groupby([*keys, "seed"])["ari"].max().groupby(keys).mean()
    summary["ari_last"] = last.groupby(keys)["ari"].mean()
    summary["exact_block_share_best"] = (
        generations.assign(share=generations["exact_blocks"] / generations["block_count"])
        .groupby([*keys, "seed"])["share"]
        .max()
        .groupby(keys)
        .mean()
    )
    return summary.reset_index()


def _population_needed(summary: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Smallest tested population that solves each problem size in enough seeds."""
    reached = summary.loc[summary["success_rate"] >= config.required_success]
    needed = reached.sort_values("population_size").groupby("block_count").head(1)
    rows = needed[
        ["block_count", "variables", "population_size", "success_rate", "mean_generations",
         "mean_evaluations_when_solved", "ari_best", "exact_block_share_best"]
    ]
    missing = sorted(set(config.block_counts) - set(rows["block_count"]))
    unsolved = pd.DataFrame(
        {"block_count": missing, "variables": [b * config.trap_size for b in missing]}
    )
    return pd.concat([rows, unsolved], ignore_index=True).sort_values("block_count", ignore_index=True)


def _scaling_exponent(needed: pd.DataFrame) -> float:
    """Slope of log(evaluations) on log(variables) over the solved problem sizes."""
    solved = needed.dropna(subset=["mean_evaluations_when_solved"])
    if len(solved) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(solved["variables"]), np.log(solved["mean_evaluations_when_solved"]), 1)
    return float(slope)


def execute_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG, output_root: Path = ARTIFACT_ROOT
) -> ExperimentResults:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    results = run_all(output_dir, config)
    keys = ("block_count", "population_size", "seed")
    runs = pd.DataFrame(
        {
            "block_count": r.block_count,
            "variables": r.block_count * config.trap_size,
            "population_size": r.population_size,
            "seed": r.seed,
            "success": r.success,
            "generations": r.generations,
            "evaluations": r.evaluations,
            "best_fitness": r.best_fitness,
            "solved_blocks": r.solved_blocks,
            "solved_share": r.solved_blocks / r.block_count,
            "stop_reason": r.stop_reason,
        }
        for r in results
    ).sort_values(list(keys), ignore_index=True)
    generations = pd.DataFrame(
        {**dict(zip(keys, (r.block_count, r.population_size, r.seed))), **row}
        for r in results
        for row in r.history
    )
    blocks = pd.DataFrame(
        {**dict(zip(keys, (r.block_count, r.population_size, r.seed))), **row}
        for r in results
        for row in r.blocks
    )
    summary = _summary(runs, generations)
    needed = _population_needed(summary, config)

    audit = {
        "config": asdict(config),
        "runs": len(runs),
        "successful_runs": int(runs["success"].sum()),
        "damicore_runs": len(generations),
        "damicore_runs_not_ok": int((generations["damicore_status"] != "ok").sum()),
        "stop_reasons": runs["stop_reason"].value_counts().to_dict(),
        "solved_problem_sizes": int(needed["population_size"].notna().sum()),
        "evaluations_scaling_exponent": _scaling_exponent(needed),
        "ari_best_when_solved": float(
            summary.loc[summary["success_rate"] >= config.required_success, "ari_best"].mean()
        ),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    for name, table in {
        "runs": runs,
        "generations": generations,
        "blocks": blocks,
        "summary": summary,
        "population_needed": needed,
    }.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    return ExperimentResults(output_dir, audit, runs, generations, blocks, summary, needed)
