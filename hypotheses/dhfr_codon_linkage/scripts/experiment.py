"""Orchestrate the hypothesis: every seed, then the optimization and structure summaries."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ARTIFACT_ROOT, DEFAULT_CONFIG, ExperimentConfig
from .landscape import CODONS, POSITIONS, Landscape, decode, load_landscape, pair_epistasis
from .search import RunResult, run_search

PAIRS = list(combinations(range(POSITIONS), 2))
LABELS = [f"p{p + 1}" for p in range(POSITIONS)]


@dataclass
class ExperimentResults:
    output_dir: Path
    audit: dict
    runs: pd.DataFrame
    generations: pd.DataFrame
    pair_frequency: pd.DataFrame
    epistasis: pd.DataFrame
    pairs: pd.DataFrame


def _task(task: tuple[int, Path, ExperimentConfig]) -> RunResult:
    seed, output_dir, config = task
    keep_dir = output_dir / "example" if seed == config.example_seed else None
    return run_search(load_landscape(), seed, config, keep_dir)


def run_all(output_dir: Path, config: ExperimentConfig) -> list[RunResult]:
    tasks = [(seed, output_dir, config) for seed in config.seeds]
    if config.workers <= 1:
        return [_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        return list(pool.map(_task, tasks))


def pair_frequency(generations: pd.DataFrame) -> pd.DataFrame:
    """Share of DAMICORE groupings in which each pair of positions shares a group."""
    counts = np.zeros((POSITIONS, POSITIONS))
    for groups in generations["groups"]:
        for group in groups.split():
            members = [int(p) - 1 for p in group.split("-")]
            for i, j in combinations(members, 2):
                counts[i, j] += 1
                counts[j, i] += 1
    return pd.DataFrame(counts / len(generations), index=LABELS, columns=LABELS)


def _codon_contrast(frequency: np.ndarray, partition: tuple[tuple[int, ...], ...]) -> float:
    """Mean frequency of pairs inside a block minus pairs across blocks."""
    block = {p: b for b, members in enumerate(partition) for p in members}
    inside = [frequency[i, j] for i, j in PAIRS if block[i] == block[j]]
    across = [frequency[i, j] for i, j in PAIRS if block[i] != block[j]]
    return float(np.mean(inside) - np.mean(across))


def _triple_partitions() -> list[tuple[tuple[int, ...], ...]]:
    """All 280 ways to split the 9 positions into three unordered groups of three."""
    partitions = []
    for first in combinations(range(1, POSITIONS), 2):
        a = (0, *first)
        rest = [p for p in range(POSITIONS) if p not in a]
        for second in combinations(rest[1:], 2):
            b = (rest[0], *second)
            partitions.append((a, b, tuple(p for p in rest if p not in b)))
    return partitions


def codon_test(frequency: pd.DataFrame) -> dict:
    """Is the codon split special? Compare it with every other split into three triples.

    The p-value is the share of the 280 splits whose inside-minus-across contrast is at
    least the codons' contrast; it needs no assumption about the frequencies.
    """
    values = frequency.to_numpy()
    contrasts = np.array([_codon_contrast(values, partition) for partition in _triple_partitions()])
    observed = _codon_contrast(values, CODONS)
    return {
        "codon_contrast": observed,
        "codon_rank": int((contrasts > observed).sum()) + 1,
        "partitions": len(contrasts),
        "p_value": float((contrasts >= observed).mean()),
    }


def _pairs(frequency: pd.DataFrame, epistasis: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "pair": f"{i + 1}-{j + 1}",
            "same_codon": i // 3 == j // 3,
            "grouped_share": frequency.iat[i, j],
            "epistasis": epistasis.iat[i, j],
        }
        for i, j in PAIRS
    ]
    return pd.DataFrame(rows).sort_values("grouped_share", ascending=False, ignore_index=True)


def execute_experiment(
    config: ExperimentConfig = DEFAULT_CONFIG, output_root: Path = ARTIFACT_ROOT
) -> ExperimentResults:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    landscape: Landscape = load_landscape()
    results = run_all(output_dir, config)
    top = landscape.top_threshold(config.top_k)
    runs = pd.DataFrame(
        {
            "seed": r.seed,
            "best_fitness": r.best_fitness,
            "best_sequence": r.best_sequence,
            "share_of_optimum": r.best_fitness / landscape.optimum,
            "reached_top": r.best_fitness >= top,
            "found_optimum": r.best_fitness == landscape.optimum,
            "evaluations_to_top": r.evaluations_to_top,
            "generations": r.generations,
        }
        for r in results
    ).sort_values("seed", ignore_index=True)
    generations = pd.DataFrame({"seed": r.seed, **row} for r in results for row in r.history)

    frequency = pair_frequency(generations)
    epistasis = pair_epistasis(landscape, config.alive_threshold)
    pairs = _pairs(frequency, epistasis)
    correlation = spearmanr(pairs["grouped_share"], pairs["epistasis"])
    solutions, fitness = landscape.table()

    audit = {
        "config": asdict(config),
        "measured_sequences": int(landscape.measured.sum()),
        "search_space_size": int(landscape.measured.size),
        "share_of_space_per_run": config.budget / landscape.measured.size,
        "optimum": {"fitness": landscape.optimum, "sequence": decode(solutions[np.argmax(fitness)])},
        "top_threshold": top,
        "median_fitness": float(np.median(fitness)),
        "runs": len(runs),
        "reached_top_share": float(runs["reached_top"].mean()),
        "found_optimum_share": float(runs["found_optimum"].mean()),
        "mean_share_of_optimum": float(runs["share_of_optimum"].mean()),
        "median_evaluations_to_top": float(runs["evaluations_to_top"].median()),
        "damicore_runs": len(generations),
        "damicore_status": generations["damicore_status"].value_counts().to_dict(),
        "restarted_generations": int(generations["restarted"].sum()),
        "codon_test": codon_test(frequency),
        "grouped_share_vs_epistasis_spearman": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    runs.to_csv(output_dir / "runs.csv", index=False)
    generations.to_csv(output_dir / "generations.csv", index=False)
    frequency.to_csv(output_dir / "pair_frequency.csv")
    epistasis.to_csv(output_dir / "epistasis.csv")
    pairs.to_csv(output_dir / "pairs.csv", index=False)

    return ExperimentResults(output_dir, audit, runs, generations, frequency, epistasis, pairs)
