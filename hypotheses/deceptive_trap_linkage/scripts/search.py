"""The AED of the "DAMICORE gerando modelos para AEDs" slide on binary strings.

Each generation:

1. The best ``selection_fraction`` of the population are the promising strings.
2. One file per variable lists the promising strings grouped by the variable's value.
3. DAMICORE (NCD -> Neighbor Joining -> FastGreedy) groups the variables.
4. Each group gets a joint table of the patterns seen among the promising strings, and
   offspring draw every group from its table.
5. Parents and offspring compete; the best ``population_size`` survive.

DAMICORE is the only source of structure: there is no uniform mixing and no fixed blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

from damicore import run
from damicore.config import ExecutionConfig
from damicore.errors import DamicoreError
import numpy as np

from .config import DEFAULT_CONFIG, ExperimentConfig
from .trap import Groups, TrapInstance, linkage_scores


@dataclass
class RunResult:
    block_count: int
    population_size: int
    seed: int
    success: bool
    generations: int
    evaluations: int
    best_fitness: int
    solved_blocks: int
    stop_reason: str
    history: list[dict]
    blocks: list[dict]


def _string_id(position: int) -> bytes:
    """Three printable bytes (base 90): zlib only copies matches of 3 bytes or more, so a
    shorter id shared by two series could not be recognised on its own."""
    if position >= 90**3:
        raise ValueError("String ids use three bytes; at most 729,000 promising strings are supported.")
    return bytes((33 + position // 8100, 33 + position // 90 % 90, 33 + position % 90))


def variable_series(promising: np.ndarray) -> list[bytes]:
    """One series per variable: the ids of the promising strings with a 0, then with a 1.

    Two variables that split the strings the same way write the same runs, which NCD can
    detect; writing the bits themselves hides most of that shared text.
    """
    series = []
    for column in promising.T:
        runs = [b"".join(_string_id(int(p)) for p in np.flatnonzero(column == value)) for value in (0, 1)]
        series.append(b"|".join(runs) + b"|")
    return series


def damicore_groups(promising: np.ndarray, keep_dir: Path | None = None) -> tuple[Groups, dict]:
    """Group the variables with DAMICORE over the promising strings.

    A constant variable has nothing to relate and stays alone; identical series carry the
    same information, so one copy goes to DAMICORE and the others join its group. When
    ``keep_dir`` is given, the NCD matrix, tree, and groups are saved there.
    """
    variables = promising.shape[1]
    series = variable_series(promising)
    constant = [v for v in range(variables) if promising[:, v].min() == promising[:, v].max()]
    copies: dict[int, list[int]] = {}
    first_with: dict[bytes, int] = {}
    for v in range(variables):
        if v not in constant:
            copies.setdefault(first_with.setdefault(series[v], v), []).append(v)
    representatives = sorted(copies)

    status = "ok"
    groups: Groups = [(v,) for v in constant]
    if len(representatives) < 3:
        status = "too_few_distinct_series"
        groups += [tuple(copies[r]) for r in representatives]
    else:
        work = Path(tempfile.mkdtemp(prefix="damicore-"))
        try:
            series_dir = work / "variable_series"
            series_dir.mkdir()
            for v in representatives:
                (series_dir / f"x{v:04d}.txt").write_bytes(series[v])
            result = run(
                series_dir,
                source_kind="files",
                output_dir=work / "damicore_run",
                progress=False,
                execution=ExecutionConfig(workers=1),
            )
            try:
                membership = result.membership.copy()
                if keep_dir is not None:
                    keep_dir.mkdir(parents=True, exist_ok=True)
                if keep_dir is not None:
                    names = {label: f"x{int(label[1:5])}" for label in membership["label"]}
                    distances = result.distance_matrix.to_pandas().rename(index=names, columns=names)
                    distances.to_csv(keep_dir / "ncd_matrix.csv")
                    tree = result.tree_newick
                    for object_id, label in zip(membership["object_id"], membership["label"], strict=True):
                        tree = tree.replace(f"'{object_id}'", f"'{names[label]}'")
                    (keep_dir / "tree.nwk").write_text(tree, encoding="utf-8")
            finally:
                result.close()
        except DamicoreError as error:
            status = f"failed: {error}"
            groups += [tuple(copies[r]) for r in representatives]
        else:
            clusters = list(membership.groupby("cluster", sort=True))
            if len(clusters) == 1:
                # One community over every series says nothing about linkage (FastGreedy
                # cannot split a tree of a few unrelated leaves); keep only the copies.
                status = "single_cluster"
                clusters = [(r, membership.loc[membership["label"].str[1:5] == f"{r:04d}"]) for r in representatives]
            for _, rows in clusters:
                members = [m for label in rows["label"] for m in copies[int(label[1:5])]]
                groups.append(tuple(sorted(members)))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    flattened = sorted(v for group in groups for v in group)
    if flattened != list(range(variables)):
        raise ValueError("DAMICORE groups must partition the variables once.")
    diagnostics = {
        "damicore_status": status,
        "constant_variables": len(constant),
        "duplicate_variables": variables - len(constant) - len(representatives),
        "group_count": len(groups),
        "largest_group": max(len(group) for group in groups),
    }
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        (keep_dir / "groups.json").write_text(
            json.dumps({**diagnostics, "groups": [list(g) for g in groups]}, indent=2) + "\n",
            encoding="utf-8",
        )
    return groups, diagnostics


def sample_offspring(
    rng: np.random.Generator, promising: np.ndarray, groups: Groups, size: int
) -> np.ndarray:
    """Each group draws one of its observed patterns, in proportion to how often it was seen."""
    offspring = np.empty((size, promising.shape[1]), dtype=promising.dtype)
    for group in groups:
        columns = list(group)
        patterns, counts = np.unique(promising[:, columns], axis=0, return_counts=True)
        offspring[:, columns] = patterns[rng.choice(len(patterns), size, p=counts / counts.sum())]
    return offspring


def run_search(
    instance: TrapInstance,
    population_size: int,
    seed: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
    keep_dir: Path | None = None,
) -> RunResult:
    """One run until the optimum, a stall, a converged population, or the generation cap."""
    rng = np.random.default_rng([seed, population_size, instance.block_count, 7])
    population = rng.integers(0, 2, (population_size, instance.variables), dtype=np.int8)
    fitness = instance.fitness(population)
    evaluations = population_size
    selected = max(2, int(round(config.selection_fraction * population_size)))
    history: list[dict] = []
    block_rows: list[dict] = []
    best, last_improvement, generation, stop_reason = int(fitness.max()), 0, 0, "max_generations"

    for generation in range(1, config.max_generations + 1):
        if best == instance.optimum:
            stop_reason = "optimum"
            generation -= 1
            break
        if (population == population[0]).all():
            stop_reason = "converged"
            generation -= 1
            break
        promising = population[np.argsort(-fitness, kind="stable")[:selected]]
        generation_dir = keep_dir / f"generation-{generation:02d}" if keep_dir is not None else None
        groups, diagnostics = damicore_groups(promising, generation_dir)
        offspring = sample_offspring(rng, promising, groups, population_size)
        offspring_fitness = instance.fitness(offspring)
        evaluations += population_size
        merged = np.vstack([population, offspring])
        merged_fitness = np.concatenate([fitness, offspring_fitness])
        survivors = np.argsort(-merged_fitness, kind="stable")[:population_size]
        population, fitness = merged[survivors], merged_fitness[survivors]

        if int(fitness.max()) > best:
            best, last_improvement = int(fitness.max()), generation
        history.append(
            {
                "generation": generation,
                "evaluations": evaluations,
                "best_fitness": best,
                "mean_fitness": float(fitness.mean()),
                "solved_blocks": instance.solved_blocks(population[0]),
                **diagnostics,
                **linkage_scores(groups, instance),
            }
        )
        block_rows.extend(
            {"generation": generation, "group_id": g, "variables": " ".join(map(str, group)), "size": len(group)}
            for g, group in enumerate(groups, start=1)
        )
        if best < instance.optimum and generation - last_improvement >= config.stall_generations:
            stop_reason = "stall"
            break
    else:
        if best == instance.optimum:
            stop_reason = "optimum"

    return RunResult(
        block_count=instance.block_count,
        population_size=population_size,
        seed=seed,
        success=best == instance.optimum,
        generations=generation,
        evaluations=evaluations,
        best_fitness=best,
        solved_blocks=instance.solved_blocks(population[int(np.argmax(fitness))]),
        stop_reason=stop_reason,
        history=history,
        blocks=block_rows,
    )
