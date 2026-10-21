"""The AED of the "DAMICORE gerando modelos para AEDs" slide on DNA sequences.

Each generation:

1. The best ``selection_fraction`` of the population are the promising sequences.
2. One file per position lists the promising sequences grouped by their base there.
3. DAMICORE (NCD -> Neighbor Joining -> FastGreedy) groups the positions.
4. Each group gets a joint table of the base patterns seen among the promising sequences,
   and offspring draw every group from its table.
5. Parents and offspring compete; the best ``population_size`` survive.

Every distinct sequence measured counts against the budget; asking again for a sequence
already measured is free, as a laboratory would look it up instead of repeating it.
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
from .landscape import POSITIONS, Landscape, decode

Groups = list[tuple[int, ...]]


@dataclass
class RunResult:
    seed: int
    best_fitness: float
    best_sequence: str
    evaluations_to_top: int | None
    generations: int
    history: list[dict]


def _string_id(position: int) -> bytes:
    """Three printable bytes (base 90): zlib only copies matches of 3 bytes or more, so a
    shorter id shared by two series could not be recognised on its own."""
    if position >= 90**3:
        raise ValueError("String ids use three bytes; at most 729,000 promising sequences are supported.")
    return bytes((33 + position // 8100, 33 + position // 90 % 90, 33 + position % 90))


def position_series(promising: np.ndarray) -> list[bytes]:
    """One series per position: the ids of the promising sequences with A, then C, G, T.

    Two positions that split the sequences the same way write the same runs, which NCD
    can detect, whatever the bases are called.
    """
    series = []
    for column in promising.T:
        runs = [b"".join(_string_id(int(p)) for p in np.flatnonzero(column == base)) for base in range(4)]
        series.append(b"|".join(runs) + b"|")
    return series


def damicore_groups(promising: np.ndarray, keep_dir: Path | None = None) -> tuple[Groups, dict]:
    """Group the positions with DAMICORE over the promising sequences.

    A constant position has nothing to relate and stays alone; identical series carry the
    same information, so one copy goes to DAMICORE and the others join its group. If
    FastGreedy returns one community over every series, it says nothing about linkage and
    each series keeps only its copies. When ``keep_dir`` is given, the NCD matrix, tree,
    and groups are saved there.
    """
    positions = promising.shape[1]
    series = position_series(promising)
    constant = [p for p in range(positions) if promising[:, p].min() == promising[:, p].max()]
    copies: dict[int, list[int]] = {}
    first_with: dict[bytes, int] = {}
    for p in range(positions):
        if p not in constant:
            copies.setdefault(first_with.setdefault(series[p], p), []).append(p)
    representatives = sorted(copies)

    status = "ok"
    groups: Groups = [(p,) for p in constant]
    if len(representatives) < 3:
        status = "too_few_distinct_series"
        groups += [tuple(copies[r]) for r in representatives]
    else:
        work = Path(tempfile.mkdtemp(prefix="damicore-"))
        try:
            series_dir = work / "position_series"
            series_dir.mkdir()
            for p in representatives:
                (series_dir / f"p{p + 1}.txt").write_bytes(series[p])
            result = run(
                series_dir,
                source_kind="files",
                output_dir=work / "damicore_run",
                progress=False,
                execution=ExecutionConfig(workers=1),
            )
            try:
                membership = result.membership.copy()
                membership["position"] = [int(label.split(".")[0][1:]) - 1 for label in membership["label"]]
                if keep_dir is not None:
                    keep_dir.mkdir(parents=True, exist_ok=True)
                    names = {label: label.split(".")[0] for label in membership["label"]}
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
            clusters = [rows["position"].tolist() for _, rows in membership.groupby("cluster", sort=True)]
            if len(clusters) == 1:
                status = "single_cluster"
                clusters = [[r] for r in representatives]
            for members in clusters:
                groups.append(tuple(sorted(m for r in members for m in copies[r])))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    if sorted(p for group in groups for p in group) != list(range(positions)):
        raise ValueError("DAMICORE groups must partition the positions once.")
    diagnostics = {
        "damicore_status": status,
        "constant_positions": len(constant),
        "group_count": len(groups),
        "largest_group": max(len(group) for group in groups),
    }
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        (keep_dir / "groups.json").write_text(
            json.dumps({**diagnostics, "groups": [[p + 1 for p in g] for g in groups]}, indent=2) + "\n",
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


class Laboratory:
    """Measures sequences from the landscape and counts the distinct ones."""

    def __init__(self, landscape: Landscape, budget: int) -> None:
        self.landscape, self.budget = landscape, budget
        self.measured: dict[int, float] = {}
        self.best_fitness, self.best_index = -np.inf, -1

    @property
    def exhausted(self) -> bool:
        return len(self.measured) >= self.budget

    def __call__(self, solutions: np.ndarray) -> np.ndarray:
        """Fitness of each sequence; past the budget, new sequences get -inf (not measured)."""
        result = np.empty(len(solutions))
        for row, index in enumerate(self.landscape.index(solutions)):
            index = int(index)
            if index not in self.measured:
                if self.exhausted:
                    result[row] = -np.inf
                    continue
                self.measured[index] = float(self.landscape.fitness[index])
                if self.measured[index] > self.best_fitness:
                    self.best_fitness, self.best_index = self.measured[index], index
            result[row] = self.measured[index]
        return result


def run_search(
    landscape: Landscape,
    seed: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
    keep_dir: Path | None = None,
) -> RunResult:
    """One run until the budget is spent."""
    rng = np.random.default_rng([seed, config.budget, 11])
    lab = Laboratory(landscape, config.budget)
    size = config.population_size
    top = landscape.top_threshold(config.top_k)
    population = rng.integers(0, 4, (size, POSITIONS))
    fitness = lab(population)
    selected = max(2, int(round(config.selection_fraction * size)))
    evaluations_to_top = len(lab.measured) if lab.best_fitness >= top else None
    history: list[dict] = []

    generation = 0
    while not lab.exhausted:
        generation += 1
        before = len(lab.measured)
        promising = population[np.argsort(-fitness, kind="stable")[:selected]]
        generation_dir = keep_dir / f"generation-{generation:02d}" if keep_dir is not None else None
        groups, diagnostics = damicore_groups(promising, generation_dir)
        offspring = sample_offspring(rng, promising, groups, size)
        offspring_fitness = lab(offspring)
        merged = np.vstack([population, offspring])
        merged_fitness = np.concatenate([fitness, offspring_fitness])
        survivors = np.argsort(-merged_fitness, kind="stable")[:size]
        population, fitness = merged[survivors], merged_fitness[survivors]
        restarted = len(lab.measured) == before
        if restarted:
            # The model only proposes sequences already measured: half the population
            # starts again at random, so the search keeps exploring.
            population[size // 2 :] = rng.integers(0, 4, (size - size // 2, POSITIONS))
            fitness[size // 2 :] = lab(population[size // 2 :])
        if evaluations_to_top is None and lab.best_fitness >= top:
            evaluations_to_top = len(lab.measured)
        history.append(
            {
                "generation": generation,
                "evaluations": len(lab.measured),
                "best_fitness": lab.best_fitness,
                "mean_promising_fitness": float(np.sort(fitness)[-selected:].mean()),
                "restarted": restarted,
                "groups": " ".join("-".join(str(p + 1) for p in g) for g in groups),
                **diagnostics,
            }
        )

    return RunResult(
        seed=seed,
        best_fitness=lab.best_fitness,
        best_sequence=decode((lab.best_index // 4 ** np.arange(POSITIONS - 1, -1, -1)) % 4),
        evaluations_to_top=evaluations_to_top,
        generations=generation,
        history=history,
    )
