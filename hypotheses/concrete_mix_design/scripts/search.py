"""The AED of the "DAMICORE gerando modelos para AEDs" slide, with DAMICORE building the whole model.

Each generation:

1. The promising mixes are the best evaluated so far, feasible Pareto fronts first.
2. DAMICORE over the mixes (one file per mix) splits them into niches: regions of the front.
3. DAMICORE over the components (one file per component) groups the variables, once over
   all promising mixes and once inside each niche.
4. Each niche gets a joint probability table per group, and new mixes are drawn from them.

Nothing else shapes the model: no uniform mixing and no fixed block structure. When a
niche's model only reproduces mixes already evaluated, the draw moves to the other niches'
models and then to the model of all promising mixes; a uniform draw is the last resort and
is logged as such.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
import json
from pathlib import Path

from damicore import run
from damicore.config import ExecutionConfig
from damicore.errors import DamicoreError
import numpy as np
import pandas as pd

from .config import COMPONENTS, DEFAULT_CONFIG, ExperimentConfig
from .landscape import COMPONENT_COUNT, Landscape, Solution
from .pareto import constrained_order

Groups = list[tuple[int, ...]]


@dataclass
class BlockModel:
    """Joint table of one group: observed level patterns and how often each appeared."""

    group: tuple[int, ...]
    patterns: list[tuple[int, ...]]
    counts: np.ndarray


@dataclass
class NicheModel:
    """One region of the promising mixes, with its own DAMICORE groups and tables."""

    members: list[Solution]
    groups: Groups
    blocks: list[BlockModel]


@dataclass
class SearchResult:
    seed: int
    history: list[dict]
    probability_tables: list[dict]
    blocks: list[dict]
    niches: list[dict]
    diagnostics: list[dict]


def solution_string(solution: Solution) -> str:
    return "".join(str(level) for level in solution)


def solution_from_string(value: str) -> Solution:
    return tuple(int(character) for character in value)


def random_solution(rng: np.random.Generator, levels: int) -> Solution:
    return tuple(int(level) for level in rng.integers(0, levels, COMPONENT_COUNT))


def initial_population(seed: int, config: ExperimentConfig = DEFAULT_CONFIG) -> list[Solution]:
    """Distinct uniform mixes, the same list for a given seed."""
    rng = np.random.default_rng(seed)
    population: list[Solution] = []
    seen: set[Solution] = set()
    while len(population) < config.population_size:
        candidate = random_solution(rng, config.levels)
        if candidate not in seen:
            population.append(candidate)
            seen.add(candidate)
    return population


def promising_solutions(
    evaluated: list[Solution], scores: list[dict], config: ExperimentConfig
) -> tuple[list[Solution], list[dict]]:
    """Best share of everything evaluated so far, feasible Pareto fronts first."""
    points = [
        (score["strength_mpa"], score["cost_usd_m3"], score["violation"]) for score in scores
    ]
    keep = min(config.archive_size, max(2, ceil(config.selection_fraction * len(evaluated))))
    ordered = constrained_order(points)[:keep]
    return [evaluated[index] for index in ordered], [scores[index] for index in ordered]


def validate_partition(groups: Groups) -> None:
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(COMPONENT_COUNT)):
        raise ValueError("Probability blocks must partition all components once.")


def _mix_id(position: int) -> bytes:
    return bytes((33 + position // 90, 33 + position % 90))


def component_series(promising: list[Solution]) -> list[bytes]:
    """One series per component: the promising mixes grouped by the level they give it.

    Each mix is a two-byte id; the ids sharing a level form one run, runs in level order.
    Two components that split the mixes the same way write the same runs whatever the level
    labels are, so "much cement, little water" shows up as shared text that NCD can detect.
    Writing the levels themselves (one character per mix) hides such relations, because the
    two series then share no characters.
    """
    if len(promising) > 90 * 90:
        raise ValueError("Mix ids use two bytes; at most 8,100 promising mixes are supported.")
    series = []
    for component in range(COMPONENT_COUNT):
        runs: dict[int, list[bytes]] = {}
        for position, solution in enumerate(promising):
            runs.setdefault(solution[component], []).append(_mix_id(position))
        series.append(b"|".join(b"".join(runs[level]) for level in sorted(runs)) + b"|")
    return series


def mix_document(solution: Solution) -> bytes:
    """One mix as text: each component's letter repeated (level + 1) times.

    Close levels share long runs of the same letter, so NCD finds similar mixes similar.
    """
    return b"".join(bytes([65 + index]) * (level + 1) + b"." for index, level in enumerate(solution))


def _run_damicore(
    source_dir: Path, output_dir: Path, num_clusters: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """DAMICORE over the files of ``source_dir``: membership, NCD matrix, and Newick tree."""
    result = run(
        source_dir,
        source_kind="files",
        num_clusters=num_clusters,
        output_dir=output_dir,
        progress=False,
        execution=ExecutionConfig(workers=1),
    )
    try:
        return result.membership.copy(), result.distance_matrix.to_pandas(), result.tree_newick
    finally:
        result.close()


def _expand(representative_groups: Groups, copies: dict[int, list[int]]) -> Groups:
    return [
        tuple(sorted(index for representative in group for index in copies[representative]))
        for group in representative_groups
    ]


def damicore_groups(
    promising: list[Solution], output_dir: Path, fallback: Groups | None = None
) -> tuple[Groups, dict]:
    """Run DAMICORE over the component series; each of its clusters becomes one block.

    Identical series carry the same information, so only one copy goes to DAMICORE and the
    others join its block. A constant series (component fixed in every mix) has nothing to
    relate and stays alone. If DAMICORE cannot build a tree, the ``fallback`` groups (an
    earlier DAMICORE result) are used, else one block per series, and the status says so.
    """
    if len(promising) < 2:
        raise ValueError("DAMICORE needs at least two promising mixes.")
    series = component_series(promising)
    constant = [
        index for index in range(COMPONENT_COUNT) if len({s[index] for s in promising}) == 1
    ]
    copies: dict[int, list[int]] = {}
    first_with: dict[bytes, int] = {}
    for index, values in enumerate(series):
        if index in constant:
            continue
        representative = first_with.setdefault(values, index)
        copies.setdefault(representative, []).append(index)
    representatives = sorted(copies)

    series_dir = output_dir / "component_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for index in representatives:
        (series_dir / f"component-{index}.txt").write_bytes(series[index])

    status = "ok"
    distances = None
    groups: Groups | None = None
    if len(representatives) < 3:
        status = "too_few_distinct_series"
    else:
        try:
            membership, distances, tree_newick = _run_damicore(series_dir, output_dir / "damicore_run")
        except DamicoreError as error:
            status = f"failed: {error}"
        else:
            label_to_index = {f"component-{index}.txt": index for index in representatives}
            if set(membership["label"]) != set(label_to_index):
                raise ValueError("DAMICORE membership does not cover every component.")
            names = {label: COMPONENTS[index] for label, index in label_to_index.items()}
            distances = distances.rename(index=names, columns=names)
            distances.to_csv(output_dir / "ncd_matrix.csv")
            # The tree names leaves by DAMICORE's object ids; write component names instead.
            for object_id, label in zip(membership["object_id"], membership["label"], strict=True):
                tree_newick = tree_newick.replace(f"'{object_id}'", f"'{names[label]}'")
            (output_dir / "tree.nwk").write_text(tree_newick, encoding="utf-8")
            representative_groups = [
                tuple(sorted(label_to_index[label] for label in rows["label"]))
                for _, rows in membership.groupby("cluster", sort=True)
            ]
            groups = _expand(representative_groups, copies) + [(index,) for index in constant]

    if groups is None:
        if fallback is not None:
            groups, status = list(fallback), f"{status}; reused earlier DAMICORE groups"
        else:
            groups = _expand([(index,) for index in representatives], copies)
            groups += [(index,) for index in constant]
    validate_partition(groups)
    off_diagonal = (
        distances.to_numpy()[~np.eye(len(distances), dtype=bool)]
        if distances is not None
        else np.array([np.nan])
    )
    diagnostics = {
        "damicore_status": status,
        "mixes": len(promising),
        "bytes_per_series": max(len(values) for values in series),
        "constant_components": len(constant),
        "duplicate_components": COMPONENT_COUNT - len(constant) - len(representatives),
        "block_count": len(groups),
        "largest_block": max(len(group) for group in groups),
        "ncd_min": float(np.min(off_diagonal)),
        "ncd_median": float(np.median(off_diagonal)),
        "ncd_max": float(np.max(off_diagonal)),
        "blocks": [[COMPONENTS[index] for index in group] for group in groups],
    }
    (output_dir / "damicore_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    return groups, diagnostics


def damicore_niches(
    promising: list[Solution], output_dir: Path, config: ExperimentConfig
) -> tuple[list[list[int]], dict]:
    """Split the promising mixes into niches with DAMICORE over one file per mix.

    DAMICORE's own FastGreedy cut is kept when it gives at most ``max_niches`` niches of at
    least ``min_niche_size`` mixes; otherwise DAMICORE is asked for fewer clusters, one less
    at a time, until every niche is large enough. Returns positions in ``promising``.
    """
    everything = [list(range(len(promising)))]
    if len(promising) < 2 * config.min_niche_size:
        return everything, {"niche_status": "too_few_mixes", "damicore_niches": 1}
    documents_dir = output_dir / "mix_documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    for position, solution in enumerate(promising):
        (documents_dir / f"mix-{position:04d}.txt").write_bytes(mix_document(solution))

    def cut(num_clusters: int | None) -> list[list[int]]:
        name = "damicore_run" if num_clusters is None else f"damicore_run_k{num_clusters}"
        membership, _, _ = _run_damicore(documents_dir, output_dir / name, num_clusters)
        return [
            sorted(int(label[4:8]) for label in rows["label"])
            for _, rows in membership.groupby("cluster", sort=True)
        ]

    def acceptable(niches: list[list[int]]) -> bool:
        return len(niches) <= config.max_niches and min(map(len, niches)) >= config.min_niche_size

    try:
        niches = cut(None)
        found = len(niches)
        status = "fastgreedy"
        if not acceptable(niches):
            status = "tree_cut"
            for count in range(min(found, config.max_niches + 1) - 1, 1, -1):
                niches = cut(count)
                if acceptable(niches):
                    break
            else:
                niches, status = everything, "single_niche"
    except DamicoreError as error:
        return everything, {"niche_status": f"failed: {error}", "damicore_niches": 1}
    return niches, {"niche_status": status, "damicore_niches": found}


def fit_blocks(promising: list[Solution], groups: Groups) -> list[BlockModel]:
    validate_partition(groups)
    models = []
    for group in groups:
        counter = Counter(tuple(solution[index] for index in group) for solution in promising)
        patterns = sorted(counter)
        models.append(BlockModel(group, patterns, np.array([counter[p] for p in patterns])))
    return models


def table_records(models: list[BlockModel]) -> list[dict]:
    """Observed patterns of each block with their sampling probability."""
    records = []
    for block_id, model in enumerate(models, start=1):
        probabilities = model.counts / model.counts.sum()
        for pattern, count, probability in zip(model.patterns, model.counts, probabilities, strict=True):
            records.append(
                {
                    "block_id": block_id,
                    "components": ", ".join(COMPONENTS[index] for index in model.group),
                    "levels": "-".join(str(level) for level in pattern),
                    "promising_count": int(count),
                    "probability": float(probability),
                }
            )
    return records


def sample_solution(rng: np.random.Generator, models: list[BlockModel]) -> Solution:
    """Each block draws one of its observed patterns, in proportion to how often it was seen."""
    levels = [0] * COMPONENT_COUNT
    for model in models:
        pattern = model.patterns[int(rng.choice(len(model.patterns), p=model.counts / model.counts.sum()))]
        for index, level in zip(model.group, pattern, strict=True):
            levels[index] = int(level)
    return tuple(levels)


def _new_candidate(
    rng: np.random.Generator,
    seen: set[Solution],
    niches: list[NicheModel] | None,
    archive: list[BlockModel] | None,
    config: ExperimentConfig,
) -> tuple[Solution, str]:
    """A not-yet-evaluated mix: a random niche's model, then the others, then all promising."""
    if niches:
        first = int(rng.integers(len(niches)))
        others = [index for index in rng.permutation(len(niches)) if index != first]
        tries = [(niches[first].blocks, "niche_sample")]
        tries += [(niches[index].blocks, "other_niche_sample") for index in others]
        if archive is not None:
            tries.append((archive, "archive_sample"))
        for models, origin in tries:
            for _ in range(config.sampling_attempts):
                candidate = sample_solution(rng, models)
                if candidate not in seen:
                    return candidate, origin
    while True:
        candidate = random_solution(rng, config.levels)
        if candidate not in seen:
            return candidate, "uniform_fallback"


def run_search(
    seed: int,
    landscape: Landscape,
    artifact_root: Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> SearchResult:
    """One run: the seed's initial population, then generations of DAMICORE-model offspring."""
    rng = np.random.default_rng([seed, 1])
    evaluated: list[Solution] = []
    scores: list[dict] = []
    seen: set[Solution] = set()
    history: list[dict] = []
    table_rows: list[dict] = []
    block_rows: list[dict] = []
    niche_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    archive_groups: Groups | None = None

    def record(solution: Solution, generation: int, origin: str) -> None:
        score = landscape.evaluate(solution)
        evaluated.append(solution)
        scores.append(score)
        seen.add(solution)
        history.append(
            {
                "seed": seed,
                "generation": generation,
                "evaluation_index": len(evaluated),
                "solution": solution_string(solution),
                "origin": origin,
                **score,
            }
        )

    def log_groups(generation: int, scope: str, groups: Groups, diagnostics: dict) -> None:
        diagnostics_rows.append({"seed": seed, "generation": generation, "scope": scope, **diagnostics})
        block_rows.extend(
            {
                "seed": seed,
                "generation": generation,
                "scope": scope,
                "block_id": block_id,
                "components": ", ".join(COMPONENTS[index] for index in group),
                "component_count": len(group),
            }
            for block_id, group in enumerate(groups, start=1)
        )

    for solution in initial_population(seed, config):
        record(solution, 0, "initial_population")

    for generation in range(1, config.generations + 1):
        generation_dir = artifact_root / f"seed-{seed}" / f"generation-{generation:02d}"
        # 1. Promising mixes.
        promising, promising_scores = promising_solutions(evaluated, scores, config)
        # 2. DAMICORE groups the components over all promising mixes...
        archive_groups, diagnostics = damicore_groups(
            promising, generation_dir / "archive", fallback=archive_groups
        )
        log_groups(generation, "archive", archive_groups, diagnostics)
        archive = fit_blocks(promising, archive_groups)
        # 3. ...splits the mixes into niches...
        positions, niche_diagnostics = damicore_niches(promising, generation_dir / "niches", config)
        niches = []
        for niche_id, members in enumerate(positions, start=1):
            mixes = [promising[position] for position in members]
            # 4. ...and groups the components again inside each niche.
            if len(positions) == 1:
                groups, diagnostics = archive_groups, diagnostics
            else:
                groups, diagnostics = damicore_groups(
                    mixes, generation_dir / f"niche-{niche_id}", fallback=archive_groups
                )
            log_groups(generation, f"niche-{niche_id}", groups, diagnostics)
            niches.append(NicheModel(mixes, groups, fit_blocks(mixes, groups)))
            strength = [promising_scores[position]["strength_mpa"] for position in members]
            cost = [promising_scores[position]["cost_usd_m3"] for position in members]
            niche_rows.append(
                {
                    "seed": seed,
                    "generation": generation,
                    "niche_id": niche_id,
                    "niche_count": len(positions),
                    **niche_diagnostics,
                    "size": len(members),
                    "strength_min": float(min(strength)),
                    "strength_median": float(np.median(strength)),
                    "strength_max": float(max(strength)),
                    "cost_median": float(np.median(cost)),
                }
            )
            tables = table_records(niches[-1].blocks)
            table_rows.extend(
                {"seed": seed, "generation": generation, "niche_id": niche_id, **row} for row in tables
            )
            pd.DataFrame(tables).to_csv(
                generation_dir / f"probability_tables_niche-{niche_id}.csv", index=False
            )
        # 5. Sample new mixes -> 6. evaluate.
        for _ in range(config.offspring_per_generation):
            candidate, origin = _new_candidate(rng, seen, niches, archive, config)
            record(candidate, generation, origin)

    if len(history) != config.evaluations_per_run:
        raise AssertionError("Every run must use exactly the configured budget.")
    return SearchResult(seed, history, table_rows, block_rows, niche_rows, diagnostics_rows)
