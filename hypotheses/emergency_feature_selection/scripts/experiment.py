"""Orchestrate the hypothesis: DAMICORE-guided AED runs, then the ground-truth check."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ARTIFACT_ROOT,
    DEFAULT_CONFIG,
    FEATURES,
    NEGATIVE_STATUS,
    POSITIVE_STATUSES,
    ExperimentConfig,
)
from .data import PreparedData
from .fitness import (
    FULL_MASK,
    CodedFold,
    Mask,
    MaskEvaluator,
    cached_exhaustive_scores,
    code_fold,
    code_folds,
    information_gain_bits,
    mask_cells,
    mask_from_string,
    mask_string,
    selected_features,
)
from .pareto import pareto_front
from .search import SearchResult, run_search


@dataclass
class ExperimentResults:
    output_dir: Path
    audit: dict
    search_history: pd.DataFrame
    generation_progress: pd.DataFrame
    probability_tables: pd.DataFrame
    damicore_blocks: pd.DataFrame
    damicore_diagnostics: pd.DataFrame
    block_cooccurrence: pd.DataFrame
    aed_front: pd.DataFrame
    recommended_mask: Mask
    seed_recommendations: pd.DataFrame
    exhaustive: pd.DataFrame
    true_front: pd.DataFrame
    held_out: pd.DataFrame


def _with_field_names(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(
        fields=table["mask"].map(lambda value: ", ".join(selected_features(mask_from_string(value))))
    )


def recommended_from_front(front: pd.DataFrame, tolerance_fraction: float) -> Mask:
    """Fewest fields whose quality reaches ``tolerance_fraction`` of the front's best."""
    threshold = tolerance_fraction * front["mean_gain_bits"].max()
    eligible = front.loc[front["mean_gain_bits"] >= threshold].sort_values("feature_count")
    return mask_from_string(eligible.iloc[0]["mask"])


# Each AED run evaluates masks directly on the data; runs are independent, so seeds are
# spread over processes that each hold the coded folds once.
_WORKER_EVALUATOR: MaskEvaluator | None = None


def _initialize_search_worker(folds: list[CodedFold], config: ExperimentConfig) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = MaskEvaluator(folds, config)


def _search_task(task: tuple[int, Path, ExperimentConfig]) -> SearchResult:
    seed, artifact_root, config = task
    assert _WORKER_EVALUATOR is not None
    return run_search(seed, _WORKER_EVALUATOR, artifact_root, config)


def run_searches(
    folds: list[CodedFold], artifact_root: Path, config: ExperimentConfig
) -> list[SearchResult]:
    tasks = [(seed, artifact_root, config) for seed in config.seeds]
    workers = min(len(tasks), config.exhaustive_workers)
    if workers <= 1:
        _initialize_search_worker(folds, config)
        return [_search_task(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_search_worker,
        initargs=(folds, config),
    ) as pool:
        return list(pool.map(_search_task, tasks))


def _generation_progress(history: pd.DataFrame) -> pd.DataFrame:
    """Per run and generation: mean quality of the new masks and best quality so far."""
    by_generation = (
        history.groupby(["seed", "generation"])["mean_gain_bits"]
        .agg(new_masks_mean_gain="mean", new_masks_best_gain="max")
        .reset_index()
    )
    by_generation["best_gain_so_far"] = by_generation.groupby("seed")[
        "new_masks_best_gain"
    ].cummax()
    return by_generation


def _block_cooccurrence(blocks: pd.DataFrame) -> pd.DataFrame:
    """Share of DAMICORE generations (all seeds) in which two fields share a block."""
    position = {feature: index for index, feature in enumerate(FEATURES)}
    counts = np.zeros((len(FEATURES), len(FEATURES)))
    generations = blocks.groupby(["seed", "generation"])
    for _, generation_blocks in generations:
        for features in generation_blocks["features"]:
            members = [position[name] for name in features.split(", ")]
            for left, right in combinations(members, 2):
                counts[left, right] += 1
                counts[right, left] += 1
    counts /= max(1, generations.ngroups)
    np.fill_diagonal(counts, 1.0)
    return pd.DataFrame(counts, index=list(FEATURES), columns=list(FEATURES))


def _held_out(
    prepared: PreparedData, masks: list[tuple[str, Mask]], config: ExperimentConfig
) -> pd.DataFrame:
    """Same quality measure on the reserved period, trained on everything before it."""
    coded = code_fold(prepared.profile.test_name, prepared.final_train, prepared.test)
    return pd.DataFrame(
        {
            "form": name,
            "feature_count": sum(mask),
            "fields": ", ".join(selected_features(mask)),
            "test_period": prepared.profile.test_name,
            "gain_bits": information_gain_bits(
                mask_cells(mask, coded), coded, config.prior_strength
            ),
        }
        for name, mask in masks
    )


def execute_experiment(
    prepared: PreparedData,
    config: ExperimentConfig = DEFAULT_CONFIG,
    output_root: Path = ARTIFACT_ROOT,
) -> ExperimentResults:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{prepared.profile.name}-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    folds = code_folds(prepared.folds)

    # 1. The AED guided by DAMICORE, one run per seed.
    searches = run_searches(folds, output_dir / "searches", config)
    history = pd.DataFrame(row for search in searches for row in search.history)
    progress = _generation_progress(history)
    probability_tables = pd.DataFrame(row for s in searches for row in s.probability_tables)
    blocks = pd.DataFrame(row for s in searches for row in s.blocks)
    diagnostics = pd.DataFrame(row for s in searches for row in s.diagnostics)
    cooccurrence = _block_cooccurrence(blocks)

    # 2. Recommendation from what the AED found (all runs pooled, and per run).
    aed_front = _with_field_names(pareto_front(history.drop_duplicates("mask")))
    recommended = recommended_from_front(aed_front, config.tolerance_fraction)

    # 3. Check against the ground truth: every one of the 32,767 subsets scored.
    exhaustive, from_cache = cached_exhaustive_scores(folds, output_root / "cache", config)
    true_front = _with_field_names(pareto_front(exhaustive))
    true_recommended = recommended_from_front(true_front, config.tolerance_fraction)
    true_masks = set(true_front["mask"])
    seed_rows = []
    for seed, run_history in history.groupby("seed"):
        seed_front = pareto_front(run_history)
        seed_mask = recommended_from_front(seed_front, config.tolerance_fraction)
        seed_rows.append(
            {
                "seed": seed,
                "evaluations": len(run_history),
                "recommended_fields": ", ".join(selected_features(seed_mask)),
                "matches_ground_truth": seed_mask == true_recommended,
                "true_front_points_found": len(true_masks.intersection(run_history["mask"])),
                "true_front_size": len(true_masks),
            }
        )
    seed_recommendations = pd.DataFrame(seed_rows)

    # 4. The recommended form on the reserved period, next to all fields.
    held_out = _held_out(
        prepared, [("recommended", recommended), ("all_fields", FULL_MASK)], config
    )

    audit = {
        **prepared.audit,
        "config": asdict(config),
        "evaluations_per_run": config.evaluations_per_seed,
        "share_of_all_forms_per_run": config.evaluations_per_seed / len(exhaustive),
        "recommended_mask": mask_string(recommended),
        "recommended_fields": selected_features(recommended),
        "ground_truth_recommended_fields": selected_features(true_recommended),
        "recommended_matches_ground_truth": recommended == true_recommended,
        "runs_matching_ground_truth": int(seed_recommendations["matches_ground_truth"].sum()),
        "exhaustive_from_cache": from_cache,
        "positive_statuses": list(POSITIVE_STATUSES),
        "negative_status": NEGATIVE_STATUS,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    history.to_csv(output_dir / "search_history.csv", index=False)
    progress.to_csv(output_dir / "generation_progress.csv", index=False)
    probability_tables.to_csv(output_dir / "probability_tables.csv", index=False)
    blocks.to_csv(output_dir / "damicore_blocks.csv", index=False)
    diagnostics.to_csv(output_dir / "damicore_diagnostics.csv", index=False)
    cooccurrence.to_csv(output_dir / "block_cooccurrence.csv")
    aed_front.to_csv(output_dir / "aed_pareto_front.csv", index=False)
    seed_recommendations.to_csv(output_dir / "seed_recommendations.csv", index=False)
    true_front.to_csv(output_dir / "true_pareto_front.csv", index=False)
    held_out.to_csv(output_dir / "held_out.csv", index=False)
    prepared.monthly_counts.to_csv(output_dir / "monthly_counts.csv", index=False)

    return ExperimentResults(
        output_dir=output_dir,
        audit=audit,
        search_history=history,
        generation_progress=progress,
        probability_tables=probability_tables,
        damicore_blocks=blocks,
        damicore_diagnostics=diagnostics,
        block_cooccurrence=cooccurrence,
        aed_front=aed_front,
        recommended_mask=recommended,
        seed_recommendations=seed_recommendations,
        exhaustive=exhaustive,
        true_front=true_front,
        held_out=held_out,
    )
