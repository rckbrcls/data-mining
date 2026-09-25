"""Run and persist the independent emergency feature-selection experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .config import ARTIFACT_ROOT, DEFAULT_CONFIG, ExperimentConfig
from .data import PreparedData, prepare_data
from .model import (
    FULL_MASK, Mask, SolutionEvaluator, calculate_metrics, calibration_bins,
    encode_data, fit_and_predict, selected_features,
)
from .search import SearchResult, greedy_forward, run_search


@dataclass
class ExperimentResults:
    output_dir: Path
    audit: dict
    validation: pd.DataFrame
    test: pd.DataFrame
    history: pd.DataFrame
    clusters: pd.DataFrame
    probabilities: pd.DataFrame
    candidates: pd.DataFrame
    damicore_diagnostics: pd.DataFrame
    calibration: pd.DataFrame
    conclusion: str


def _row(method: str, seed: int | None, mask: Mask, metrics, acceptable: bool) -> dict:
    return {
        "method": method,
        "seed": seed,
        "mask": "".join(map(str, mask)),
        "selected_features": ", ".join(selected_features(mask)),
        "acceptable": acceptable,
        **metrics.as_dict(),
    }


def _conclusion(validation: pd.DataFrame, test: pd.DataFrame) -> str:
    grouped = validation.loc[validation.method == "damicore"].set_index("seed")
    independent = validation.loc[validation.method == "independent"].set_index("seed")
    grouped_test = test.loc[test.method == "damicore"]
    if not grouped_test.acceptable.all():
        return "inconclusive_final_quality"
    if not grouped.acceptable.all() or not independent.acceptable.all():
        return "inconclusive_validation_quality"
    differences = independent.feature_count - grouped.feature_count
    if (differences >= 0).all() and (differences > 0).sum() >= 2:
        return "exploratory_support"
    if (differences <= 0).all():
        return "no_observed_gain"
    return "inconclusive_mixed_results"


def execute_experiment(
    prepared: PreparedData,
    config: ExperimentConfig = DEFAULT_CONFIG,
    output_root: Path = ARTIFACT_ROOT,
) -> ExperimentResults:
    """Compare searches on fixed samples and refit winners before the held-out test."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    evaluator = SolutionEvaluator(prepared.search_train, prepared.validation, config)
    greedy_mask = greedy_forward(evaluator)
    searches: list[SearchResult] = []
    for seed in config.seeds:
        for method in ("random", "independent", "damicore"):
            searches.append(run_search(method, seed, evaluator, output_dir, config))

    validation_rows = [
        _row("all_fields", None, FULL_MASK, evaluator.reference, True),
        _row("greedy_forward", None, greedy_mask, evaluator.evaluate(greedy_mask),
             evaluator.is_acceptable(greedy_mask)),
    ]
    validation_rows.extend(
        _row(result.method, result.seed, result.best_mask,
             evaluator.evaluate(result.best_mask), evaluator.is_acceptable(result.best_mask))
        for result in searches
    )
    validation = pd.DataFrame(validation_rows)

    final_encoded = encode_data(prepared.final_train, prepared.test)
    final_masks = {FULL_MASK, greedy_mask, *(result.best_mask for result in searches)}
    final_probabilities = {
        mask: fit_and_predict(final_encoded, mask) for mask in final_masks
    }
    final_metrics = {
        mask: calculate_metrics(
            final_encoded.evaluation_target, probabilities, sum(mask), config
        )
        for mask, probabilities in final_probabilities.items()
    }
    full_test = final_metrics[FULL_MASK]

    def test_acceptable(mask: Mask) -> bool:
        metric = final_metrics[mask]
        return (
            metric.log_loss <= full_test.log_loss * (1 + config.loss_tolerance)
            and metric.recall_at_capacity >= full_test.recall_at_capacity - config.recall_tolerance
        )

    test_rows = [
        _row("all_fields", None, FULL_MASK, full_test, True),
        _row("greedy_forward", None, greedy_mask, final_metrics[greedy_mask],
             test_acceptable(greedy_mask)),
    ]
    test_rows.extend(
        _row(result.method, result.seed, result.best_mask,
             final_metrics[result.best_mask], test_acceptable(result.best_mask))
        for result in searches
    )
    test = pd.DataFrame(test_rows)
    history = pd.DataFrame(row for result in searches for row in result.history)
    clusters = pd.DataFrame(row for result in searches for row in result.clusters)
    probabilities = pd.DataFrame(row for result in searches for row in result.probabilities)
    candidates = pd.DataFrame(row for result in searches for row in result.candidates)
    diagnostic_rows = []
    for seed in config.seeds:
        for generation in range(1, config.generations + 1):
            path = output_dir / f"seed-{seed}" / f"generation-{generation:02d}" / "damicore_summary.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            diagnostic_rows.append({
                "seed": seed,
                "generation": generation,
                "elite_size": report["elite_size"],
                "file_count": report["file_count"],
                "bytes_per_file": report["bytes_per_file"],
                "ncd_min": report["ncd_min"],
                "ncd_max": report["ncd_max"],
                "ncd_out_of_range_count": report["ncd_out_of_range_count"],
                "negative_branch_count": report["negative_branch_count"],
                "cluster_count": report["cluster_count"],
                "warnings": " | ".join(report["warnings"]),
            })
    damicore_diagnostics = pd.DataFrame(diagnostic_rows)
    calibration_rows = []
    for row in test_rows:
        mask = tuple(int(bit) for bit in row["mask"])
        bins = calibration_bins(
            final_encoded.evaluation_target, final_probabilities[mask]
        )
        calibration_rows.extend(
            {"method": row["method"], "seed": row["seed"], **bin_row}
            for bin_row in bins.to_dict("records")
        )
    calibration = pd.DataFrame(calibration_rows)

    if any(len(result.evaluated_masks) != config.evaluations_per_method for result in searches):
        raise AssertionError("All search methods must have the same solution budget.")
    for seed in config.seeds:
        first_pools = [
            result.evaluated_masks[: config.initial_pool_size]
            for result in searches if result.seed == seed
        ]
        if not all(pool == first_pools[0] for pool in first_pools):
            raise AssertionError("All methods must share the same initial solutions.")

    conclusion = _conclusion(validation, test)
    audit = {
        **prepared.audit,
        "config": asdict(config),
        "selection_reference": "all_fields_validation",
        "final_reference": "all_fields_held_out_test",
        "conclusion": conclusion,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prepared.monthly_counts.to_csv(output_dir / "monthly_counts.csv", index=False)
    validation.to_csv(output_dir / "validation_summary.csv", index=False)
    test.to_csv(output_dir / "test_summary.csv", index=False)
    history.to_csv(output_dir / "search_history.csv", index=False)
    clusters.to_csv(output_dir / "damicore_blocks.csv", index=False)
    probabilities.to_csv(output_dir / "damicore_probabilities.csv", index=False)
    candidates.to_csv(output_dir / "damicore_candidates.csv", index=False)
    damicore_diagnostics.to_csv(output_dir / "damicore_diagnostics.csv", index=False)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)

    return ExperimentResults(
        output_dir, audit, validation, test, history, clusters,
        probabilities, candidates, damicore_diagnostics, calibration, conclusion,
    )


def run_experiment(
    database_url: str | None = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
    output_root: Path = ARTIFACT_ROOT,
) -> ExperimentResults:
    return execute_experiment(prepare_data(database_url, config), config, output_root)
