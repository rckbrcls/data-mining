"""Quality objective of a field subset, and the exhaustive ground truth used as a check.

A subset of fields partitions the reports into cells (one cell per observed combination of
values). The training period gives each cell an emergency rate, shrunk toward the base rate;
the quality of the subset is how much those rates reduce the log-loss on the later
validation period, expressed in bits per report. This is an out-of-sample estimate of the
mutual information between the chosen fields and the target, so sparse combinations that
only memorize the training period are penalized automatically.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, FEATURES, ExperimentConfig

if TYPE_CHECKING:
    from .data import ValidationFold


Mask = tuple[int, ...]
FEATURE_COUNT = len(FEATURES)
FULL_MASK: Mask = (1,) * FEATURE_COUNT
EPSILON = 1e-12


def validate_mask(mask: Sequence[int]) -> Mask:
    if len(mask) != FEATURE_COUNT or any(bit not in (0, 1) for bit in mask):
        raise ValueError(f"A candidate must contain {FEATURE_COUNT} binary decisions.")
    if not any(mask):
        raise ValueError("A candidate must include at least one feature.")
    return tuple(int(bit) for bit in mask)


def mask_string(mask: Sequence[int]) -> str:
    return "".join(str(bit) for bit in validate_mask(mask))


def mask_from_string(value: str) -> Mask:
    return validate_mask(tuple(int(bit) for bit in value))


def selected_features(mask: Sequence[int]) -> list[str]:
    return [
        feature
        for included, feature in zip(validate_mask(mask), FEATURES, strict=True)
        if included
    ]


def all_masks() -> list[Mask]:
    return [
        tuple((index >> position) & 1 for position in range(FEATURE_COUNT))
        for index in range(1, 2**FEATURE_COUNT)
    ]


@dataclass(frozen=True)
class CodedFold:
    """One temporal fold with every field coded as integers learned from training only."""

    name: str
    codes: np.ndarray  # (train + validation rows, features); unseen values share one code
    level_counts: tuple[int, ...]
    train_rows: int
    train_target: np.ndarray
    validation_target: np.ndarray

    @property
    def base_rate(self) -> float:
        return float(self.train_target.mean())


def code_fold(name: str, train: pd.DataFrame, validation: pd.DataFrame) -> CodedFold:
    columns: list[np.ndarray] = []
    level_counts: list[int] = []
    for feature in FEATURES:
        categories = pd.Index(sorted(train[feature].astype(str).unique()))
        train_codes = categories.get_indexer(train[feature].astype(str))
        validation_codes = categories.get_indexer(validation[feature].astype(str))
        unseen = len(categories)
        validation_codes = np.where(validation_codes < 0, unseen, validation_codes)
        columns.append(np.concatenate([train_codes, validation_codes]).astype(np.int64))
        level_counts.append(unseen + 1)
    return CodedFold(
        name=name,
        codes=np.column_stack(columns),
        level_counts=tuple(level_counts),
        train_rows=len(train),
        train_target=train["target"].to_numpy(dtype=np.float64),
        validation_target=validation["target"].to_numpy(dtype=np.float64),
    )


def code_folds(folds: Sequence[ValidationFold]) -> list[CodedFold]:
    return [code_fold(fold.name, fold.train, fold.validation) for fold in folds]


def extend_cells(
    cells: np.ndarray | None, fold: CodedFold, feature_index: int
) -> np.ndarray:
    """Refine a cell partition by one more field, keeping cell ids compact."""
    column = fold.codes[:, feature_index]
    if cells is None:
        return column
    combined = cells * fold.level_counts[feature_index] + column
    return pd.factorize(combined, sort=False)[0].astype(np.int64)


def _log_loss(target: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return float(
        -np.mean(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities))
    )


def cell_probabilities(
    cells: np.ndarray, fold: CodedFold, prior_strength: float
) -> np.ndarray:
    """Shrunk training emergency rate of each validation report's cell."""
    train_cells = cells[: fold.train_rows]
    validation_cells = cells[fold.train_rows :]
    cell_count = int(cells.max()) + 1
    counts = np.bincount(train_cells, minlength=cell_count)
    positives = np.bincount(train_cells, weights=fold.train_target, minlength=cell_count)
    base = fold.base_rate
    rates = (positives + prior_strength * base) / (counts + prior_strength)
    return rates[validation_cells]


def information_gain_bits(
    cells: np.ndarray, fold: CodedFold, prior_strength: float
) -> float:
    target = fold.validation_target
    base_loss = _log_loss(target, np.full(len(target), fold.base_rate))
    cell_loss = _log_loss(target, cell_probabilities(cells, fold, prior_strength))
    return (base_loss - cell_loss) / log(2)


def mask_cells(mask: Sequence[int], fold: CodedFold) -> np.ndarray:
    cells: np.ndarray | None = None
    for feature_index, included in enumerate(validate_mask(mask)):
        if included:
            cells = extend_cells(cells, fold, feature_index)
    assert cells is not None
    return cells


def score_mask(
    mask: Sequence[int],
    folds: Sequence[CodedFold],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    return {
        fold.name: information_gain_bits(mask_cells(mask, fold), fold, config.prior_strength)
        for fold in folds
    }


# Exhaustive enumeration --------------------------------------------------------------
#
# Every subset is reached from its parent (the subset without its highest field) by one
# extend_cells call, so the 32,767 subsets cost 32,767 refinements per fold. Work is split
# by fold and by the lowest field of the subset; worker globals avoid re-sending the data.

_WORKER_FOLDS: list[CodedFold] = []
_WORKER_PRIOR = DEFAULT_CONFIG.prior_strength


def _initialize_worker(folds: list[CodedFold], prior_strength: float) -> None:
    global _WORKER_FOLDS, _WORKER_PRIOR
    _WORKER_FOLDS = folds
    _WORKER_PRIOR = prior_strength


def _subtree_scores(
    fold: CodedFold, root: int, prior_strength: float
) -> list[tuple[int, float]]:
    """Scores for every subset whose lowest field is ``root``, as (bitmask, bits)."""
    results: list[tuple[int, float]] = []
    # Children hold a reference to the parent's cells and are refined only when popped,
    # so memory stays at one partition per depth level.
    stack: list[tuple[int, int, np.ndarray | None]] = [(1 << root, root, None)]
    while stack:
        bitmask, added, parent_cells = stack.pop()
        cells = extend_cells(parent_cells, fold, added)
        results.append((bitmask, information_gain_bits(cells, fold, prior_strength)))
        for feature_index in range(added + 1, FEATURE_COUNT):
            stack.append((bitmask | (1 << feature_index), feature_index, cells))
    return results


def _worker_task(task: tuple[int, int]) -> tuple[int, list[tuple[int, float]]]:
    fold_index, root = task
    return fold_index, _subtree_scores(_WORKER_FOLDS[fold_index], root, _WORKER_PRIOR)


def exhaustive_scores(
    folds: Sequence[CodedFold],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Information gain of all 2^15 - 1 field subsets on every validation fold."""
    tasks = [
        (fold_index, root)
        for root in range(FEATURE_COUNT)
        for fold_index in range(len(folds))
    ]
    by_fold: dict[int, dict[int, float]] = {index: {} for index in range(len(folds))}
    if config.exhaustive_workers <= 1:
        _initialize_worker(list(folds), config.prior_strength)
        outputs = map(_worker_task, tasks)
        for fold_index, rows in outputs:
            by_fold[fold_index].update(rows)
    else:
        with ProcessPoolExecutor(
            max_workers=config.exhaustive_workers,
            initializer=_initialize_worker,
            initargs=(list(folds), config.prior_strength),
        ) as pool:
            for fold_index, rows in pool.map(_worker_task, tasks):
                by_fold[fold_index].update(rows)

    bitmasks = sorted(by_fold[0])
    if len(bitmasks) != 2**FEATURE_COUNT - 1:
        raise AssertionError("The exhaustive enumeration missed some subsets.")
    table = pd.DataFrame({"bitmask": bitmasks})
    table["mask"] = [
        "".join(str((bitmask >> position) & 1) for position in range(FEATURE_COUNT))
        for bitmask in bitmasks
    ]
    table["feature_count"] = [bin(bitmask).count("1") for bitmask in bitmasks]
    fold_columns = []
    for fold_index, fold in enumerate(folds):
        column = f"gain_bits_{fold.name}"
        table[column] = [by_fold[fold_index][bitmask] for bitmask in bitmasks]
        fold_columns.append(column)
    table["mean_gain_bits"] = table[fold_columns].mean(axis=1)
    table["worst_gain_bits"] = table[fold_columns].min(axis=1)
    return table.drop(columns="bitmask")


def folds_fingerprint(folds: Sequence[CodedFold], config: ExperimentConfig) -> str:
    digest = hashlib.sha256()
    digest.update(repr((FEATURES, config.prior_strength)).encode())
    for fold in folds:
        digest.update(fold.name.encode())
        digest.update(np.ascontiguousarray(fold.codes).tobytes())
        digest.update(fold.train_target.tobytes())
        digest.update(fold.validation_target.tobytes())
    return digest.hexdigest()[:16]


def cached_exhaustive_scores(
    folds: Sequence[CodedFold],
    cache_dir: Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, bool]:
    """Reuse a previous enumeration of the same data and prior; returns (table, from_cache)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"exhaustive_{folds_fingerprint(folds, config)}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype={"mask": str}), True
    table = exhaustive_scores(folds, config)
    table.to_csv(cache_path, index=False)
    return table, False


class MaskEvaluator:
    """Scores a mask directly on the validation folds the first time the AED asks for it."""

    def __init__(
        self, folds: Sequence[CodedFold], config: ExperimentConfig = DEFAULT_CONFIG
    ) -> None:
        self._folds = list(folds)
        self._prior_strength = config.prior_strength
        self._scores: dict[Mask, tuple[float, float]] = {}

    def _score(self, mask: Mask) -> tuple[float, float]:
        mask = validate_mask(mask)
        if mask not in self._scores:
            gains = [
                information_gain_bits(mask_cells(mask, fold), fold, self._prior_strength)
                for fold in self._folds
            ]
            self._scores[mask] = (float(np.mean(gains)), float(min(gains)))
        return self._scores[mask]

    def quality(self, mask: Mask) -> float:
        return self._score(mask)[0]

    def worst_fold(self, mask: Mask) -> float:
        return self._score(mask)[1]
