"""Comparable probabilistic scoring for every feature-subset solution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import OneHotEncoder

from .config import DEFAULT_CONFIG, FEATURES, ExperimentConfig


Mask = tuple[int, ...]
FULL_MASK: Mask = (1,) * len(FEATURES)


@dataclass(frozen=True)
class ModelMetrics:
    log_loss: float
    brier_score: float
    average_precision: float
    recall_at_capacity: float
    calibration_error: float
    feature_count: int
    report_count: int
    emergency_count: int
    reviewed_count: int
    emergencies_reviewed: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class EncodedData:
    train_matrix: sparse.csr_matrix
    evaluation_matrix: sparse.csr_matrix
    train_target: np.ndarray
    evaluation_target: np.ndarray
    feature_columns: tuple[tuple[int, ...], ...]


def validate_mask(mask: Sequence[int]) -> Mask:
    if len(mask) != len(FEATURES) or any(bit not in (0, 1) for bit in mask):
        raise ValueError(f"A solution must contain {len(FEATURES)} binary decisions.")
    if not any(mask):
        raise ValueError("A solution must include at least one feature.")
    return tuple(int(bit) for bit in mask)


def encode_data(train: pd.DataFrame, evaluation: pd.DataFrame) -> EncodedData:
    """Fit category mappings on training reports only."""
    train_blocks: list[sparse.csr_matrix] = []
    evaluation_blocks: list[sparse.csr_matrix] = []
    feature_columns: list[tuple[int, ...]] = []
    next_column = 0
    for feature in FEATURES:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        train_block = encoder.fit_transform(train[[feature]]).tocsr()
        evaluation_block = encoder.transform(evaluation[[feature]]).tocsr()
        train_blocks.append(train_block)
        evaluation_blocks.append(evaluation_block)
        feature_columns.append(tuple(range(next_column, next_column + train_block.shape[1])))
        next_column += train_block.shape[1]
    return EncodedData(
        train_matrix=sparse.hstack(train_blocks, format="csr"),
        evaluation_matrix=sparse.hstack(evaluation_blocks, format="csr"),
        train_target=train["target"].to_numpy(dtype=np.int8),
        evaluation_target=evaluation["target"].to_numpy(dtype=np.int8),
        feature_columns=tuple(feature_columns),
    )


def recall_at_capacity(
    target: np.ndarray, probabilities: np.ndarray, top_fraction: float
) -> float:
    positives = int(target.sum())
    if positives == 0:
        raise ValueError("Recall is undefined when no urgent reports are present.")
    _, found = capacity_counts(target, probabilities, top_fraction)
    return found / positives


def capacity_counts(
    target: np.ndarray, probabilities: np.ndarray, top_fraction: float
) -> tuple[int, int]:
    reviewed = max(1, ceil(len(target) * top_fraction))
    top_indices = np.argsort(-probabilities, kind="stable")[:reviewed]
    return reviewed, int(target[top_indices].sum())


def calibration_error(target: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the prevalence-weighted gap across ten fixed probability bins."""
    bins = np.minimum((probabilities * 10).astype(int), 9)
    total = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            total += (selected.sum() / len(target)) * abs(
                float(target[selected].mean()) - float(probabilities[selected].mean())
            )
    return total


def calculate_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    feature_count: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> ModelMetrics:
    reviewed_count, emergencies_reviewed = capacity_counts(
        target, probabilities, config.top_fraction
    )
    return ModelMetrics(
        log_loss=float(log_loss(target, probabilities, labels=[0, 1])),
        brier_score=float(brier_score_loss(target, probabilities)),
        average_precision=float(average_precision_score(target, probabilities)),
        recall_at_capacity=recall_at_capacity(target, probabilities, config.top_fraction),
        calibration_error=calibration_error(target, probabilities),
        feature_count=feature_count,
        report_count=len(target),
        emergency_count=int(target.sum()),
        reviewed_count=reviewed_count,
        emergencies_reviewed=emergencies_reviewed,
    )


def calibration_bins(target: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Keep only aggregate counts and means for a reliability diagram."""
    assignments = np.minimum((probabilities * 10).astype(int), 9)
    rows = []
    for index in range(10):
        selected = assignments == index
        if selected.any():
            rows.append({
                "bin": index,
                "report_count": int(selected.sum()),
                "predicted_mean": float(probabilities[selected].mean()),
                "observed_rate": float(target[selected].mean()),
            })
    return pd.DataFrame(rows)


def fit_and_predict(
    encoded: EncodedData, mask: Sequence[int]
) -> np.ndarray:
    """Fit on historical reports and return held-out emergency probabilities."""
    selected = validate_mask(mask)
    columns = [
        column
        for enabled, feature_columns in zip(selected, encoded.feature_columns, strict=True)
        if enabled
        for column in feature_columns
    ]
    model = LogisticRegression(solver="liblinear", max_iter=200, random_state=0)
    model.fit(encoded.train_matrix[:, columns], encoded.train_target)
    return model.predict_proba(encoded.evaluation_matrix[:, columns])[:, 1]


def fit_and_score(
    encoded: EncodedData,
    mask: Sequence[int],
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> ModelMetrics:
    selected = validate_mask(mask)
    probabilities = fit_and_predict(encoded, selected)
    return calculate_metrics(
        encoded.evaluation_target, probabilities, sum(selected), config
    )


def selected_features(mask: Sequence[int]) -> list[str]:
    selected = validate_mask(mask)
    return [feature for feature, enabled in zip(FEATURES, selected, strict=True) if enabled]


class SolutionEvaluator:
    def __init__(
        self, train: pd.DataFrame, validation: pd.DataFrame,
        config: ExperimentConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        self.encoded = encode_data(train, validation)
        self.cache: dict[Mask, ModelMetrics] = {}
        self.reference = self.evaluate(FULL_MASK)

    def evaluate(self, mask: Sequence[int]) -> ModelMetrics:
        selected = validate_mask(mask)
        if selected not in self.cache:
            self.cache[selected] = fit_and_score(self.encoded, selected, self.config)
        return self.cache[selected]

    def is_acceptable(self, mask: Sequence[int]) -> bool:
        result = self.evaluate(mask)
        return (
            result.log_loss <= self.reference.log_loss * (1 + self.config.loss_tolerance)
            and result.recall_at_capacity
            >= self.reference.recall_at_capacity - self.config.recall_tolerance
        )

    def rank(self, mask: Sequence[int]) -> tuple[float, ...]:
        selected = validate_mask(mask)
        result = self.evaluate(selected)
        loss_violation = max(
            0.0,
            result.log_loss / self.reference.log_loss - (1 + self.config.loss_tolerance),
        )
        recall_violation = max(
            0.0,
            self.reference.recall_at_capacity
            - self.config.recall_tolerance
            - result.recall_at_capacity,
        )
        if loss_violation == 0 and recall_violation == 0:
            return (0.0, float(sum(selected)), result.log_loss)
        return (1.0, loss_violation + recall_violation, float(sum(selected)), result.log_loss)
