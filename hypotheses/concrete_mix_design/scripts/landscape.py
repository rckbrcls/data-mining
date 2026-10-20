"""Data, discretized search space, strength surrogate, and the scored landscape.

A solution is one level index per component. Its two objectives are the 28-day strength
predicted by a gradient-boosting model trained on the UCI laboratory tests, and the
material cost. Mixes far from every real mix are infeasible: the model would be
extrapolating there.

Because the space is small enough (8^7 ≈ 2.1M mixes), every mix is scored once and cached.
The searches only read the mixes they evaluate; the full table is the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_validate

from .config import (
    AGE_COLUMN,
    COMPONENTS,
    DATA_PATH,
    PRICES_USD_PER_KG,
    TARGET_COLUMN,
    ExperimentConfig,
)

Solution = tuple[int, ...]
COMPONENT_COUNT = len(COMPONENTS)
# More water lowers strength; more cement and more age raise it. Other effects are free.
MONOTONIC = {"cement": 1, "water": -1, AGE_COLUMN: 1}


def load_tests(path: Path = DATA_PATH) -> pd.DataFrame:
    tests = pd.read_csv(path)
    expected = [*COMPONENTS, AGE_COLUMN, TARGET_COLUMN]
    if list(tests.columns) != expected:
        raise ValueError(f"Unexpected columns {list(tests.columns)}; expected {expected}.")
    return tests


def unique_mixes(tests: pd.DataFrame) -> np.ndarray:
    return tests[list(COMPONENTS)].drop_duplicates().to_numpy(dtype=float)


def component_levels(mixes: np.ndarray, levels: int) -> list[np.ndarray]:
    """Quantile values of the real mixes; components that are often absent keep 0 as a level."""
    result = []
    for column in range(mixes.shape[1]):
        values = mixes[:, column]
        if (values == 0).mean() > 0.1:
            positive = values[values > 0]
            middle = (np.arange(levels - 1) + 0.5) / (levels - 1)
            grid = np.r_[0.0, np.quantile(positive, middle)]
        else:
            grid = np.quantile(values, (np.arange(levels) + 0.5) / levels)
        grid = np.round(grid, 1)
        if len(np.unique(grid)) != levels:
            raise ValueError(f"{COMPONENTS[column]} levels are not distinct: {grid}")
        result.append(grid)
    return result


def mix_cost(mixes: np.ndarray) -> np.ndarray:
    prices = np.array([PRICES_USD_PER_KG[name] for name in COMPONENTS])
    return mixes @ prices


def strength_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        random_state=0,
        monotonic_cst=[MONOTONIC.get(name, 0) for name in (*COMPONENTS, AGE_COLUMN)],
    )


def surrogate_validation(tests: pd.DataFrame) -> dict:
    """Cross-validation where all ages of one mix stay in the same fold."""
    features = tests[[*COMPONENTS, AGE_COLUMN]]
    groups = tests[list(COMPONENTS)].astype(str).agg("|".join, axis=1)
    scores = cross_validate(
        strength_model(),
        features,
        tests[TARGET_COLUMN],
        cv=GroupKFold(n_splits=5),
        groups=groups,
        scoring=("r2", "neg_mean_absolute_error"),
    )
    return {
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_folds": [float(value) for value in scores["test_r2"]],
        "cv_mae_mpa": float(-scores["test_neg_mean_absolute_error"].mean()),
    }


@dataclass
class Landscape:
    """Every mix of the grid with its strength, cost, and distance to the real data."""

    levels: list[np.ndarray]
    strength: np.ndarray
    cost: np.ndarray
    distance: np.ndarray
    max_distance: float

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.levels)

    @property
    def feasible(self) -> np.ndarray:
        return self.distance <= self.max_distance

    def index(self, solution: Solution) -> int:
        return int(np.ravel_multi_index(solution, self.shape))

    def mix(self, solution: Solution) -> dict[str, float]:
        return {name: float(self.levels[j][solution[j]]) for j, name in enumerate(COMPONENTS)}

    def evaluate(self, solution: Solution) -> dict:
        position = self.index(solution)
        distance = float(self.distance[position])
        return {
            "strength_mpa": float(self.strength[position]),
            "cost_usd_m3": float(self.cost[position]),
            "distance_to_data": distance,
            "violation": max(0.0, distance - self.max_distance),
        }


def _grid_mixes(levels: list[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*levels, indexing="ij")
    return np.stack([axis.ravel() for axis in mesh], axis=1)


def score_grid(
    tests: pd.DataFrame, levels: list[np.ndarray], config: ExperimentConfig
) -> Landscape:
    model = strength_model().fit(tests[[*COMPONENTS, AGE_COLUMN]], tests[TARGET_COLUMN])
    mixes = _grid_mixes(levels)
    frame = pd.DataFrame(mixes, columns=list(COMPONENTS))
    frame[AGE_COLUMN] = config.design_age_days
    strength = model.predict(frame)

    real = unique_mixes(tests)
    center, scale = real.mean(axis=0), real.std(axis=0)
    distance, _ = cKDTree((real - center) / scale).query((mixes - center) / scale)
    return Landscape(
        levels=levels,
        strength=strength.astype(np.float32),
        cost=mix_cost(mixes).astype(np.float32),
        distance=distance.astype(np.float32),
        max_distance=config.max_distance_to_data,
    )


def _cache_key(tests: pd.DataFrame, config: ExperimentConfig) -> str:
    payload = {
        "data": hashlib.sha256(pd.util.hash_pandas_object(tests).to_numpy().tobytes()).hexdigest(),
        "levels": config.levels,
        "age": config.design_age_days,
        "prices": PRICES_USD_PER_KG,
        "model": strength_model().get_params(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_landscape(
    tests: pd.DataFrame, config: ExperimentConfig, cache_dir: Path
) -> tuple[Landscape, Path, bool]:
    """Score the grid once per data/model/price setup and reuse it afterwards."""
    levels = component_levels(unique_mixes(tests), config.levels)
    path = cache_dir / f"landscape_{_cache_key(tests, config)}.npz"
    if path.exists():
        return load_landscape(path, tests, config), path, True
    landscape = score_grid(tests, levels, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, strength=landscape.strength, cost=landscape.cost, distance=landscape.distance)
    return landscape, path, False


def load_landscape(path: Path, tests: pd.DataFrame, config: ExperimentConfig) -> Landscape:
    stored = np.load(path)
    return Landscape(
        levels=component_levels(unique_mixes(tests), config.levels),
        strength=stored["strength"],
        cost=stored["cost"],
        distance=stored["distance"],
        max_distance=config.max_distance_to_data,
    )


def describe_levels(levels: list[np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: values for name, values in zip(COMPONENTS, levels, strict=True)}
    ).rename_axis("level")
