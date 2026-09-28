"""Two-objective Pareto tools: maximize quality, minimize the number of collected fields."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def dominates(left: tuple[float, int], right: tuple[float, int]) -> bool:
    """``left`` = (quality, field_count) is no worse on both objectives and better on one."""
    return (
        left[0] >= right[0]
        and left[1] <= right[1]
        and (left[0] > right[0] or left[1] < right[1])
    )


def nondominated_ranks(points: Sequence[tuple[float, int]]) -> np.ndarray:
    """Front index of every point (0 = Pareto front), by repeated peeling."""
    if not points:
        return np.zeros(0, dtype=int)
    quality = np.array([point[0] for point in points], dtype=float)
    count = np.array([point[1] for point in points], dtype=float)
    # dominated_by[i, j]: point j dominates point i.
    no_worse = (quality[None, :] >= quality[:, None]) & (count[None, :] <= count[:, None])
    better = (quality[None, :] > quality[:, None]) | (count[None, :] < count[:, None])
    dominated_by = no_worse & better
    ranks = np.full(len(points), -1, dtype=int)
    remaining = np.ones(len(points), dtype=bool)
    rank = 0
    while remaining.any():
        front = remaining & ~(dominated_by[:, remaining].any(axis=1))
        ranks[front] = rank
        remaining &= ~front
        rank += 1
    return ranks


def crowding_distances(points: Sequence[tuple[float, int]], members: Sequence[int]) -> dict[int, float]:
    """NSGA-II crowding distance of ``members`` inside their own front."""
    distances = {index: 0.0 for index in members}
    if len(members) <= 2:
        return {index: float("inf") for index in members}
    for objective in (0, 1):
        ordered = sorted(members, key=lambda index: points[index][objective])
        low = points[ordered[0]][objective]
        high = points[ordered[-1]][objective]
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            gap = points[ordered[position + 1]][objective] - points[ordered[position - 1]][objective]
            distances[ordered[position]] += gap / (high - low)
    return distances


def pareto_order(points: Sequence[tuple[float, int]]) -> list[int]:
    """Indices sorted by (front, most isolated first); ties broken by index for determinism."""
    ranks = nondominated_ranks(points)
    crowding: dict[int, float] = {}
    for rank in np.unique(ranks):
        crowding.update(crowding_distances(points, list(np.flatnonzero(ranks == rank))))
    return sorted(range(len(points)), key=lambda index: (ranks[index], -crowding[index], index))


def pareto_front(table: pd.DataFrame, quality: str = "mean_gain_bits") -> pd.DataFrame:
    """Non-dominated rows: for each field count, the best quality, kept while it improves."""
    best_by_count = (
        table.sort_values([quality, "mask"], ascending=[False, True])
        .drop_duplicates("feature_count")
        .sort_values("feature_count")
    )
    front_rows = []
    best_so_far = -np.inf
    for row in best_by_count.itertuples(index=False):
        value = getattr(row, quality)
        if value > best_so_far:
            front_rows.append(row._asdict())
            best_so_far = value
    return pd.DataFrame(front_rows).reset_index(drop=True)

