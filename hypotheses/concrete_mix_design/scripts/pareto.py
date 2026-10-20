"""Constrained two-objective tools: maximize strength, minimize cost, feasible mixes first."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# A point is (strength, cost, violation); violation 0 means feasible.
Point = tuple[float, float, float]


def nondominated_ranks(strength: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Front index of every point (0 = Pareto front), by repeated peeling."""
    if len(strength) == 0:
        return np.zeros(0, dtype=int)
    no_worse = (strength[None, :] >= strength[:, None]) & (cost[None, :] <= cost[:, None])
    better = (strength[None, :] > strength[:, None]) | (cost[None, :] < cost[:, None])
    dominated_by = no_worse & better  # [i, j]: point j dominates point i.
    ranks = np.full(len(strength), -1, dtype=int)
    remaining = np.ones(len(strength), dtype=bool)
    rank = 0
    while remaining.any():
        front = remaining & ~(dominated_by[:, remaining].any(axis=1))
        ranks[front] = rank
        remaining &= ~front
        rank += 1
    return ranks


def crowding_distances(values: np.ndarray, members: Sequence[int]) -> dict[int, float]:
    """NSGA-II crowding distance of ``members`` inside their own front."""
    if len(members) <= 2:
        return {index: float("inf") for index in members}
    distances = {index: 0.0 for index in members}
    for objective in range(values.shape[1]):
        ordered = sorted(members, key=lambda index: values[index, objective])
        low, high = values[ordered[0], objective], values[ordered[-1], objective]
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            gap = values[ordered[position + 1], objective] - values[ordered[position - 1], objective]
            distances[ordered[position]] += gap / (high - low)
    return distances


def constrained_order(points: Sequence[Point]) -> list[int]:
    """Deb's rule: feasible points by (front, most isolated), then infeasible by violation."""
    values = np.array(points, dtype=float).reshape(-1, 3)
    feasible = np.flatnonzero(values[:, 2] == 0)
    infeasible = np.flatnonzero(values[:, 2] > 0)

    ranks = nondominated_ranks(values[feasible, 0], values[feasible, 1])
    crowding: dict[int, float] = {}
    for rank in np.unique(ranks):
        members = [int(index) for index in feasible[ranks == rank]]
        crowding.update(crowding_distances(values[:, :2], members))
    rank_of = dict(zip(feasible.tolist(), ranks.tolist()))
    ordered_feasible = sorted(rank_of, key=lambda index: (rank_of[index], -crowding[index], index))
    ordered_infeasible = sorted(infeasible.tolist(), key=lambda index: (values[index, 2], index))
    return ordered_feasible + ordered_infeasible


def front_mask(strength: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Non-dominated points in O(n log n): sweep by cost, keep each new strength record."""
    order = np.lexsort((-strength, cost))
    keep = np.zeros(len(strength), dtype=bool)
    best = -np.inf
    for index in order:
        if strength[index] > best:
            keep[index] = True
            best = strength[index]
    return keep


def pareto_front(table: pd.DataFrame) -> pd.DataFrame:
    """Feasible non-dominated rows of a table with strength_mpa, cost_usd_m3, violation."""
    feasible = table.loc[table["violation"] == 0].drop_duplicates("solution")
    keep = front_mask(feasible["strength_mpa"].to_numpy(), feasible["cost_usd_m3"].to_numpy())
    return feasible.loc[keep].sort_values("cost_usd_m3").reset_index(drop=True)


def hypervolume(
    strength: np.ndarray, cost: np.ndarray, reference: tuple[float, float]
) -> float:
    """Area dominated by the points and bounded by (min strength, max cost) = ``reference``."""
    ref_strength, ref_cost = reference
    inside = (strength > ref_strength) & (cost < ref_cost)
    if not inside.any():
        return 0.0
    strength, cost = strength[inside], cost[inside]
    keep = front_mask(strength, cost)
    order = np.argsort(cost[keep])
    front_cost, front_strength = cost[keep][order], strength[keep][order]
    # Staircase: from each front point's cost to the next one, the best strength so far.
    widths = np.diff(np.r_[front_cost, ref_cost])
    return float(np.sum(widths * (front_strength - ref_strength)))
