"""Shared scope and fixed parameters for the deceptive-trap DAMICORE AED."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


HYPOTHESIS_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = HYPOTHESIS_ROOT / "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    # Problem: `block_count` traps of `trap_size` bits, their bits scattered over the string.
    trap_size: int = 5
    block_counts: tuple[int, ...] = (4, 6, 8, 10, 12, 16)

    # Each (block count, population size) pair runs once per seed.
    population_sizes: tuple[int, ...] = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
    seeds: tuple[int, ...] = tuple(range(1, 11))
    # A population size "solves" a problem size when at least this share of seeds succeed.
    required_success: float = 0.9

    # The best `selection_fraction` of the population goes to DAMICORE; the offspring
    # (as many as the population) compete with the parents and the best survive.
    selection_fraction: float = 0.3
    max_generations: int = 60
    # Stop when the best fitness has not improved for this many generations.
    stall_generations: int = 10

    # The DAMICORE tree and NCD matrix are kept for runs of this seed only (to show them).
    example_seed: int = 1

    workers: int = max(1, (os.cpu_count() or 2) - 1)


DEFAULT_CONFIG = ExperimentConfig()
