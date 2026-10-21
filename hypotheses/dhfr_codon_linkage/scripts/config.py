"""Shared scope and fixed parameters for the DHFR DAMICORE AED."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


HYPOTHESIS_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = HYPOTHESIS_ROOT / "data" / "papkou2023_dhfr_fitness.csv.gz"
ARTIFACT_ROOT = HYPOTHESIS_ROOT / "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    # Each run starts from a random population and may measure at most `budget` distinct
    # sequences (a laboratory does not measure the same sequence twice).
    population_size: int = 200
    selection_fraction: float = 0.3
    budget: int = 1_000
    seeds: tuple[int, ...] = tuple(range(1, 31))

    # A run "reaches the top" when it finds one of the `top_k` best measured sequences.
    top_k: int = 70

    # Pair epistasis (the reference for DAMICORE's groups) is measured among the
    # sequences that grow at all: below this fitness the cells are dead and every
    # position looks alike.
    alive_threshold: float = -0.5

    # The DAMICORE tree and NCD matrix are kept for this seed only (to show them).
    example_seed: int = 1

    workers: int = max(1, (os.cpu_count() or 2) - 1)


DEFAULT_CONFIG = ExperimentConfig()
