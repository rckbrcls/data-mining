"""Shared scope and fixed parameters for the concrete mix-design DAMICORE AED."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


HYPOTHESIS_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = HYPOTHESIS_ROOT / "data" / "concrete.csv"
ARTIFACT_ROOT = HYPOTHESIS_ROOT / "artifacts"

# Decision variables: kg of each component per m³ of concrete (UCI order).
COMPONENTS = (
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
)
AGE_COLUMN = "age_days"
TARGET_COLUMN = "strength_mpa"

# Illustrative material prices in US$/kg, only to give the cost objective realistic
# proportions (cement and superplasticizer dominate, aggregates and water are cheap).
PRICES_USD_PER_KG = {
    "cement": 0.11,
    "slag": 0.05,
    "fly_ash": 0.03,
    "water": 0.001,
    "superplasticizer": 2.50,
    "coarse_aggregate": 0.012,
    "fine_aggregate": 0.010,
}

@dataclass(frozen=True)
class ExperimentConfig:
    # Search space: every component takes one of `levels` values (quantiles of real mixes).
    levels: int = 8
    design_age_days: int = 28
    # A mix is trusted only if its standardized distance to the nearest real mix is at most
    # this value (about the 90th percentile of real mixes' own nearest-neighbour distance).
    max_distance_to_data: float = 1.0
    # Strength targets (MPa) for the "cheapest mix" recommendations.
    strength_targets: tuple[float, ...] = (30.0, 40.0, 50.0, 60.0)

    # AED budget per run.
    seeds: tuple[int, ...] = tuple(range(101, 2101, 100))
    population_size: int = 200
    selection_fraction: float = 0.1
    archive_size: int = 100
    generations: int = 20
    offspring_per_generation: int = 50
    # DAMICORE splits the promising mixes into niches (regions of the front). Its own
    # FastGreedy cut is used when it gives at most `max_niches` niches of at least
    # `min_niche_size` mixes; otherwise its tree is cut into fewer niches until they do.
    max_niches: int = 6
    min_niche_size: int = 10
    # Draws from a niche's model before trying the other niches' models for an unseen mix.
    sampling_attempts: int = 200
    # Hypervolume is recorded every this many evaluations.
    checkpoint_every: int = 100

    workers: int = max(1, (os.cpu_count() or 2) - 1)

    @property
    def evaluations_per_run(self) -> int:
        return self.population_size + self.generations * self.offspring_per_generation


DEFAULT_CONFIG = ExperimentConfig()
