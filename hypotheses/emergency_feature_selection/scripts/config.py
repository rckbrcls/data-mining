"""Shared scope and fixed parameters for the DAMICORE-guided multiobjective AED."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


HYPOTHESIS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = HYPOTHESIS_ROOT / "artifacts"

FEATURES = (
    "service_channel",
    "reporter_type",
    "violation_setting",
    "frequency",
    "violation_start_period",
    "vulnerable_group",
    "motivation",
    "victim_suspect_relationship",
    "victim_gender",
    "victim_age_group",
    "victim_disability",
    "victim_race_color",
    "suspect_age_group",
    "suspect_gender",
    "suspect_legal_nature",
)

POSITIVE_STATUSES = (
    "RISCO IMINENTE DE MORTE DA VÍTIMA",
    "SITUAÇÃO FLAGRANTE - ATÉ 24H DA OCORRÊNCIA - SEM FREQUÊNCIA",
    "SITUAÇÃO FLAGRANTE - ATÉ 24H DA OCORRÊNCIA",
    "VÍTIMA EM SANGRAMENTO",
)
NEGATIVE_STATUS = "NÃO"
KNOWN_STATUSES = (NEGATIVE_STATUS, *POSITIVE_STATUSES)


@dataclass(frozen=True)
class ValidationWindow:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


@dataclass(frozen=True)
class DataProfile:
    """Temporal scope: expanding-window validation folds plus one held-out test period."""

    name: str
    train_start: str
    test_start: str
    data_end: str
    test_name: str
    windows: tuple[ValidationWindow, ...]
    database_url: str | None = None


FULL_PROFILE = DataProfile(
    name="full",
    train_start="2024-01-01",
    test_start="2026-04-01",
    data_end="2026-07-01",
    test_name="2026-Q2",
    windows=(
        ValidationWindow("2025-Q3", "2024-01-01", "2025-07-01", "2025-07-01", "2025-10-01"),
        ValidationWindow("2025-Q4", "2024-01-01", "2025-10-01", "2025-10-01", "2026-01-01"),
        ValidationWindow("2026-Q1", "2024-01-01", "2026-01-01", "2026-01-01", "2026-04-01"),
    ),
)

# Development profile for the local PostgreSQL copy, which only holds 2026-S1.
LOCAL_2026_PROFILE = DataProfile(
    name="local_2026",
    train_start="2026-01-01",
    test_start="2026-06-01",
    data_end="2026-07-01",
    test_name="2026-06",
    windows=(
        ValidationWindow("2026-03", "2026-01-01", "2026-03-01", "2026-03-01", "2026-04-01"),
        ValidationWindow("2026-04", "2026-01-01", "2026-04-01", "2026-04-01", "2026-05-01"),
        ValidationWindow("2026-05", "2026-01-01", "2026-05-01", "2026-05-01", "2026-06-01"),
    ),
    database_url="postgresql://127.0.0.1:5432/disque100",
)

PROFILES = {profile.name: profile for profile in (FULL_PROFILE, LOCAL_2026_PROFILE)}


def active_profile() -> DataProfile:
    name = os.getenv("EMERGENCY_DATA_PROFILE", FULL_PROFILE.name)
    if name not in PROFILES:
        raise ValueError(f"Unknown EMERGENCY_DATA_PROFILE {name!r}; use one of {sorted(PROFILES)}.")
    return PROFILES[name]


@dataclass(frozen=True)
class ExperimentConfig:
    # Quality objective: cell rates shrunk toward the base rate with this pseudo-count.
    prior_strength: float = 50.0
    exhaustive_workers: int = max(1, (os.cpu_count() or 2) - 1)

    # AED budget per seed.
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
    population_size: int = 400
    selection_fraction: float = 0.3
    archive_size: int = 500
    generations: int = 10
    offspring_per_generation: int = 80
    laplace_alpha: float = 1.0

    # Recommended mask: fewest fields keeping this share of the best information gain.
    tolerance_fraction: float = 0.95

    @property
    def evaluations_per_seed(self) -> int:
        return self.population_size + self.generations * self.offspring_per_generation


DEFAULT_CONFIG = ExperimentConfig()
