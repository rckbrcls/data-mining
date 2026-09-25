"""Pre-registered scope and search settings for the emergency hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HYPOTHESIS_ROOT = Path(__file__).resolve().parents[1]
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

TRAIN_START = "2024-01-01"
VALIDATION_START = "2026-01-01"
TEST_START = "2026-04-01"
END_DATE = "2026-07-01"


@dataclass(frozen=True)
class ExperimentConfig:
    search_train_limit: int = 50_000
    validation_limit: int = 30_000
    final_train_limit: int = 200_000
    initial_pool_size: int = 64
    elite_size: int = 32
    generations: int = 5
    offspring_per_generation: int = 16
    random_exploration: float = 0.05
    max_block_size: int = 4
    top_fraction: float = 0.05
    loss_tolerance: float = 0.02
    recall_tolerance: float = 0.02
    seeds: tuple[int, ...] = (101, 202, 303)

    @property
    def evaluations_per_method(self) -> int:
        return self.initial_pool_size + self.generations * self.offspring_per_generation


DEFAULT_CONFIG = ExperimentConfig()
