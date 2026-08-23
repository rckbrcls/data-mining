from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


HASH_PATTERN = re.compile(r"(?i)\b[0-9a-f]{64}\b")
PUBLIC_TEXT_SUFFIXES = {".csv", ".graphml", ".json", ".md", ".txt"}


@dataclass(frozen=True)
class ArtifactPaths:
    base: Path
    work_damicore: Path
    results_damicore: Path
    results_semantic: Path
    results_temporal: Path

    @classmethod
    def for_experiment(
        cls, project_root: str | Path, config: ExperimentConfig
    ) -> "ArtifactPaths":
        base = Path(project_root).resolve() / "artifacts" / config.experiment_id
        return cls(
            base=base,
            work_damicore=base / "work" / "01_damicore",
            results_damicore=base / "results" / "01_damicore",
            results_semantic=base / "results" / "02_semantic",
            results_temporal=base / "results" / "03_temporal",
        )

    @property
    def manifest(self) -> Path:
        return self.base / "manifest.json"

    def ensure_directories(self) -> None:
        for path in (
            self.work_damicore,
            self.results_damicore,
            self.results_semantic,
            self.results_temporal,
        ):
            path.mkdir(parents=True, exist_ok=True)


def require_artifact(path: str | Path, producer_notebook: str) -> Path:
    artifact = Path(path)
    if not artifact.exists() or artifact.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing artifact {artifact}. Run {producer_notebook} first."
        )
    return artifact


def write_manifest(
    paths: ArtifactPaths,
    config: ExperimentConfig,
    observed: dict[str, Any] | None = None,
) -> None:
    paths.base.mkdir(parents=True, exist_ok=True)
    previous_observed: dict[str, Any] = {}
    if paths.manifest.exists():
        previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
        previous_fingerprint = previous.get("config_fingerprint")
        if previous_fingerprint != config.fingerprint:
            raise RuntimeError(
                "Cannot update a manifest created by a different experiment configuration."
            )
        previous_observed.update(previous.get("observed", {}))
    previous_observed.update(observed or {})
    payload = {
        "config_fingerprint": config.fingerprint,
        "experiment": {
            "experiment_id": config.experiment_id,
            "registered_from": config.registered_from,
            "registered_before": config.registered_before,
            "victim_gender": config.victim_gender,
            "sample_size": config.sample_size,
            "sample_seed": config.sample_seed,
        },
        "observed": previous_observed,
    }
    paths.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_matching_manifest(paths: ArtifactPaths, config: ExperimentConfig) -> dict[str, Any]:
    require_artifact(paths.manifest, "01_damicore_structural_baseline.ipynb")
    payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
    actual = payload.get("config_fingerprint")
    if actual != config.fingerprint:
        raise RuntimeError(
            f"Artifact fingerprint mismatch: expected {config.fingerprint}, found {actual}."
        )
    return payload


def assert_public_artifact_safe(path: str | Path) -> None:
    artifact = Path(path)
    if artifact.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
        return
    content = artifact.read_text(encoding="utf-8")
    if "source_hash" in content or HASH_PATTERN.search(content):
        raise ValueError(f"Private identifier found in public artifact: {artifact}")
