from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    registered_from: str
    registered_before: str
    victim_gender: str
    sample_size: int
    sample_seed: int
    database_url_env: str

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("rb") as handle:
            values = tomllib.load(handle)
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
