from __future__ import annotations

from typing import Any

import pandas as pd


VIOLATION_LEVELS = ("domain", "category", "subtype", "detail")


def split_violation(value: Any) -> list[str]:
    """Return non-empty levels from one source violation path."""
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(">") if part.strip()]


def split_violation_label(value: Any) -> dict[str, str | None]:
    """Map a source label to four semantic levels without translating it."""
    parts = split_violation(value)
    return {
        "source_label": None if not parts else " > ".join(parts),
        "domain": parts[0] if len(parts) > 0 else None,
        "category": parts[1] if len(parts) > 1 else None,
        "subtype": parts[2] if len(parts) > 2 else None,
        "detail": " > ".join(parts[3:]) if len(parts) > 3 else None,
    }


def category_from_violation(value: Any) -> str | None:
    parts = split_violation(value)
    return " > ".join(parts[:2]) if parts else None


def enrich_violations(frame: pd.DataFrame, column: str = "violation") -> pd.DataFrame:
    if column not in frame.columns:
        raise KeyError(f"Missing violation column: {column}")
    parsed = frame[column].map(split_violation_label).apply(pd.Series)
    return pd.concat([frame.copy(), parsed.drop(columns=["source_label"])], axis=1)
