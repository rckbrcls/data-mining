"""Load and audit report-level Disque 100 data from PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

from .config import (
    FEATURES,
    HYPOTHESIS_ROOT,
    KNOWN_STATUSES,
    NEGATIVE_STATUS,
    DataProfile,
    active_profile,
)


@dataclass
class ValidationFold:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass
class PreparedData:
    folds: list[ValidationFold]
    final_train: pd.DataFrame
    test: pd.DataFrame
    audit: dict[str, int | str | dict[str, int]]
    monthly_counts: pd.DataFrame
    profile: DataProfile


def configured_database_url(profile: DataProfile | None = None) -> str:
    if profile is not None and profile.database_url is not None:
        return profile.database_url
    load_dotenv(HYPOTHESIS_ROOT.parents[1] / ".env")
    return os.getenv(
        "DISQUE100_DATABASE_URL",
        "postgresql://postgres@nitro.taile3753b.ts.net:5432/disque100",
    )


def _normalized_sql(column: str, uppercase: bool = False) -> str:
    if column not in (*FEATURES, "emergency_status"):
        raise ValueError(f"Unexpected source column: {column}")
    trimmed = f"nullif(btrim({column}), '')"
    value = f"upper({trimmed})" if uppercase else trimmed
    return (
        f"CASE WHEN upper({trimmed}) IN ('NULL', 'N/D', 'NULO') "
        f"THEN NULL ELSE {value} END"
    )


def report_query() -> str:
    """Aggregate repeated source rows to one row per report hash in PostgreSQL."""
    source_columns = [
        "source_hash",
        "registered_at",
        f"{_normalized_sql('emergency_status', uppercase=True)} AS emergency_status",
    ]
    source_columns.extend(
        f"{_normalized_sql(feature)} AS {feature}" for feature in FEATURES
    )

    aggregate_columns = [
        "source_hash",
        "min(registered_at) AS report_at",
        "max(registered_at) AS report_at_max",
        "count(*) AS row_count",
        "count(emergency_status) AS status_count",
        "min(emergency_status) AS status_min",
        "max(emergency_status) AS status_max",
    ]
    aggregate_columns.extend(
        f"min({feature}) AS {feature}_min, max({feature}) AS {feature}_max"
        for feature in FEATURES
    )
    final_features = [
        (
            f"CASE WHEN {feature}_min IS NULL THEN 'UNKNOWN' "
            f"WHEN {feature}_min = {feature}_max THEN {feature}_min "
            f"ELSE 'MULTIPLE' END AS {feature}"
        )
        for feature in FEATURES
    ]

    return f"""
CREATE TEMP TABLE emergency_reports ON COMMIT DROP AS
WITH source_rows AS (
    SELECT {', '.join(source_columns)}
    FROM public.disque100_reports
    WHERE registered_at >= %s AND registered_at < %s
), report_aggregates AS (
    SELECT {', '.join(aggregate_columns)}
    FROM source_rows
    GROUP BY source_hash
)
SELECT source_hash, report_at, report_at_max, row_count, status_count,
       status_min, status_max,
       {', '.join(
           f'{feature}_min, {feature}_max, {feature}'
           for feature in FEATURES
       )}
FROM (
    SELECT source_hash, report_at, report_at_max, row_count, status_count,
           status_min, status_max, {', '.join(final_features)},
           {', '.join(
               f'{feature}_min, {feature}_max'
               for feature in FEATURES
           )}
    FROM report_aggregates
) AS report_rows
"""


def _fetch_dataframe(
    cursor: psycopg.Cursor, query: str, parameters: tuple = ()
) -> pd.DataFrame:
    cursor.execute(query, parameters)
    return pd.DataFrame(
        cursor.fetchall(),
        columns=[column.name for column in cursor.description],
    )


def _valid_report_clause() -> str:
    return """
FROM pg_temp.emergency_reports
WHERE report_at = report_at_max
  AND status_count = row_count
  AND status_min = status_max
  AND status_min = ANY(%s)
"""


def _load_period(cursor: psycopg.Cursor, start: str, end: str) -> pd.DataFrame:
    return _fetch_dataframe(
        cursor,
        f"""
SELECT report_at, {", ".join(FEATURES)},
       CASE WHEN status_min = %s THEN 0 ELSE 1 END AS target
{_valid_report_clause()}
  AND report_at >= %s AND report_at < %s
ORDER BY report_at, md5(source_hash)
""",
        (NEGATIVE_STATUS, list(KNOWN_STATUSES), start, end),
    )


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].value_counts().to_dict()
    return {
        "non_emergency": int(counts.get(0, 0)),
        "emergency": int(counts.get(1, 0)),
    }


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, [*FEATURES, "target"]].reset_index(drop=True)


def _validate_classes(frame: pd.DataFrame, name: str) -> None:
    if frame.empty or frame["target"].nunique() != 2:
        raise ValueError(f"{name} must contain both target classes.")


def prepare_data(
    database_url: str | None = None,
    profile: DataProfile | None = None,
) -> PreparedData:
    """Read distinct reports and construct the profile's temporal validation folds."""
    profile = profile or active_profile()
    try:
        connection = psycopg.connect(
            database_url or configured_database_url(profile),
            connect_timeout=5,
        )
    except psycopg.OperationalError as error:
        raise RuntimeError(
            "Cannot connect to public.disque100_reports in the configured PostgreSQL "
            "database. This experiment requires that source table."
        ) from error

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL work_mem = '128MB'")
                cursor.execute(report_query(), (profile.train_start, profile.data_end))
                cursor.execute("ANALYZE pg_temp.emergency_reports")

                audit_frame = _fetch_dataframe(
                    cursor,
                    """
SELECT count(*) AS reports_in_scope,
       count(*) FILTER (WHERE report_at <> report_at_max) AS date_conflicts,
       count(*) FILTER (
           WHERE status_min IS DISTINCT FROM status_max
              OR (status_count > 0 AND status_count < row_count)
       ) AS status_conflicts,
       count(*) FILTER (
           WHERE status_min IS NULL AND status_max IS NULL
       ) AS unknown_status,
       count(*) FILTER (
           WHERE status_min = status_max
             AND status_min IS NOT NULL
             AND NOT status_min = ANY(%s)
       ) AS unsupported_status,
       count(*) FILTER (
           WHERE report_at = report_at_max
             AND status_count = row_count
             AND status_min = status_max
             AND status_min = ANY(%s)
       ) AS usable_reports,
       max(report_at_max) AS latest_registered_at
FROM pg_temp.emergency_reports
""",
                    (list(KNOWN_STATUSES), list(KNOWN_STATUSES)),
                )
                conflict_frame = _fetch_dataframe(
                    cursor,
                    "SELECT "
                    + ", ".join(
                        f"count(*) FILTER (WHERE {feature}_min IS DISTINCT FROM "
                        f"{feature}_max) AS {feature}"
                        for feature in FEATURES
                    )
                    + " FROM pg_temp.emergency_reports",
                )
                history = _load_period(cursor, profile.train_start, profile.test_start)
                test = _load_period(cursor, profile.test_start, profile.data_end)
    finally:
        connection.close()

    audit_row = audit_frame.iloc[0].to_dict()
    latest_date = pd.Timestamp(audit_row["latest_registered_at"]).date()
    expected_last_day = (pd.Timestamp(profile.data_end) - pd.Timedelta(days=1)).date()
    if latest_date < expected_last_day:
        raise ValueError(
            "The configured final test quarter is not complete in the source table: "
            f"latest registration is {latest_date}, expected data through {expected_last_day}."
        )

    history["report_at"] = pd.to_datetime(history["report_at"])
    test["report_at"] = pd.to_datetime(test["report_at"])
    all_usable = pd.concat([history, test], ignore_index=True)
    monthly_counts = (
        all_usable.assign(
            report_month=all_usable["report_at"].dt.to_period("M").astype(str)
        )
        .groupby(["report_month", "target"], as_index=False)
        .size()
        .rename(columns={"size": "report_count"})
    )

    folds: list[ValidationFold] = []
    fold_counts: dict[str, dict[str, dict[str, int]]] = {}
    for window in profile.windows:
        train = history.loc[
            (history["report_at"] >= pd.Timestamp(window.train_start))
            & (history["report_at"] < pd.Timestamp(window.train_end))
        ]
        validation = history.loc[
            (history["report_at"] >= pd.Timestamp(window.validation_start))
            & (history["report_at"] < pd.Timestamp(window.validation_end))
        ]
        _validate_classes(train, f"{window.name} training period")
        _validate_classes(validation, f"{window.name} validation period")
        folds.append(
            ValidationFold(
                window.name,
                _model_frame(train),
                _model_frame(validation),
            )
        )
        fold_counts[window.name] = {
            "train": _class_counts(train),
            "validation": _class_counts(validation),
        }

    final_train = history.loc[history["report_at"] < pd.Timestamp(profile.test_start)]
    _validate_classes(final_train, "Final training period")
    _validate_classes(test, "Final test quarter")

    audit: dict[str, int | str | dict[str, int]] = {
        key: int(value)
        for key, value in audit_row.items()
        if key != "latest_registered_at"
    }
    audit.update(
        {
            "latest_registered_at": str(audit_row["latest_registered_at"]),
            "excluded_reports": int(audit_row["reports_in_scope"])
            - int(audit_row["usable_reports"]),
            "feature_conflicts": {
                key: int(value)
                for key, value in conflict_frame.iloc[0].to_dict().items()
            },
            "source_table": "public.disque100_reports",
            "report_unit": "one source_hash after within-report aggregation",
            "data_profile": profile.name,
            "scope_start": profile.train_start,
            "scope_end_exclusive": profile.data_end,
            "test_period": profile.test_name,
            "validation_folds": fold_counts,
            "final_train": _class_counts(final_train),
            "final_test": _class_counts(test),
        }
    )

    return PreparedData(
        folds=folds,
        final_train=_model_frame(final_train),
        test=_model_frame(test),
        audit=audit,
        monthly_counts=monthly_counts,
        profile=profile,
    )
