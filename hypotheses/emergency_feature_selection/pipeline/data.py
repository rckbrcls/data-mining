"""Build one report-level observation per source_hash from the canonical table."""

from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

from .config import (
    DEFAULT_CONFIG,
    END_DATE,
    FEATURES,
    HYPOTHESIS_ROOT,
    KNOWN_STATUSES,
    NEGATIVE_STATUS,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
    ExperimentConfig,
)


@dataclass
class PreparedData:
    search_train: pd.DataFrame
    validation: pd.DataFrame
    final_train: pd.DataFrame
    test: pd.DataFrame
    audit: dict[str, int | str | dict[str, int]]
    monthly_counts: pd.DataFrame


def configured_database_url() -> str:
    load_dotenv(HYPOTHESIS_ROOT.parents[1] / ".env")
    return os.getenv(
        "DISQUE100_DATABASE_URL",
        "postgresql://postgres@127.0.0.1:5433/disque100",
    )


def _normalized(column: str) -> str:
    if column not in (*FEATURES, "emergency_status"):
        raise ValueError(f"Unexpected source column: {column}")
    trimmed = f"nullif(btrim({column}), '')"
    value = f"upper({trimmed})" if column == "emergency_status" else trimmed
    return (
        f"CASE WHEN upper({trimmed}) IN ('NULL', 'N/D', 'NULO') "
        f"THEN NULL ELSE {value} END"
    )


def report_query() -> str:
    """Use min/max to flag conflicts without multiplying reports by source rows."""
    source_columns = [
        "source_hash",
        "registered_at",
        f"{_normalized('emergency_status')} AS emergency_status",
    ]
    source_columns.extend(f"{_normalized(feature)} AS {feature}" for feature in FEATURES)
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
       {', '.join(final_features)}
FROM report_aggregates
"""


def _fetch_dataframe(cursor: psycopg.Cursor, query: str, parameters: tuple) -> pd.DataFrame:
    cursor.execute(query, parameters)
    return pd.DataFrame(
        cursor.fetchall(),
        columns=[column.name for column in cursor.description],
    )


def _valid_reports_query() -> str:
    return """
FROM pg_temp.emergency_reports
WHERE report_at = report_at_max
  AND status_count = row_count
  AND status_min = status_max
  AND status_min = ANY(%s)
"""


def _load_split(
    cursor: psycopg.Cursor,
    start: str,
    end: str,
    limit: int | None,
) -> pd.DataFrame:
    columns = ", ".join(FEATURES)
    count_frame = _fetch_dataframe(
        cursor,
        f"""SELECT CASE WHEN status_min = %s THEN 0 ELSE 1 END AS target,
                   count(*) AS report_count
{_valid_reports_query()}
  AND report_at >= %s AND report_at < %s
GROUP BY 1""",
        (NEGATIVE_STATUS, list(KNOWN_STATUSES), start, end),
    )
    counts = {int(row.target): int(row.report_count) for row in count_frame.itertuples()}
    if set(counts) != {0, 1}:
        raise ValueError(f"The {start} to {end} split must contain both classes.")
    quotas = stratified_limits(counts, limit)
    frames = []
    for target in (0, 1):
        comparison = "=" if target == 0 else "<>"
        class_query = f"""
SELECT {columns}, CASE WHEN status_min = %s THEN 0 ELSE 1 END AS target
{_valid_reports_query()}
  AND report_at >= %s AND report_at < %s
  AND status_min {comparison} %s
ORDER BY md5(source_hash)
"""
        parameters: tuple = (NEGATIVE_STATUS, list(KNOWN_STATUSES), start, end, NEGATIVE_STATUS)
        if quotas[target] is not None:
            class_query += "LIMIT %s"
            parameters += (quotas[target],)
        frames.append(_fetch_dataframe(cursor, class_query, parameters))
    return pd.concat(frames, ignore_index=True)


def stratified_limits(counts: dict[int, int], limit: int | None) -> dict[int, int | None]:
    """Allocate a fixed-size sample proportionally, preserving both classes."""
    if limit is None or limit >= sum(counts.values()):
        return {0: None, 1: None}
    if limit < 2 or min(counts.values()) < 1:
        raise ValueError("A stratified sample needs at least one report per class.")
    negative = round(limit * counts[0] / sum(counts.values()))
    negative = max(1, min(limit - 1, negative))
    negative = min(negative, counts[0])
    positive = min(limit - negative, counts[1])
    negative += limit - negative - positive
    return {0: negative, 1: positive}


def prepare_data(
    database_url: str | None = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> PreparedData:
    """Read the database once and keep identifiers inside PostgreSQL only."""
    try:
        connection = psycopg.connect(database_url or configured_database_url())
    except psycopg.OperationalError as error:
        raise RuntimeError(
            "Cannot connect to the existing Disque 100 PostgreSQL database. "
            "This experiment does not replace it with CSV input."
        ) from error

    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL work_mem = '128MB'")
            cursor.execute(report_query(), (TRAIN_START, END_DATE))
            cursor.execute("ANALYZE pg_temp.emergency_reports")
            audit_table = _fetch_dataframe(
                cursor,
                """
SELECT count(*) AS reports_in_scope,
       count(*) FILTER (WHERE report_at <> report_at_max) AS date_conflicts,
       count(*) FILTER (
           WHERE status_min IS DISTINCT FROM status_max
              OR (status_count > 0 AND status_count < row_count)
       ) AS status_conflicts,
       count(*) FILTER (WHERE status_min IS NULL AND status_max IS NULL) AS unknown_status,
       count(*) FILTER (
           WHERE status_min = status_max AND status_min IS NOT NULL
             AND NOT status_min = ANY(%s)
       ) AS unsupported_status,
       count(*) FILTER (
           WHERE report_at = report_at_max AND status_count = row_count
             AND status_min = status_max
             AND status_min = ANY(%s)
       ) AS usable_reports
FROM pg_temp.emergency_reports
""",
                (list(KNOWN_STATUSES), list(KNOWN_STATUSES)),
            )
            monthly_counts = _fetch_dataframe(
                cursor,
                f"""
SELECT to_char(date_trunc('month', report_at), 'YYYY-MM') AS report_month,
       CASE WHEN status_min = %s THEN 0 ELSE 1 END AS target,
       count(*) AS report_count
{_valid_reports_query()}
GROUP BY 1, 2 ORDER BY 1, 2
""",
                (NEGATIVE_STATUS, list(KNOWN_STATUSES)),
            )
            feature_conflicts = _fetch_dataframe(
                cursor,
                "SELECT " + ", ".join(
                    f"count(*) FILTER (WHERE {feature} = 'MULTIPLE') AS {feature}"
                    for feature in FEATURES
                ) + " FROM pg_temp.emergency_reports",
                (),
            )
            search_train = _load_split(
                cursor, TRAIN_START, VALIDATION_START, config.search_train_limit
            )
            validation = _load_split(
                cursor, VALIDATION_START, TEST_START, config.validation_limit
            )
            final_train = _load_split(
                cursor, TRAIN_START, TEST_START, config.final_train_limit
            )
            test = _load_split(cursor, TEST_START, END_DATE, None)

    audit = {key: int(value) for key, value in audit_table.iloc[0].to_dict().items()}
    audit["excluded_reports"] = audit["reports_in_scope"] - audit["usable_reports"]
    audit["feature_conflicts"] = {
        key: int(value) for key, value in feature_conflicts.iloc[0].to_dict().items()
    }
    audit.update(
        {
            "source_table": "public.disque100_reports",
            "scope_start": TRAIN_START,
            "scope_end": END_DATE,
            "search_train_rows": len(search_train),
            "validation_rows": len(validation),
            "final_train_rows": len(final_train),
            "test_rows": len(test),
        }
    )
    for name, frame in (
        ("search_train", search_train),
        ("validation", validation),
        ("final_train", final_train),
        ("test", test),
    ):
        if frame.empty or frame["target"].nunique() != 2:
            raise ValueError(f"{name} must contain both emergency classes.")
        if set(frame.columns) != {*FEATURES, "target"}:
            raise ValueError(f"Unexpected columns returned for {name}.")
    return PreparedData(search_train, validation, final_train, test, audit, monthly_counts)
