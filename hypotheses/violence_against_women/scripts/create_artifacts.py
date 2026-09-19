"""Prepare fresh versioned DAMICORE inputs from the shared Disque 100 database."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from math import log2
import os
import shutil
import tempfile
from typing import Any, Iterator

import pandas as pd
import psycopg
from dotenv import load_dotenv

from .category_sets import CATEGORY_SETS, category_definition_hash, get_category_set
from .damicore_case_experiment import (
    CASE_COLUMNS,
    CASE_FIELDS,
    _fetch_prepared_case_records,
    _finalize_case_records,
    _validate_workers,
    build_case_corpora,
    compare_combination_distributions,
)
from .experiment_common import (
    ARTIFACT_SCHEMA_VERSION,
    CASE_SEEDS,
    END_DATE,
    HYPOTHESIS_ROOT,
    LOG_RATIO_LIMIT,
    MIN_REPORT_COUNT,
    PROJECT_ROOT,
    SMOOTHING_ALPHA,
    START_DATE,
    VICTIM_GENDER,
    ArtifactPaths,
    artifact_paths,
    ensure_artifact_directories,
    write_json,
)


PREPARATION_STAGES = ("source", "normalized", "cases", "manifest")


class PreparationLock:
    """Keep one fresh preparation active for a selected artifact output."""

    def __init__(self, paths: ArtifactPaths, descriptor: int, lock_path: str) -> None:
        self.paths = paths
        self._descriptor: int | None = descriptor
        self.lock_path = lock_path
        self._next_stage = 0

    @property
    def active(self) -> bool:
        return self._descriptor is not None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.active:
            raise RuntimeError("The preparation lock is no longer active; start a fresh run.")
        expected = PREPARATION_STAGES[self._next_stage]
        if name != expected:
            self.release()
            raise RuntimeError(
                f"Preparation stage {name!r} cannot run; expected {expected!r}. "
                "Start a fresh run."
            )
        try:
            yield
        except BaseException:
            self.release()
            raise
        else:
            self._next_stage += 1
            if name == PREPARATION_STAGES[-1]:
                self.release()

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    def __enter__(self) -> PreparationLock:
        if not self.active:
            raise RuntimeError("The preparation lock is no longer active; start a fresh run.")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def acquire_preparation_lock(
    category_set_version: str, workers: int = 2
) -> PreparationLock:
    """Acquire a non-blocking process lock before deleting versioned outputs."""
    _validate_workers(workers)
    paths = artifact_paths(category_set_version)
    lock_key = hashlib.sha256(os.fsencode(str(paths.root.resolve()))).hexdigest()
    lock_path = os.path.join(tempfile.gettempdir(), f"damicore-artifacts-{lock_key}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.lseek(descriptor, 0, os.SEEK_SET)
        owner = os.read(descriptor, 512).decode("utf-8", errors="replace").strip()
        os.close(descriptor)
        detail = f" ({owner})" if owner else ""
        raise RuntimeError(
            f"Artifact preparation is already active for {category_set_version}{detail}."
        ) from error
    owner = f"pid={os.getpid()} category_set_version={category_set_version}\n".encode()
    os.ftruncate(descriptor, 0)
    os.write(descriptor, owner)
    os.fsync(descriptor)
    return PreparationLock(paths, descriptor, lock_path)


def initialize_preparation(category_set_version: str, workers: int = 2) -> ArtifactPaths:
    """Validate configuration before rebuilding only the selected version."""
    _validate_workers(workers)
    paths = artifact_paths(category_set_version)
    if paths.root.exists():
        shutil.rmtree(paths.root)
    ensure_artifact_directories(paths)
    return paths


def configured_database_url() -> str:
    return os.getenv(
        "DISQUE100_DATABASE_URL",
        "postgresql://postgres@127.0.0.1:5433/disque100",
    )


def prepare_source_data(
    database_url: str, paths: ArtifactPaths
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch shared counts and un-serialized cases in one managed SQL connection."""
    DATABASE_URL = database_url
    CATEGORY_SET_VERSION = paths.category_set_version
    COMMON_WORK_ROOT = paths.common
    if CATEGORY_SET_VERSION == "v1_14":
        CATEGORY_SQL = """
        nullif(concat_ws(' > ',
            nullif(trim(split_part(violation, '>', 1)), ''),
            nullif(trim(split_part(violation, '>', 2)), '')
        ), '')
        """
    else:
        CATEGORY_SQL = """
        nullif(
            array_to_string(
                ARRAY(
                    SELECT nullif(trim(path_part), '')
                    FROM unnest(string_to_array(violation, '>')) AS split(path_part)
                    WHERE nullif(trim(path_part), '') IS NOT NULL
                ),
                ' > '
            ),
            ''
        )
        """

    CONTEXT_FIELDS = [
        ("faixa_etaria", "victim_age_group"),
        ("relacao_vitima_suspeito", "victim_suspect_relationship"),
        ("ambiente", "violation_setting"),
        ("mes", "registered_at"),
        ("inicio_violacoes", "violation_start_period"),
        ("canal_atendimento", "service_channel"),
        ("tipo_denunciante", "reporter_type"),
        ("frequencia", "frequency"),
        ("situacao_emergencia", "emergency_status"),
        ("motivacao", "motivation"),
        ("grupo_vulneravel", "vulnerable_group"),
        ("deficiencia_vitima", "victim_disability"),
        ("raca_cor_vitima", "victim_race_color"),
        ("escolaridade_vitima", "victim_education_level"),
        ("renda_vitima", "victim_income_range"),
        ("etnia_vitima", "victim_ethnicity"),
        ("faixa_etaria_suspeito", "suspect_age_group"),
        ("genero_suspeito", "suspect_gender"),
        ("escolaridade_suspeito", "suspect_education_level"),
        ("natureza_juridica_suspeito", "suspect_legal_nature"),
    ]
    assert {label for label, _ in CONTEXT_FIELDS} == {label for _, label in CASE_FIELDS}
    assert len(CONTEXT_FIELDS) == len(CASE_COLUMNS) == 20

    value_selects = [
        "source_hash",
        "violation AS raw_violation",
        "violation IS NOT NULL AS has_violation",
    ]
    for label, source_column in CONTEXT_FIELDS:
        if label == "mes":
            expression = "to_char(date_trunc('month', registered_at), 'YYYY-MM')"
        else:
            expression = f"coalesce(nullif(trim({source_column}), ''), 'DESCONHECIDO')"
        value_selects.append(f"{expression} AS {label}")

    context_values = ["('denuncia'::text, 'TODAS'::text)"]
    for label, _ in CONTEXT_FIELDS:
        context_values.append(f"('{label}'::text, report_values.{label})")

    # Normalize contexts once and retain raw violations for an exact category lookup.
    prepare_query = f"""
    CREATE TEMP TABLE damicore_report_values ON COMMIT DROP AS
    SELECT {', '.join(value_selects)}
    FROM public.disque100_reports
    WHERE registered_at >= %s AND registered_at < %s
      AND victim_gender = %s
    """

    # Deduplicate raw texts exactly, then restore the source column's default collation.
    category_map_query = f"""
    CREATE TEMP TABLE damicore_category_map ON COMMIT DROP AS
    SELECT violation, {CATEGORY_SQL} AS category
    FROM (
        SELECT raw_violation COLLATE "default" AS violation
        FROM (
            SELECT DISTINCT raw_violation COLLATE "C" AS raw_violation
            FROM pg_temp.damicore_report_values
            WHERE raw_violation IS NOT NULL
        ) AS distinct_violations
    ) AS source_violations
    """

    coverage_query = """
    WITH reports AS (
        SELECT report_values.source_hash,
               bool_or(report_values.has_violation) AS has_violation,
               bool_or(category_map.category IS NOT NULL) AS has_category
        FROM pg_temp.damicore_report_values AS report_values
        LEFT JOIN pg_temp.damicore_category_map AS category_map
          ON report_values.raw_violation COLLATE "C" = category_map.violation COLLATE "C"
        GROUP BY report_values.source_hash
    )
    SELECT count(*) AS female_reports,
           count(*) FILTER (WHERE has_category) AS reports_with_category,
           count(*) FILTER (WHERE NOT has_violation) AS reports_without_violation
    FROM reports
    """

    context_query = f"""
    SELECT category_map.category, contexts.dimension, contexts.value,
           count(DISTINCT report_values.source_hash) AS report_count
    FROM pg_temp.damicore_report_values AS report_values
    JOIN pg_temp.damicore_category_map AS category_map
      ON report_values.raw_violation COLLATE "C" = category_map.violation COLLATE "C"
    CROSS JOIN LATERAL (
        VALUES {', '.join(context_values)}
    ) AS contexts(dimension, value)
    WHERE category_map.category IS NOT NULL
    GROUP BY category_map.category, contexts.dimension, contexts.value
    ORDER BY category, dimension, value
    """

    parameters = (START_DATE, END_DATE, VICTIM_GENDER)
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL work_mem = '128MB'")
            cursor.execute("SET LOCAL temp_buffers = '64MB'")
            cursor.execute("SET LOCAL client_connection_check_interval = '5s'")
            cursor.execute(prepare_query, parameters)
            cursor.execute("ANALYZE pg_temp.damicore_report_values")
            cursor.execute(category_map_query)
            cursor.execute("ANALYZE pg_temp.damicore_category_map")
            cursor.execute(coverage_query)
            coverage = pd.DataFrame(
                cursor.fetchall(),
                columns=[column.name for column in cursor.description],
            )
            cursor.execute(context_query)
            context_counts = pd.DataFrame(
                cursor.fetchall(),
                columns=[column.name for column in cursor.description],
            )
        # Fetch arrays while the temporary tables exist; validation and serialization follow
        # the category-support checks in the later preparation cells.
        prepared_case_records = _fetch_prepared_case_records(
            connection, get_category_set(CATEGORY_SET_VERSION)
        )

    assert set(context_counts.columns) == {"category", "dimension", "value", "report_count"}
    assert set(context_counts["dimension"].unique()) == {
        "denuncia",
        *{label for _, label in CASE_FIELDS},
    }
    coverage.to_csv(COMMON_WORK_ROOT / "coverage.csv", index=False)
    context_counts.to_csv(COMMON_WORK_ROOT / "context-counts.csv", index=False)
    return coverage, context_counts, prepared_case_records


def prepare_normalized_artifacts(
    context_counts: pd.DataFrame, paths: ArtifactPaths
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, int]:
    """Validate category support and write the original normalized profiles."""
    CATEGORY_SET_VERSION = paths.category_set_version
    COMMON_WORK_ROOT = paths.common
    NORMALIZED_WORK_ROOT = paths.normalized
    category_summary = context_counts.loc[
        context_counts["dimension"] == "denuncia",
        ["category", "report_count"],
    ].rename(columns={"report_count": "support"})

    category_order = list(get_category_set(CATEGORY_SET_VERSION))
    category_set = set(category_order)
    category_support = category_summary.copy()
    category_support["status"] = "out_of_version"
    category_support.loc[category_support["category"].isin(category_set), "status"] = "included"
    category_support.loc[
        category_support["category"].isin(category_set)
        & category_support["support"].lt(MIN_REPORT_COUNT),
        "status",
    ] = "below_minimum_support"

    selected_support = category_summary.loc[
        category_summary["category"].isin(category_set)
    ].set_index("category")
    missing_categories = [
        category for category in category_order if category not in selected_support.index
    ]
    if missing_categories:
        raise ValueError(
            f"Category set {CATEGORY_SET_VERSION} is missing from the source taxonomy: "
            f"{missing_categories}"
        )
    below_minimum = [
        category
        for category in category_order
        if int(selected_support.loc[category, "support"]) < MIN_REPORT_COUNT
    ]
    if below_minimum:
        raise ValueError(
            f"Category set {CATEGORY_SET_VERSION} contains categories below the minimum "
            f"support of {MIN_REPORT_COUNT}: {below_minimum}"
        )

    included_support = selected_support.loc[category_order].reset_index()
    assert len(included_support) == len(category_order)

    profile_counts = context_counts.loc[
        context_counts["category"].isin(category_order)
        & context_counts["dimension"].ne("denuncia")
    ].copy()
    vocabulary = (
        profile_counts[["dimension", "value"]]
        .drop_duplicates()
        .sort_values(["dimension", "value"])
    )
    expected_dimensions = {label for _, label in CASE_FIELDS}
    assert set(vocabulary["dimension"]) == expected_dimensions

    normalized_profiles = (
        included_support[["category", "support"]]
        .merge(vocabulary, how="cross")
        .merge(profile_counts, on=["category", "dimension", "value"], how="left")
        .sort_values(["category", "dimension", "value"])
    )
    normalized_profiles["report_count"] = normalized_profiles["report_count"].fillna(0).astype(int)
    global_counts = (
        profile_counts.groupby(["dimension", "value"], as_index=False)["report_count"]
        .sum()
        .rename(columns={"report_count": "global_count"})
    )
    global_counts["global_prevalence"] = (
        global_counts["global_count"]
        / global_counts.groupby("dimension")["global_count"].transform("sum")
    )
    normalized_profiles = normalized_profiles.merge(
        global_counts[["dimension", "value", "global_prevalence"]],
        on=["dimension", "value"],
        how="left",
        validate="many_to_one",
    )
    normalized_profiles["smoothed_prevalence"] = (
        normalized_profiles["report_count"]
        + SMOOTHING_ALPHA * normalized_profiles["global_prevalence"]
    ) / (normalized_profiles["support"] + SMOOTHING_ALPHA)
    normalized_profiles["relative_bps"] = (
        10_000
        * (
            normalized_profiles["smoothed_prevalence"].div(
                normalized_profiles["global_prevalence"]
            ).map(log2).clip(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT)
            + LOG_RATIO_LIMIT
        )
        / (2 * LOG_RATIO_LIMIT)
    ).round().astype(int)

    profiles_by_category = {
        category: rows
        for category, rows in normalized_profiles.groupby("category", sort=False)
    }
    category_rows = []
    for number, category in enumerate(category_order, start=1):
        label = f"category-{number:03d}.txt"
        rows = profiles_by_category[category]
        lines = [
            f"{row.dimension}|{row.value}|{row.relative_bps:05d}"
            for row in rows.itertuples(index=False)
        ]
        corpus_path = NORMALIZED_WORK_ROOT / "corpus" / label
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        category_rows.append(
            {"label": label, "category": category, "support": int(rows["support"].iloc[0])}
        )

    category_map = pd.DataFrame(category_rows)
    category_map.to_csv(COMMON_WORK_ROOT / "category-map.csv", index=False)
    included_support.to_csv(COMMON_WORK_ROOT / "category-support.csv", index=False)
    category_support.to_csv(COMMON_WORK_ROOT / "category-scope.csv", index=False)
    context_counts.loc[
        context_counts["category"].isin(category_order)
    ].to_csv(COMMON_WORK_ROOT / "eligible-context-counts.csv", index=False)

    corpus_sizes = pd.Series(
        {
            path.name: path.stat().st_size
            for path in (NORMALIZED_WORK_ROOT / "corpus").glob("*.txt")
        },
        name="bytes",
    )
    assert corpus_sizes.nunique() == 1
    return category_order, included_support, category_map, int(corpus_sizes.iloc[0])


def prepare_case_artifacts(
    prepared_case_records: pd.DataFrame,
    category_order: list[str],
    included_support: pd.DataFrame,
    category_map: pd.DataFrame,
    paths: ArtifactPaths,
    *,
    workers: int = 2,
) -> tuple[pd.DataFrame, int, int]:
    """Serialize cases, consume their context arrays, and write the original case corpora."""
    PATHS = paths
    COMMON_WORK_ROOT = paths.common
    expected_dimensions = {label for _, label in CASE_FIELDS}
    case_records = _finalize_case_records(prepared_case_records, category_order, workers=workers)
    del prepared_case_records
    case_support = (
        case_records.groupby("category", as_index=False)["source_hash"]
        .nunique()
        .rename(columns={"source_hash": "case_count"})
    )
    expected_support = included_support[["category", "support"]].rename(
        columns={"support": "expected_case_count"}
    )
    case_support_check = expected_support.merge(
        case_support, on="category", validate="one_to_one"
    ).set_index("category").loc[category_order]
    assert case_support_check["expected_case_count"].equals(
        case_support_check["case_count"]
    )

    # Arrays are no longer needed after serialization and support validation.
    case_records.drop(columns=CASE_COLUMNS, inplace=True)
    case_category_map, case_combination_counts, balanced_sample_size = build_case_corpora(
        case_records=case_records,
        category_map=category_map,
        work_dir=PATHS.work,
        seeds=CASE_SEEDS,
        metadata_dir=COMMON_WORK_ROOT,
        workers=workers,
    )
    compare_combination_distributions(
        case_combination_counts,
        COMMON_WORK_ROOT / "case-combination-comparison.csv",
    )
    assert not case_records.duplicated(["source_hash", "category"]).any()
    has_private_keys = False
    has_expected_dimensions = True
    for canonical_record in case_records["canonical_record"]:
        keys = set(json.loads(canonical_record))
        has_private_keys |= bool({"source_hash", "category", "id"}.intersection(keys))
        has_expected_dimensions &= keys == expected_dimensions
    assert not has_private_keys
    assert has_expected_dimensions
    assert case_category_map["case_count"].gt(0).all()
    assert set(case_category_map["category"]) == set(category_order)
    assert case_category_map.loc[
        case_category_map["regime"] == "case-balanced", "case_count"
    ].eq(balanced_sample_size).all()
    return case_category_map, balanced_sample_size, len(case_records)


def write_artifact_manifests(
    paths: ArtifactPaths, category_order: list[str], balanced_sample_size: int
) -> dict[str, Any]:
    """Write compatibility manifests only after all preparation stages succeed."""
    CATEGORY_SET_VERSION = paths.category_set_version
    COMMON_WORK_ROOT = paths.common
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "hypothesis": "violence_against_women",
        "category_set_version": CATEGORY_SET_VERSION,
        "category_definition_hash": category_definition_hash(
            get_category_set(CATEGORY_SET_VERSION)
        ),
        "source_table": "public.disque100_reports",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "victim_gender": VICTIM_GENDER,
        "minimum_report_count": MIN_REPORT_COUNT,
        "smoothing_alpha": SMOOTHING_ALPHA,
        "log_ratio_limit": LOG_RATIO_LIMIT,
        "case_seeds": CASE_SEEDS,
        "dimensions": [label for _, label in CASE_FIELDS],
        "dimension_count": len(CASE_FIELDS),
        "category_order": category_order,
        "category_count": len(category_order),
        "balanced_sample_size": balanced_sample_size,
        "paths": {
            "category_map": "common/category-map.csv",
            "category_support": "common/category-support.csv",
            "case_category_map": "common/case-category-map.csv",
            "normalized_corpus": "normalized_categories/corpus",
            "case_full_corpus": "case_full/corpus",
            "case_balanced_root": "case_balanced",
        },
    }
    assert manifest["category_count"] == len(get_category_set(CATEGORY_SET_VERSION))
    assert manifest["dimension_count"] == 20
    write_json(COMMON_WORK_ROOT / "artifact-manifest.json", manifest)
    write_json(HYPOTHESIS_ROOT / "manifest.json", {
        "hypothesis": "violence_against_women",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "default_category_set_version": "v2_30",
        "category_set_versions": ["v1_14", "v2_30"],
        "notebooks": [
            "00_create_artifacts.ipynb",
            "01_experiment_case_full.ipynb",
            "02_experiment_case_balanced.ipynb",
            "03_experiment_normalized_categories.ipynb",
            "04_compare_experiments.ipynb",
            "05_compare_category_sets.ipynb",
        ],
    })
    return manifest


def run_artifact_preparation(
    category_set_version: str,
    database_url: str,
    *,
    workers: int = 2,
) -> dict[str, Any]:
    """Build new inputs; never resume or reuse earlier experiment runs."""
    preparation_lock = acquire_preparation_lock(category_set_version, workers)
    try:
        paths = initialize_preparation(category_set_version, workers)
        with preparation_lock.stage("source"):
            coverage, context_counts, prepared_case_records = prepare_source_data(
                database_url, paths
            )
            print(coverage.to_string(index=False))
            print(context_counts.head().to_string(index=False))
            print(f"Context rows: {len(context_counts):,}")
            del coverage

        with preparation_lock.stage("normalized"):
            category_order, included_support, category_map, corpus_bytes = (
                prepare_normalized_artifacts(context_counts, paths)
            )
            del context_counts
            print(category_map.to_string(index=False))
            print(f"Included categories: {len(category_order)}")
            print(f"Normalized corpus bytes per category: {corpus_bytes:,}")

        with preparation_lock.stage("cases"):
            case_category_map, balanced_sample_size, case_count = prepare_case_artifacts(
                prepared_case_records,
                category_order,
                included_support,
                category_map,
                paths,
                workers=workers,
            )
            del prepared_case_records, included_support, category_map
            print(f"Case records in memory: {case_count:,}")
            print(f"Balanced sample size per category and replica: {balanced_sample_size:,}")
            print(f"Balanced replicas: {len(CASE_SEEDS)}")
            print(case_category_map.head().to_string(index=False))
            del case_category_map

        with preparation_lock.stage("manifest"):
            manifest = write_artifact_manifests(
                paths, category_order, balanced_sample_size
            )
            print("Artifact manifest written.")
        return manifest
    finally:
        preparation_lock.release()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category-set-version",
        choices=tuple(CATEGORY_SETS),
        default=os.getenv("DAMICORE_CATEGORY_SET_VERSION", "v2_30"),
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    arguments = parser.parse_args()
    run_artifact_preparation(
        arguments.category_set_version,
        configured_database_url(),
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
