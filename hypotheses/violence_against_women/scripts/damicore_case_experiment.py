from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
from itertools import islice
import json
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import pandas as pd
import psycopg


CASE_FIELDS = [
    ("age_group", "faixa_etaria"),
    ("relationship", "relacao_vitima_suspeito"),
    ("setting", "ambiente"),
    ("month", "mes"),
    ("violation_start_period", "inicio_violacoes"),
    ("service_channel", "canal_atendimento"),
    ("reporter_type", "tipo_denunciante"),
    ("frequency", "frequencia"),
    ("emergency_status", "situacao_emergencia"),
    ("motivation", "motivacao"),
    ("vulnerable_group", "grupo_vulneravel"),
    ("victim_disability", "deficiencia_vitima"),
    ("victim_race_color", "raca_cor_vitima"),
    ("victim_education_level", "escolaridade_vitima"),
    ("victim_income_range", "renda_vitima"),
    ("victim_ethnicity", "etnia_vitima"),
    ("suspect_age_group", "faixa_etaria_suspeito"),
    ("suspect_gender", "genero_suspeito"),
    ("suspect_education_level", "escolaridade_suspeito"),
    ("suspect_legal_nature", "natureza_juridica_suspeito"),
]

CASE_COLUMNS = [column for column, _ in CASE_FIELDS]
SERIALIZATION_BATCH_SIZE = 2_048

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def _validate_workers(workers: int) -> None:
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("Artifact workers must be either 1 or 2.")


def _ordered_process_map(
    executor: ProcessPoolExecutor,
    function: Callable[[_Input], _Output],
    arguments: Iterable[_Input],
    max_pending: int,
) -> Iterator[_Output]:
    """Keep submissions bounded and consume results in their original order."""
    iterator = iter(arguments)
    pending = deque()
    try:
        for argument in islice(iterator, max_pending):
            pending.append(executor.submit(function, argument))
        while pending:
            yield pending.popleft().result()
            for argument in islice(iterator, 1):
                pending.append(executor.submit(function, argument))
    finally:
        for future in pending:
            future.cancel()


def _case_value_batches(
    values: Iterable[tuple],
) -> Iterator[list[tuple]]:
    iterator = iter(values)
    while batch := list(islice(iterator, SERIALIZATION_BATCH_SIZE)):
        yield batch


def _serialize_case_batch(values: list[tuple]) -> list[str]:
    return [_serialize_case_values(row) for row in values]


def _safe_text_expression(column: str) -> str:
    return f"coalesce(nullif(trim({column}), ''), 'DESCONHECIDO')"


def _case_query(category_sql: str) -> str:
    expressions = {
        column: _safe_text_expression(source_column)
        for column, source_column in {
            "age_group": "victim_age_group",
            "relationship": "victim_suspect_relationship",
            "setting": "violation_setting",
            "violation_start_period": "violation_start_period",
            "service_channel": "service_channel",
            "reporter_type": "reporter_type",
            "frequency": "frequency",
            "emergency_status": "emergency_status",
            "motivation": "motivation",
            "vulnerable_group": "vulnerable_group",
            "victim_disability": "victim_disability",
            "victim_race_color": "victim_race_color",
            "victim_education_level": "victim_education_level",
            "victim_income_range": "victim_income_range",
            "victim_ethnicity": "victim_ethnicity",
            "suspect_age_group": "suspect_age_group",
            "suspect_gender": "suspect_gender",
            "suspect_education_level": "suspect_education_level",
            "suspect_legal_nature": "suspect_legal_nature",
        }.items()
    }

    return f"""
WITH source_rows AS (
    SELECT
        source_hash,
        {category_sql} AS category,
        {expressions["age_group"]} AS age_group,
        {expressions["relationship"]} AS relationship,
        {expressions["setting"]} AS setting,
        to_char(date_trunc('month', registered_at), 'YYYY-MM') AS month,
        {expressions["violation_start_period"]} AS violation_start_period,
        {expressions["service_channel"]} AS service_channel,
        {expressions["reporter_type"]} AS reporter_type,
        {expressions["frequency"]} AS frequency,
        {expressions["emergency_status"]} AS emergency_status,
        {expressions["motivation"]} AS motivation,
        {expressions["vulnerable_group"]} AS vulnerable_group,
        {expressions["victim_disability"]} AS victim_disability,
        {expressions["victim_race_color"]} AS victim_race_color,
        {expressions["victim_education_level"]} AS victim_education_level,
        {expressions["victim_income_range"]} AS victim_income_range,
        {expressions["victim_ethnicity"]} AS victim_ethnicity,
        {expressions["suspect_age_group"]} AS suspect_age_group,
        {expressions["suspect_gender"]} AS suspect_gender,
        {expressions["suspect_education_level"]} AS suspect_education_level,
        {expressions["suspect_legal_nature"]} AS suspect_legal_nature
    FROM public.disque100_reports
    WHERE registered_at >= %s
      AND registered_at < %s
      AND victim_gender = %s
      AND ({category_sql}) = ANY(%s)
)
SELECT
    source_hash,
    category,
    array_agg(DISTINCT age_group ORDER BY age_group) AS age_group,
    array_agg(DISTINCT relationship ORDER BY relationship) AS relationship,
    array_agg(DISTINCT setting ORDER BY setting) AS setting,
    array_agg(DISTINCT month ORDER BY month) AS month,
    array_agg(DISTINCT violation_start_period ORDER BY violation_start_period) AS violation_start_period,
    array_agg(DISTINCT service_channel ORDER BY service_channel) AS service_channel,
    array_agg(DISTINCT reporter_type ORDER BY reporter_type) AS reporter_type,
    array_agg(DISTINCT frequency ORDER BY frequency) AS frequency,
    array_agg(DISTINCT emergency_status ORDER BY emergency_status) AS emergency_status,
    array_agg(DISTINCT motivation ORDER BY motivation) AS motivation,
    array_agg(DISTINCT vulnerable_group ORDER BY vulnerable_group) AS vulnerable_group,
    array_agg(DISTINCT victim_disability ORDER BY victim_disability) AS victim_disability,
    array_agg(DISTINCT victim_race_color ORDER BY victim_race_color) AS victim_race_color,
    array_agg(DISTINCT victim_education_level ORDER BY victim_education_level) AS victim_education_level,
    array_agg(DISTINCT victim_income_range ORDER BY victim_income_range) AS victim_income_range,
    array_agg(DISTINCT victim_ethnicity ORDER BY victim_ethnicity) AS victim_ethnicity,
    array_agg(DISTINCT suspect_age_group ORDER BY suspect_age_group) AS suspect_age_group,
    array_agg(DISTINCT suspect_gender ORDER BY suspect_gender) AS suspect_gender,
    array_agg(DISTINCT suspect_education_level ORDER BY suspect_education_level) AS suspect_education_level,
    array_agg(DISTINCT suspect_legal_nature ORDER BY suspect_legal_nature) AS suspect_legal_nature
FROM source_rows
GROUP BY source_hash, category
ORDER BY category, source_hash
"""


def _fetch_prepared_case_records(
    connection: psycopg.Connection,
    included_categories: Iterable[str],
) -> pd.DataFrame:
    aggregates = ",\n    ".join(
        f"array_agg(DISTINCT report_values.{label} ORDER BY report_values.{label}) AS {column}"
        for column, label in CASE_FIELDS
    )
    query = f"""
SELECT
    report_values.source_hash,
    category_map.category,
    {aggregates}
FROM pg_temp.damicore_report_values AS report_values
JOIN pg_temp.damicore_category_map AS category_map
  ON report_values.raw_violation COLLATE "C" = category_map.violation COLLATE "C"
WHERE category_map.category = ANY(%s)
GROUP BY report_values.source_hash, category_map.category
ORDER BY category, source_hash
"""
    with connection.cursor() as cursor:
        cursor.execute(query, (list(included_categories),))
        return pd.DataFrame(
            cursor.fetchall(),
            columns=[column.name for column in cursor.description],
        )


def load_case_records(
    database_url: str,
    start_date: str,
    end_date: str,
    victim_gender: str,
    included_categories: Iterable[str],
    category_sql: str,
) -> pd.DataFrame:
    categories = list(included_categories)
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                _case_query(category_sql),
                (start_date, end_date, victim_gender, categories),
            )
            records = pd.DataFrame(
                cursor.fetchall(),
                columns=[column.name for column in cursor.description],
            )

    return _finalize_case_records(records, categories)


def _finalize_case_records(
    records: pd.DataFrame,
    included_categories: Iterable[str],
    *,
    workers: int = 1,
) -> pd.DataFrame:
    _validate_workers(workers)
    categories = list(included_categories)
    expected_columns = {"source_hash", "category", *CASE_COLUMNS}
    if set(records.columns) != expected_columns:
        raise ValueError(f"Unexpected case-record columns: {records.columns.tolist()}")
    if records.duplicated(["source_hash", "category"]).any():
        raise ValueError("The case corpus contains duplicate source_hash + category records.")
    if set(records["category"]) != set(categories):
        raise ValueError("The case corpus category set differs from the normalized experiment.")

    values = records[CASE_COLUMNS].itertuples(index=False, name=None)
    if workers == 1 or len(records) <= SERIALIZATION_BATCH_SIZE:
        canonical_records = [_serialize_case_values(row) for row in values]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as executor:
            canonical_records = [
                record
                for batch in _ordered_process_map(
                    executor, _serialize_case_batch, _case_value_batches(values), workers
                )
                for record in batch
            ]
    records["canonical_record"] = canonical_records
    return records


def serialize_case(row: pd.Series) -> str:
    return _serialize_case_values(row[column] for column in CASE_COLUMNS)


def _serialize_case_values(context_values: Iterable[Sequence[object] | None]) -> str:
    payload = {}
    for (_, label), values in zip(CASE_FIELDS, context_values):
        if values is None or len(values) == 0:
            values = ["DESCONHECIDO"]
        payload[label] = sorted({str(value) for value in values})
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write_category_regime(
    arguments: tuple[str, pd.DataFrame, str, str, Path, str],
) -> tuple[dict, list[dict]]:
    category, category_records, label, regime, output_dir, replicate = arguments
    category_records = category_records.sort_values(["canonical_record", "source_hash"])
    corpus_path = output_dir / label
    with corpus_path.open("w", encoding="utf-8") as stream:
        stream.writelines(record + "\n" for record in category_records["canonical_record"])

    counts = (
        category_records.groupby("canonical_record", as_index=False)
        .size()
        .rename(columns={"canonical_record": "canonical_combination", "size": "case_count"})
    )
    counts["category"] = category
    counts["regime"] = regime
    counts["replicate"] = replicate
    counts["case_share"] = counts["case_count"] / len(category_records)
    map_row = {
        "label": label,
        "category": category,
        "regime": regime,
        "replicate": replicate,
        "case_count": len(category_records),
        "bytes": corpus_path.stat().st_size,
        "sha256": _sha256(corpus_path),
    }
    return map_row, counts.to_dict(orient="records")


def _write_regime(
    regime: str,
    records: pd.DataFrame,
    output_dir: Path,
    label_by_category: dict[str, str],
    replicate: str,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    map_rows = []
    combination_rows = []

    arguments = (
        (category, category_records, label_by_category[category], regime, output_dir, replicate)
        for category, category_records in records.groupby("category", sort=True)
    )
    if executor is None:
        results = map(_write_category_regime, arguments)
    else:
        results = _ordered_process_map(executor, _write_category_regime, arguments, 2)
    for map_row, counts in results:
        map_rows.append(map_row)
        combination_rows.extend(counts)

    return pd.DataFrame(map_rows), pd.DataFrame(combination_rows)


def build_case_corpora(
    case_records: pd.DataFrame,
    category_map: pd.DataFrame,
    work_dir: Path,
    seeds: Iterable[int],
    metadata_dir: Path | None = None,
    *,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    _validate_workers(workers)
    required_columns = {"source_hash", "category", "canonical_record"}
    if not required_columns.issubset(case_records.columns):
        raise ValueError("case_records must include source_hash, category, and canonical_record.")
    if case_records.duplicated(["source_hash", "category"]).any():
        raise ValueError("case_records is not unique at source_hash + category grain.")

    case_records = case_records[["source_hash", "category", "canonical_record"]]
    case_work_dir = work_dir
    full_dir = case_work_dir / "case_full" / "corpus"
    balanced_dir = case_work_dir / "case_balanced"
    full_dir.mkdir(parents=True, exist_ok=True)
    balanced_dir.mkdir(parents=True, exist_ok=True)

    label_by_category = category_map.set_index("category")["label"].to_dict()
    support = (
        case_records.groupby("category", as_index=False)["source_hash"]
        .nunique()
        .rename(columns={"source_hash": "case_count"})
    )
    balanced_sample_size = int(support["case_count"].min())

    maps = []
    combinations = []

    execution = (
        nullcontext(None)
        if workers == 1
        else ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    )
    with execution as executor:
        full_records = case_records.sort_values(["category", "source_hash"]).copy()
        full_map, full_combinations = _write_regime(
            "case-full", full_records, full_dir, label_by_category, "full", executor
        )
        maps.append(full_map)
        combinations.append(full_combinations)
        del full_records

        category_groups = [
            category_records.sort_values("source_hash")
            for _, category_records in case_records.groupby("category", sort=True)
        ]
        for index, seed in enumerate(seeds, start=1):
            replicate = f"replicate-{index:03d}"
            selected_parts = []
            for category_records in category_groups:
                selected_parts.append(
                    category_records.sample(
                        n=balanced_sample_size,
                        replace=False,
                        random_state=seed,
                    )
                )
            selected = pd.concat(selected_parts, ignore_index=True)
            replicate_map, replicate_combinations = _write_regime(
                "case-balanced",
                selected,
                balanced_dir / replicate,
                label_by_category,
                replicate,
                executor,
            )
            maps.append(replicate_map)
            combinations.append(replicate_combinations)
            del selected, selected_parts

    case_category_map = pd.concat(maps, ignore_index=True)
    case_combination_counts = pd.concat(combinations, ignore_index=True)

    metadata_dir = metadata_dir or case_work_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    case_category_map.to_csv(metadata_dir / "case-category-map.csv", index=False)
    case_combination_counts.to_csv(
        metadata_dir / "case-combination-counts.csv",
        index=False,
    )
    return case_category_map, case_combination_counts, balanced_sample_size


def compare_combination_distributions(
    combination_counts: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    full = combination_counts.loc[
        combination_counts["regime"] == "case-full",
        ["category", "canonical_combination", "case_share"],
    ].rename(columns={"case_share": "full_case_share"})
    balanced = combination_counts.loc[
        combination_counts["regime"] == "case-balanced",
        ["category", "replicate", "canonical_combination", "case_share"],
    ].rename(columns={"case_share": "balanced_case_share"})

    comparison = balanced.merge(
        full,
        on=["category", "canonical_combination"],
        how="outer",
    )
    comparison["balanced_case_share"] = comparison["balanced_case_share"].fillna(0)
    comparison["full_case_share"] = comparison["full_case_share"].fillna(0)
    comparison["absolute_share_difference"] = (
        comparison["balanced_case_share"] - comparison["full_case_share"]
    ).abs()
    comparison.to_csv(output_path, index=False)
    return comparison
