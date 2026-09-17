from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

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

    expected_columns = {"source_hash", "category", *CASE_COLUMNS}
    if set(records.columns) != expected_columns:
        raise ValueError(f"Unexpected case-record columns: {records.columns.tolist()}")
    if records.duplicated(["source_hash", "category"]).any():
        raise ValueError("The case corpus contains duplicate source_hash + category records.")
    if set(records["category"]) != set(categories):
        raise ValueError("The case corpus category set differs from the normalized experiment.")

    records["canonical_record"] = records.apply(serialize_case, axis=1)
    return records


def serialize_case(row: pd.Series) -> str:
    payload = {}
    for column, label in CASE_FIELDS:
        values = row[column]
        if values is None or len(values) == 0:
            values = ["DESCONHECIDO"]
        payload[label] = sorted({str(value) for value in values})
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_regime(
    regime: str,
    records: pd.DataFrame,
    output_dir: Path,
    label_by_category: dict[str, str],
    replicate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    map_rows = []
    combination_rows = []

    for category, category_records in records.groupby("category", sort=True):
        label = label_by_category[category]
        category_records = category_records.sort_values(
            ["canonical_record", "source_hash"]
        )
        corpus_path = output_dir / label
        corpus_path.write_text(
            "\n".join(category_records["canonical_record"].tolist()) + "\n",
            encoding="utf-8",
        )

        counts = (
            category_records.groupby("canonical_record", as_index=False)
            .size()
            .rename(columns={"canonical_record": "canonical_combination", "size": "case_count"})
        )
        counts["category"] = category
        counts["regime"] = regime
        counts["replicate"] = replicate
        counts["case_share"] = counts["case_count"] / len(category_records)
        combination_rows.extend(counts.to_dict(orient="records"))

        map_rows.append(
            {
                "label": label,
                "category": category,
                "regime": regime,
                "replicate": replicate,
                "case_count": len(category_records),
                "bytes": corpus_path.stat().st_size,
                "sha256": _sha256(corpus_path),
            }
        )

    return pd.DataFrame(map_rows), pd.DataFrame(combination_rows)


def build_case_corpora(
    case_records: pd.DataFrame,
    category_map: pd.DataFrame,
    work_dir: Path,
    seeds: Iterable[int],
    metadata_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    required_columns = {"source_hash", "category", "canonical_record"}
    if not required_columns.issubset(case_records.columns):
        raise ValueError("case_records must include source_hash, category, and canonical_record.")
    if case_records.duplicated(["source_hash", "category"]).any():
        raise ValueError("case_records is not unique at source_hash + category grain.")

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

    full_records = case_records.sort_values(["category", "source_hash"]).copy()
    full_map, full_combinations = _write_regime(
        "case-full", full_records, full_dir, label_by_category, "full"
    )
    maps.append(full_map)
    combinations.append(full_combinations)

    for index, seed in enumerate(seeds, start=1):
        replicate = f"replicate-{index:03d}"
        selected_parts = []
        for category, category_records in case_records.groupby("category", sort=True):
            category_records = category_records.sort_values("source_hash")
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
        )
        maps.append(replicate_map)
        combinations.append(replicate_combinations)

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
