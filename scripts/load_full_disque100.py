#!/usr/bin/env python3
"""Load every Disque 100 CSV into the canonical PostgreSQL table."""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COLUMNS = [
    "source_hash", "registered_at", "service_channel", "emergency_status",
    "reporter_type", "violation_setting", "country", "state", "municipality",
    "frequency", "violation_start_period", "victim_count", "vulnerable_group",
    "motivation", "victim_suspect_relationship", "victim_entity_type",
    "victim_gender", "victim_sexual_orientation", "victim_age_group",
    "victim_nationality", "victim_birthplace", "victim_naturalized_state",
    "victim_naturalized_municipality", "victim_disability", "victim_rare_disease",
    "victim_disability_related_to_rare_disease", "victim_incarceration_status",
    "victim_country", "victim_state", "victim_municipality", "victim_occupation",
    "victim_education_level", "victim_religion", "victim_race_color",
    "victim_ethnicity", "victim_income_range", "suspect_legal_nature",
    "suspect_gender", "suspect_sexual_orientation", "suspect_age_group",
    "suspect_nationality", "suspect_birthplace", "suspect_naturalized_state",
    "suspect_naturalized_municipality", "suspect_disability", "suspect_rare_disease",
    "suspect_disability_related_to_rare_disease", "suspect_incarceration_status",
    "suspect_country", "suspect_state", "suspect_municipality", "suspect_occupation",
    "suspect_education_level", "suspect_religion", "suspect_race_color",
    "suspect_ethnicity", "suspect_income_range", "suspect_organization_relationship",
    "suspect_business_sector", "suspect_ethnicity_details", "victim_ethnicity_details",
    "violation",
]

HASH_RE = re.compile(r"^(?:[0-9A-F]{32}|[0-9A-F]{64})$")


def clean(value):
    value = value.strip()
    return None if value in {"", "NULL"} else value


def timestamp(value):
    value = clean(value)
    if value is None:
        raise ValueError("registered_at is empty")
    if "/" in value:
        parsed = datetime.strptime(value, "%d/%m/%Y %H:%M")
    else:
        parsed = datetime.fromisoformat(value.replace("T", " "))
    return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def canonical_row(row):
    if len(row) == 34:
        old = [clean(value) for value in row]
        result = [None] * len(COLUMNS)
        mapping = [
            (0, 0), (1, 1), (4, 2), (24, 3), (2, 4), (9, 5), (5, 6),
            (6, 7), (8, 8), (10, 12), (31, 13), (23, 14), (11, 16),
            (33, 17), (12, 18), (25, 19), (5, 27), (6, 28), (8, 29),
            (13, 31), (15, 33), (28, 34), (14, 35), (17, 37), (18, 39),
            (26, 40), (22, 44), (27, 48), (19, 52), (21, 54), (29, 55),
            (20, 56), (30, 61),
        ]
        for source_index, target_index in mapping:
            result[target_index] = old[source_index]
        result[11] = "1"
    elif len(row) == 62:
        result = [clean(value) for value in row]
        result[1] = timestamp(result[1])
        result[11] = result[11] or "1"
    else:
        raise ValueError(f"unsupported column count: {len(row)}")

    result[0] = (result[0] or "").upper()
    if not HASH_RE.fullmatch(result[0]):
        raise ValueError(f"invalid source_hash: {result[0]!r}")
    result[1] = timestamp(result[1])
    try:
        if int(result[11]) <= 0:
            result[11] = "1"
    except (TypeError, ValueError):
        result[11] = "1"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--container", default="disque100-postgres")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("disque100-*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    copy_sql = (
        "COPY public.disque100_reports ("
        + ", ".join(COLUMNS)
        + ") FROM STDIN WITH (FORMAT csv, DELIMITER ';', NULL 'NULL')"
    )
    process = subprocess.Popen(
        ["docker", "exec", "-i", args.container, "psql", "-X",
         "-v", "ON_ERROR_STOP=1", "-U", "postgres", "-d", "disque100",
         "-c", copy_sql],
        stdin=subprocess.PIPE,
        text=True,
    )
    writer = csv.writer(process.stdin, delimiter=";", lineterminator="\n")
    total = 0
    try:
        for path in files:
            count = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=";")
                next(reader)
                for row in reader:
                    writer.writerow(canonical_row(row))
                    count += 1
                    total += 1
                    if total % 250_000 == 0:
                        print(f"Loaded {total:,} rows", file=sys.stderr, flush=True)
            print(f"{path.name}: {count:,} rows", file=sys.stderr, flush=True)
        process.stdin.close()
        status = process.wait()
    except Exception:
        process.stdin.close()
        process.terminate()
        process.wait()
        raise
    if status:
        raise SystemExit(f"COPY failed with exit code {status}")

    index_sql = """
    CREATE INDEX IF NOT EXISTS idx_disque100_source_hash ON public.disque100_reports (source_hash);
    CREATE INDEX IF NOT EXISTS idx_disque100_registered_at ON public.disque100_reports (registered_at);
    CREATE INDEX IF NOT EXISTS idx_disque100_state_hash ON public.disque100_reports (state, source_hash);
    CREATE INDEX IF NOT EXISTS idx_disque100_vulnerable_hash ON public.disque100_reports (vulnerable_group, source_hash);
    CREATE INDEX IF NOT EXISTS idx_disque100_violation_hash ON public.disque100_reports (violation, source_hash);
    ANALYZE public.disque100_reports;
    """
    subprocess.run(
        ["docker", "exec", args.container, "psql", "-X", "-v", "ON_ERROR_STOP=1",
         "-U", "postgres", "-d", "disque100", "-c", index_sql],
        check=True,
    )
    print(f"Completed: {total:,} rows")


if __name__ == "__main__":
    main()
