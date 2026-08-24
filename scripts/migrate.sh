#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_FILE="$REPO_ROOT/data/disque100-primeiro-semestre-2026.csv"
MIGRATION_FILE="$REPO_ROOT/migrations/001_create_disque100_reports.sql"
DB_TARGET="${DISQUE100_DATABASE_URL:-disque100}"

if ! command -v psql >/dev/null 2>&1; then
    echo "Error: psql is not installed or not in PATH." >&2
    exit 1
fi

if [[ ! -r "$CSV_FILE" ]]; then
    echo "Error: CSV file not readable at $CSV_FILE" >&2
    exit 1
fi

if [[ ! -r "$MIGRATION_FILE" ]]; then
    echo "Error: Migration file not readable at $MIGRATION_FILE" >&2
    exit 1
fi

if ! psql -X -v ON_ERROR_STOP=1 "$DB_TARGET" -Atqc "SELECT 1" >/dev/null 2>&1; then
    echo "Error: Could not connect to the target database." >&2
    exit 1
fi

FREE_KIB="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
if (( FREE_KIB < 8388608 )); then
    echo "Error: Less than 8 GiB free space available on the filesystem." >&2
    exit 1
fi

TABLE_EXISTS="$(psql -X -v ON_ERROR_STOP=1 "$DB_TARGET" -Atqc "SELECT to_regclass('public.disque100_reports') IS NOT NULL;")"

if [[ "$TABLE_EXISTS" == "t" ]]; then
    if ! INVARIANTS="$(psql -X -v ON_ERROR_STOP=1 "$DB_TARGET" -Atqc "
        SELECT concat_ws('|',
            count(*),
            count(distinct source_hash),
            to_char(min(registered_at), 'YYYY-MM-DD HH24:MI:SS.MS'),
            to_char(max(registered_at), 'YYYY-MM-DD HH24:MI:SS.MS'),
            count(*) filter (where source_hash is null or source_hash !~ '^(?:[0-9A-F]{32}|[0-9A-F]{64})$'),
            count(*) filter (where registered_at is null),
            count(*) filter (where victim_count is null or victim_count <= 0),
            (select count(*) from pg_tables where schemaname = 'public')
        )
        FROM public.disque100_reports;
    ")"; then
        echo "Error: Table 'disque100_reports' exists but its schema could not be verified." >&2
        exit 1
    fi

    EXPECTED="2825614|371117|2026-01-01 00:02:44.687|2026-06-30 23:59:03.497|0|0|0|1"
    if [[ "$INVARIANTS" == "$EXPECTED" ]]; then
        echo "Migration already applied and verified. Exiting successfully without reimport."
        exit 0
    else
        echo "Error: Table 'disque100_reports' exists but is incomplete or invariants do not match." >&2
        echo "Expected: $EXPECTED" >&2
        echo "Found:    $INVARIANTS" >&2
        exit 1
    fi
fi

echo "Running migration..."
cd "$REPO_ROOT"
psql -X -v ON_ERROR_STOP=1 "$DB_TARGET" -f "$MIGRATION_FILE"
echo "Migration completed successfully."
