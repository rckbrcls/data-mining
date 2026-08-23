from __future__ import annotations

import os
from typing import Any, Sequence

import pandas as pd
import psycopg

from .config import ExperimentConfig


def query_frame(
    config: ExperimentConfig,
    query: Any,
    parameters: Sequence[Any] = (),
) -> pd.DataFrame:
    """Execute a bounded query using the configured environment variable."""
    database_url = os.getenv(config.database_url_env)
    if not database_url:
        raise RuntimeError(
            f"Set {config.database_url_env} before running database-backed notebooks."
        )

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            if cursor.description is None:
                return pd.DataFrame()
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
    return pd.DataFrame(rows, columns=columns)
