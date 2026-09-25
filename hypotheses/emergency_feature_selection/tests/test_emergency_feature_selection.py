"""Small synthetic checks for report preparation and equal-budget search."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import nbformat
import psycopg

from ..pipeline.config import DEFAULT_CONFIG, FEATURES, KNOWN_STATUSES, NEGATIVE_STATUS
from ..pipeline.data import prepare_data, report_query, stratified_limits
from ..pipeline.experiment import execute_experiment
from ..pipeline.model import FULL_MASK, validate_mask
from ..pipeline.search import (
    block_distributions,
    damicore_groups,
    initial_pool,
    run_search,
    validate_partition,
)


class PreparationTests(unittest.TestCase):
    def test_fixed_stratified_allocation(self) -> None:
        self.assertEqual(stratified_limits({0: 90, 1: 10}, 20), {0: 18, 1: 2})
        self.assertEqual(stratified_limits({0: 3, 1: 97}, 20), {0: 1, 1: 19})
        self.assertEqual(stratified_limits({0: 3, 1: 2}, None), {0: None, 1: None})

    def test_forbidden_predictors_not_selected(self) -> None:
        self.assertEqual(len(FEATURES), 15)
        self.assertFalse(
            {"emergency_status", "violation", "registered_at", "source_hash"}
            .intersection(FEATURES)
        )
        self.assertIn("GROUP BY source_hash", report_query())
        self.assertEqual(sum(validate_mask(FULL_MASK)), 15)


class TemporaryPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not shutil.which("initdb") or not shutil.which("pg_ctl"):
            raise unittest.SkipTest("PostgreSQL server binaries are unavailable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="emergency-postgres-")
        root = Path(cls.temporary.name)
        cls.data_dir = root / "data"
        cls.socket_dir = root / "socket"
        cls.socket_dir.mkdir()
        initialization = subprocess.run(
            ["initdb", "-A", "trust", "-D", str(cls.data_dir), "--no-instructions"],
            capture_output=True, text=True,
        )
        if initialization.returncode:
            raise RuntimeError(initialization.stderr)
        cls.port = 40000 + secrets.randbelow(20000)
        start = subprocess.run(
            ["pg_ctl", "-D", str(cls.data_dir), "-l", str(root / "server.log"), "-o",
             f"-k {cls.socket_dir} -p {cls.port} -c listen_addresses=''",
             "-w", "start"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        if start.returncode:
            raise RuntimeError((root / "server.log").read_text(encoding="utf-8"))
        cls.dsn = f"host={cls.socket_dir} port={cls.port} dbname=postgres"
        with psycopg.connect(cls.dsn) as connection:
            with connection.cursor() as cursor:
                fields = ", ".join(f"{feature} text" for feature in FEATURES)
                cursor.execute(
                    "CREATE TABLE public.disque100_reports ("
                    f"source_hash text, registered_at timestamp, emergency_status text, {fields})"
                )
                columns = ("source_hash", "registered_at", "emergency_status", *FEATURES)
                placeholders = ", ".join("%s" for _ in columns)
                insert = f"INSERT INTO public.disque100_reports VALUES ({placeholders})"

                def add(hash_value: str, date: str, status: str | None,
                        channel: str = "WEB") -> None:
                    values = {feature: "VALUE" for feature in FEATURES}
                    values["service_channel"] = channel
                    cursor.execute(
                        insert,
                        (hash_value, date, status, *(values[field] for field in FEATURES)),
                    )

                for prefix, date in (("train", "2024-02-01"),
                                     ("validation", "2026-02-01"),
                                     ("test", "2026-05-01")):
                    for number in range(4):
                        status = NEGATIVE_STATUS if number < 2 else KNOWN_STATUSES[1]
                        add(f"{prefix}-{number}", date, status)
                add("train-0", "2024-02-01", NEGATIVE_STATUS, "PHONE")
                add("conflict", "2024-03-01", NEGATIVE_STATUS)
                add("conflict", "2024-03-01", KNOWN_STATUSES[1])
                add("unknown", "2024-03-01", None)
                add("unsupported", "2024-03-01", "UNDEFINED")
                add("date-conflict", "2024-03-01", NEGATIVE_STATUS)
                add("date-conflict", "2024-03-02", NEGATIVE_STATUS)
                add("mixed-null", "2024-03-01", NEGATIVE_STATUS)
                add("mixed-null", "2024-03-01", None)

    @classmethod
    def tearDownClass(cls) -> None:
        subprocess.run(
            ["pg_ctl", "-D", str(cls.data_dir), "-m", "immediate", "-w", "stop"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30,
        )
        cls.temporary.cleanup()

    def test_report_aggregation_and_target(self) -> None:
        config = replace(DEFAULT_CONFIG, search_train_limit=4,
                         validation_limit=4, final_train_limit=8)
        prepared = prepare_data(self.dsn, config)
        self.assertEqual(prepared.audit["reports_in_scope"], 17)
        self.assertEqual(prepared.audit["usable_reports"], 12)
        self.assertEqual(prepared.audit["excluded_reports"], 5)
        self.assertEqual(prepared.audit["date_conflicts"], 1)
        self.assertEqual(prepared.audit["status_conflicts"], 2)
        self.assertEqual(prepared.audit["unknown_status"], 1)
        self.assertEqual(prepared.audit["unsupported_status"], 1)
        self.assertEqual(prepared.audit["feature_conflicts"]["service_channel"], 1)
        self.assertIn("MULTIPLE", set(prepared.search_train.service_channel))
        for frame in (prepared.search_train, prepared.validation,
                      prepared.final_train, prepared.test):
            self.assertEqual(set(frame.columns), {*FEATURES, "target"})
            self.assertEqual(set(frame.target), {0, 1})
            self.assertNotIn("source_hash", frame.columns)
        self.assertEqual(len(prepared.test), 4)

    def test_end_to_end_search_and_aggregate_artifacts(self) -> None:
        config = replace(DEFAULT_CONFIG, search_train_limit=4,
                         validation_limit=4, final_train_limit=8,
                         initial_pool_size=18, elite_size=8,
                         generations=2, offspring_per_generation=4,
                         seeds=(7,))
        prepared = prepare_data(self.dsn, config)
        with tempfile.TemporaryDirectory() as directory:
            results = execute_experiment(prepared, config, Path(directory))
            self.assertEqual(set(results.validation.method),
                             {"all_fields", "greedy_forward", "random",
                              "independent", "damicore"})
            self.assertEqual(set(results.test.method), set(results.validation.method))
            self.assertEqual(set(results.history.evaluations), {18, 22, 26})
            self.assertEqual(
                results.history.groupby("method").evaluations.max().to_dict(),
                {"random": 26, "independent": 26, "damicore": 26},
            )
            self.assertTrue((results.output_dir / "audit.json").exists())
            self.assertTrue((results.output_dir / "damicore_blocks.csv").exists())
            self.assertNotIn("source_hash", " ".join(
                path.read_text(encoding="utf-8")
                for path in results.output_dir.glob("*.csv")
            ))

    def test_all_notebook_code_cells_on_synthetic_database(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "emergency_feature_selection.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        namespace: dict = {"__name__": "__main__"}
        previous_url = os.environ.get("DISQUE100_DATABASE_URL")
        os.environ["DISQUE100_DATABASE_URL"] = self.dsn
        try:
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                first_code_cell = True
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    exec(compile(cell.source, str(notebook_path), "exec"), namespace)
                    if first_code_cell:
                        first_code_cell = False
                        namespace["DEFAULT_CONFIG"] = replace(
                            DEFAULT_CONFIG, search_train_limit=4,
                            validation_limit=4, final_train_limit=8,
                            initial_pool_size=18, elite_size=8,
                            generations=1, offspring_per_generation=4,
                            seeds=(7,),
                        )
                        original_execute = namespace["execute_experiment"]
                        namespace["execute_experiment"] = (
                            lambda prepared, config: original_execute(
                                prepared, config, Path(directory)
                            )
                        )
                self.assertTrue(
                    (namespace["results"].output_dir / "search_evolution.png").exists()
                )
        finally:
            if previous_url is None:
                os.environ.pop("DISQUE100_DATABASE_URL", None)
            else:
                os.environ["DISQUE100_DATABASE_URL"] = previous_url


class CheapEvaluator:
    def __init__(self) -> None:
        self.masks = set()

    def evaluate(self, mask):
        self.masks.add(mask)
        return self

    def rank(self, mask):
        self.evaluate(mask)
        return (abs(sum(mask) - 4), sum(mask), mask)

    def is_acceptable(self, mask):
        return sum(mask) == 4

    def as_dict(self):
        return {"log_loss": 0.1}


class SearchTests(unittest.TestCase):
    def test_probability_blocks_and_equal_budgets(self) -> None:
        config = replace(DEFAULT_CONFIG, initial_pool_size=18, elite_size=8,
                         generations=2, offspring_per_generation=4)
        groups = [tuple(range(0, 4)), tuple(range(4, 8)),
                  tuple(range(8, 12)), tuple(range(12, 15))]
        validate_partition(groups, config)
        elite = initial_pool(7, config)[:config.elite_size]
        distributions = block_distributions(elite, groups, config)
        for distribution in distributions:
            self.assertAlmostEqual(float(distribution.sum()), 1.0)
            self.assertTrue(np.all(distribution > 0))

        def fixed_groups(*_args):
            return groups

        with tempfile.TemporaryDirectory() as directory:
            results = [
                run_search(method, 7, CheapEvaluator(), Path(directory), config,
                           group_builder=fixed_groups)
                for method in ("random", "independent", "damicore")
            ]
            repeated = run_search("damicore", 7, CheapEvaluator(), Path(directory),
                                  config, group_builder=fixed_groups)
        self.assertTrue(all(len(result.evaluated_masks) == 26 for result in results))
        self.assertEqual(
            [result.evaluated_masks[:18] for result in results],
            [results[0].evaluated_masks[:18]] * 3,
        )
        self.assertEqual(results[2].evaluated_masks, repeated.evaluated_masks)

    def test_real_damicore_partitions_all_decisions(self) -> None:
        config = replace(DEFAULT_CONFIG, initial_pool_size=32, elite_size=16)
        elite = initial_pool(11, config)[:16]
        with tempfile.TemporaryDirectory() as directory:
            groups = damicore_groups(elite, Path(directory), 11, 1, config)
        validate_partition(groups, config)


if __name__ == "__main__":
    unittest.main()
