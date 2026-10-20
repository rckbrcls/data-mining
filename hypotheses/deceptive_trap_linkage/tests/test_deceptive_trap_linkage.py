"""Checks for the trap function, the linkage scores, the DAMICORE inputs, and the AED."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..scripts.config import DEFAULT_CONFIG
from ..scripts.search import damicore_groups, run_search, sample_offspring, variable_series
from ..scripts.trap import linkage_scores, make_instance

SMALL = replace(DEFAULT_CONFIG, max_generations=15, stall_generations=5, workers=1)


class TrapTests(unittest.TestCase):
    def test_trap_values_and_optimum(self) -> None:
        instance = make_instance(2, 5, seed=0)
        population = np.zeros((3, 10), dtype=np.int8)
        population[1, list(instance.blocks[0])] = 1  # one trap all ones, the other all zeros
        population[2, list(instance.blocks[0])[:4]] = 1  # four ones: the worst value of a trap
        np.testing.assert_array_equal(instance.fitness(population), [8, 9, 4])
        self.assertEqual(instance.fitness(np.ones((1, 10), dtype=np.int8))[0], instance.optimum)

    def test_blocks_partition_and_are_scattered(self) -> None:
        instance = make_instance(6, 5, seed=3)
        self.assertEqual(sorted(v for b in instance.blocks for v in b), list(range(30)))
        self.assertNotEqual(instance.blocks[0], tuple(range(5)))


class LinkageScoreTests(unittest.TestCase):
    def test_true_grouping_scores_one(self) -> None:
        instance = make_instance(4, 5, seed=1)
        scores = linkage_scores(list(instance.blocks), instance)
        self.assertAlmostEqual(scores["ari"], 1.0)
        self.assertEqual((scores["pair_precision"], scores["pair_recall"]), (1.0, 1.0))
        self.assertEqual(scores["exact_blocks"], 4)

    def test_singletons_recall_nothing(self) -> None:
        instance = make_instance(4, 5, seed=1)
        scores = linkage_scores([(v,) for v in range(20)], instance)
        self.assertEqual(scores["pair_recall"], 0.0)
        self.assertEqual(scores["exact_blocks"], 0)


class DamicoreInputTests(unittest.TestCase):
    def test_series_list_string_ids_by_value(self) -> None:
        promising = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=np.int8)
        series = variable_series(promising)
        self.assertEqual(series[0], b'!!!!!#|!!"|')  # strings 0 and 2 have a 0, string 1 a 1
        self.assertEqual(series[1], b'|!!!!!"!!#|')

    def test_damicore_recovers_traps_from_selected_strings(self) -> None:
        """Strings whose traps are each all ones or all zeros, with 5% of the bits flipped."""
        instance = make_instance(4, 5, seed=2)
        scores = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            promising = np.zeros((300, 20), dtype=np.int8)
            for block in instance.blocks:
                promising[:, list(block)] = rng.integers(0, 2, (300, 1))
            promising = np.where(rng.random(promising.shape) < 0.05, 1 - promising, promising)
            groups, diagnostics = damicore_groups(promising)
            self.assertEqual(diagnostics["damicore_status"], "ok")
            scores.append(linkage_scores(groups, instance)["ari"])
        self.assertGreater(np.mean(scores), 0.85)

    def test_identical_series_are_one_group_each(self) -> None:
        """Without noise every trap is one repeated series; a single DAMICORE cluster is not kept."""
        instance = make_instance(4, 5, seed=2)
        rng = np.random.default_rng(0)
        promising = np.zeros((300, 20), dtype=np.int8)
        for block in instance.blocks:
            promising[:, list(block)] = rng.integers(0, 2, (300, 1))
        groups, diagnostics = damicore_groups(promising)
        self.assertEqual(diagnostics["duplicate_variables"], 16)
        self.assertEqual(sorted(groups), sorted(instance.blocks))

    def test_constant_variables_stay_alone(self) -> None:
        rng = np.random.default_rng(1)
        promising = rng.integers(0, 2, (100, 8), dtype=np.int8)
        promising[:, 3] = 1
        groups, diagnostics = damicore_groups(promising)
        self.assertIn((3,), groups)
        self.assertEqual(diagnostics["constant_variables"], 1)

    def test_sampling_keeps_group_patterns(self) -> None:
        promising = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.int8)
        offspring = sample_offspring(np.random.default_rng(2), promising, [(0, 1), (2, 3)], 50)
        np.testing.assert_array_equal(offspring[:, 0], offspring[:, 1])
        np.testing.assert_array_equal(offspring[:, 2], offspring[:, 3])


class SearchTests(unittest.TestCase):
    def test_small_trap_is_solved_and_linkage_found(self) -> None:
        instance = make_instance(4, 5, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            result = run_search(instance, 3_000, seed=1, config=SMALL, keep_dir=Path(directory))
            kept = sorted(p.name for p in Path(directory, "generation-01").iterdir())
        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "optimum")
        self.assertEqual(result.evaluations, 3_000 * (result.generations + 1))
        self.assertGreater(max(row["ari"] for row in result.history), 0.9)
        self.assertEqual(kept, ["groups.json", "ncd_matrix.csv", "tree.nwk"])


if __name__ == "__main__":
    unittest.main()
