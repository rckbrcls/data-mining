"""Checks for the landscape, the DAMICORE inputs, the AED, and the codon test."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ..scripts.config import DEFAULT_CONFIG
from ..scripts.experiment import _triple_partitions, codon_test, pair_frequency
from ..scripts.landscape import CODONS, decode, encode, load_landscape
from ..scripts.search import damicore_groups, position_series, run_search, sample_offspring

SMALL = replace(DEFAULT_CONFIG, budget=400, workers=1)
LANDSCAPE = load_landscape()


class LandscapeTests(unittest.TestCase):
    def test_wild_type_is_zero_and_encoding_round_trips(self) -> None:
        wild_type = encode(pd.Series(["GCCGATCTC"]))
        self.assertEqual(decode(wild_type[0]), "GCCGATCTC")
        self.assertAlmostEqual(float(LANDSCAPE.evaluate(wild_type)[0]), 0.0, places=6)

    def test_every_sequence_has_a_fitness(self) -> None:
        self.assertEqual(LANDSCAPE.fitness.size, 4**9)
        self.assertEqual(int(LANDSCAPE.measured.sum()), 261_333)
        self.assertFalse(np.isnan(LANDSCAPE.fitness).any())


class DamicoreInputTests(unittest.TestCase):
    def test_series_ignore_base_names(self) -> None:
        promising = np.array([[0, 2, 1], [1, 3, 1], [0, 2, 3]])
        series = position_series(promising)
        runs = [set(s.split(b"|")) - {b""} for s in series]
        self.assertEqual(runs[0], runs[1])  # same split, different bases: same runs
        self.assertNotEqual(runs[0], runs[2])

    def test_linked_positions_are_grouped(self) -> None:
        rng = np.random.default_rng(0)
        codon = rng.integers(0, 4, (60, 1))
        promising = np.hstack([codon, (codon + 1) % 4, (codon + 2) % 4, rng.integers(0, 4, (60, 6))])
        promising[:5, 1] = rng.integers(0, 4, 5)  # not identical copies: DAMICORE must decide
        with tempfile.TemporaryDirectory() as folder:
            groups, diagnostics = damicore_groups(promising, Path(folder))
            self.assertTrue((Path(folder) / "tree.nwk").exists())
        self.assertEqual(sorted(p for g in groups for p in g), list(range(9)))
        self.assertTrue(any({0, 2} <= set(g) for g in groups), groups)

    def test_offspring_copy_whole_group_patterns(self) -> None:
        promising = np.array([[0, 0, 3], [1, 1, 2]])
        offspring = sample_offspring(np.random.default_rng(1), promising, [(0, 1), (2,)], 50)
        np.testing.assert_array_equal(offspring[:, 0], offspring[:, 1])


class SearchTests(unittest.TestCase):
    def test_run_respects_budget_and_improves(self) -> None:
        result = run_search(LANDSCAPE, seed=3, config=SMALL)
        self.assertEqual(result.history[-1]["evaluations"], SMALL.budget)
        self.assertGreater(result.best_fitness, np.median(LANDSCAPE.fitness[LANDSCAPE.measured]))
        best = encode(pd.Series([result.best_sequence]))
        self.assertAlmostEqual(float(LANDSCAPE.evaluate(best)[0]), result.best_fitness)


class CodonTestTests(unittest.TestCase):
    def test_partitions_are_distinct(self) -> None:
        partitions = _triple_partitions()
        self.assertEqual(len({frozenset(map(frozenset, p)) for p in partitions}), 280)
        self.assertIn(CODONS, partitions)

    def test_codon_grouping_ranks_first(self) -> None:
        generations = pd.DataFrame({"groups": ["1-2-3 4-5-6 7-8-9", "1-2 3 4-5-6 7-8 9"]})
        result = codon_test(pair_frequency(generations))
        self.assertEqual(result["codon_rank"], 1)
        self.assertAlmostEqual(result["p_value"], 1 / 280)


if __name__ == "__main__":
    unittest.main()
