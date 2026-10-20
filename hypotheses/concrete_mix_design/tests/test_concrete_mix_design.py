"""Synthetic checks for the search space, Pareto tools, DAMICORE inputs, and the DAMICORE AED."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..scripts.config import DEFAULT_CONFIG
from ..scripts.experiment import ground_truth
from ..scripts.landscape import COMPONENT_COUNT, Landscape, component_levels
from ..scripts.pareto import constrained_order, front_mask, hypervolume, nondominated_ranks
from ..scripts.search import (
    component_series,
    damicore_groups,
    damicore_niches,
    fit_blocks,
    initial_population,
    mix_document,
    run_search,
    sample_solution,
    validate_partition,
)

SMALL = replace(
    DEFAULT_CONFIG,
    levels=3,
    seeds=(1,),
    population_size=40,
    generations=3,
    offspring_per_generation=20,
    archive_size=30,
    max_niches=3,
    min_niche_size=8,
    strength_targets=(5.0,),
    workers=1,
)


def synthetic_landscape(config=SMALL) -> Landscape:
    """Strength rewards components 0 and 1 together (an interaction); cost counts levels."""
    shape = (config.levels,) * COMPONENT_COUNT
    grid = np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=1)
    strength = grid.sum(axis=1) + 3.0 * (grid[:, 0] == grid[:, 1])
    cost = grid.sum(axis=1) + 0.1 * grid[:, 6]
    distance = np.where(grid[:, 2] == config.levels - 1, 2.0, 0.0)  # last level of 2 is infeasible
    return Landscape(
        levels=[np.arange(config.levels, dtype=float)] * COMPONENT_COUNT,
        strength=strength.astype(np.float32),
        cost=cost.astype(np.float32),
        distance=distance.astype(np.float32),
        max_distance=config.max_distance_to_data,
    )


class LevelTests(unittest.TestCase):
    def test_often_absent_component_keeps_zero_level(self) -> None:
        rng = np.random.default_rng(0)
        mixes = np.column_stack(
            [rng.uniform(100, 500, 200), np.where(rng.random(200) < 0.5, 0, rng.uniform(50, 200, 200))]
        )
        levels = component_levels(mixes, 8)
        self.assertEqual(levels[1][0], 0.0)
        self.assertGreater(levels[0][0], 0.0)
        for values in levels:
            self.assertEqual(len(values), 8)
            self.assertTrue(np.all(np.diff(values) > 0))


class ParetoTests(unittest.TestCase):
    def test_hypervolume_of_staircase(self) -> None:
        strength = np.array([2.0, 4.0, 1.0])
        cost = np.array([1.0, 3.0, 2.0])  # the third point is dominated by the first
        # Reference (0, 5): [1, 3) x 2 + [3, 5) x 4 = 4 + 8.
        self.assertAlmostEqual(hypervolume(strength, cost, (0.0, 5.0)), 12.0)

    def test_front_mask_matches_peeling(self) -> None:
        rng = np.random.default_rng(1)
        strength, cost = rng.random(200), rng.random(200)
        np.testing.assert_array_equal(front_mask(strength, cost), nondominated_ranks(strength, cost) == 0)

    def test_feasible_points_come_before_infeasible(self) -> None:
        points = [(1.0, 5.0, 0.3), (9.0, 1.0, 0.0), (2.0, 2.0, 0.0), (8.0, 1.0, 0.1)]
        self.assertEqual(constrained_order(points), [1, 2, 3, 0])


class DamicoreInputTests(unittest.TestCase):
    def test_series_group_mix_ids_by_level(self) -> None:
        promising = [(0, 1, 2, 0, 1, 2, 0), (2, 2, 2, 1, 1, 1, 0), (0, 0, 0, 0, 0, 0, 0)]
        series = component_series(promising)
        self.assertEqual(len(series), COMPONENT_COUNT)
        # Mix ids are two bytes: "!!" = mix 0, "!\"" = mix 1, "!#" = mix 2.
        self.assertEqual(series[0], b'!!!#|!"|')
        self.assertEqual(series[6], b'!!!"!#|')

    def test_relabelled_components_write_the_same_series(self) -> None:
        """Opposite levels split the mixes the same way, so NCD sees the same runs."""
        rng = np.random.default_rng(5)
        promising = []
        for _ in range(40):
            level = int(rng.integers(0, 3))
            promising.append((level, 2 - level, *(int(v) for v in rng.integers(0, 3, 5))))
        series = component_series(promising)
        self.assertEqual(sorted(series[0][:-1].split(b"|")), sorted(series[1][:-1].split(b"|")))

    def test_mix_document_repeats_letters_by_level(self) -> None:
        self.assertEqual(mix_document((0, 2, 0, 0, 0, 0, 1)), b"A.BBB.C.D.E.F.GG.")

    def test_damicore_recovers_planted_groups(self) -> None:
        rng = np.random.default_rng(6)
        promising = []
        for _ in range(100):
            a, b = (int(v) for v in rng.integers(0, 8, 2))
            noise = [int(v) for v in rng.integers(0, 8, 3)]
            promising.append((a, 7 - a, b, (b + 3) % 8, *noise))
        with tempfile.TemporaryDirectory() as directory:
            groups, diagnostics = damicore_groups(promising, Path(directory))
        self.assertEqual(diagnostics["damicore_status"], "ok")
        self.assertTrue(any({0, 1} <= set(group) for group in groups))
        self.assertTrue(any({2, 3} <= set(group) for group in groups))

    def test_constant_and_duplicate_series_still_partition(self) -> None:
        rng = np.random.default_rng(2)
        promising = []
        for _ in range(60):
            a, b, c = (int(value) for value in rng.integers(0, 3, 3))
            promising.append((a, a, b, c, 0, int(rng.integers(0, 3)), int(rng.integers(0, 3))))
        with tempfile.TemporaryDirectory() as directory:
            groups, diagnostics = damicore_groups(promising, Path(directory))
        validate_partition(groups)
        self.assertIn((4,), groups)  # constant component stays alone
        self.assertTrue(any({0, 1} <= set(group) for group in groups))  # duplicates share a block
        self.assertEqual(diagnostics["constant_components"], 1)
        self.assertEqual(diagnostics["duplicate_components"], 1)

    def test_failed_damicore_reuses_fallback_groups(self) -> None:
        promising = [(0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1)]  # one distinct series
        fallback = [(0, 1, 2), (3, 4), (5, 6)]
        with tempfile.TemporaryDirectory() as directory:
            groups, diagnostics = damicore_groups(promising, Path(directory), fallback=fallback)
        self.assertEqual(groups, fallback)
        self.assertIn("reused earlier DAMICORE groups", diagnostics["damicore_status"])

    def test_niches_separate_distinct_regions_and_respect_sizes(self) -> None:
        rng = np.random.default_rng(7)
        low = [tuple(int(v) for v in rng.integers(0, 2, COMPONENT_COUNT)) for _ in range(20)]
        high = [tuple(int(v) for v in rng.integers(6, 8, COMPONENT_COUNT)) for _ in range(20)]
        config = replace(SMALL, max_niches=2, min_niche_size=8)
        with tempfile.TemporaryDirectory() as directory:
            niches, diagnostics = damicore_niches(low + high, Path(directory), config)
        self.assertEqual(sorted(p for niche in niches for p in niche), list(range(40)))
        self.assertLessEqual(len(niches), 2)
        self.assertTrue(all(len(niche) >= 8 for niche in niches))
        if len(niches) == 2:
            self.assertEqual({frozenset(n) for n in niches}, {frozenset(range(20)), frozenset(range(20, 40))})
        self.assertNotIn("failed", diagnostics["niche_status"])

    def test_sampling_draws_only_observed_patterns(self) -> None:
        promising = [(0, 0, 1, 1, 2, 2, 0), (1, 1, 0, 0, 2, 2, 1)]
        groups = [(0, 1), (2, 3), (4, 5, 6)]
        models = fit_blocks(promising, groups)
        rng = np.random.default_rng(3)
        for _ in range(50):
            solution = sample_solution(rng, models)
            self.assertEqual(solution[0], solution[1])
            self.assertIn(solution[2:4], {(1, 1), (0, 0)})


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.landscape = synthetic_landscape()
        cls.directory = tempfile.TemporaryDirectory()
        cls.result = run_search(1, cls.landscape, Path(cls.directory.name), SMALL)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_spends_the_distinct_budget_from_the_initial_population(self) -> None:
        solutions = [row["solution"] for row in self.result.history]
        self.assertEqual(len(solutions), SMALL.evaluations_per_run)
        self.assertEqual(len(set(solutions)), len(solutions))
        start = [row["solution"] for row in self.result.history if row["generation"] == 0]
        self.assertEqual(len(start), len(initial_population(1, SMALL)))

    def test_damicore_runs_every_generation_at_archive_scope(self) -> None:
        archive = [row for row in self.result.diagnostics if row["scope"] == "archive"]
        self.assertEqual(len(archive), SMALL.generations)
        for generation in range(1, SMALL.generations + 1):
            scopes = {row["scope"] for row in self.result.blocks if row["generation"] == generation}
            self.assertIn("archive", scopes)
            self.assertTrue(any(scope.startswith("niche-") for scope in scopes))

    def test_niches_cover_the_promising_mixes(self) -> None:
        for generation in range(1, SMALL.generations + 1):
            rows = [row for row in self.result.niches if row["generation"] == generation]
            self.assertEqual(len(rows), rows[0]["niche_count"])
            self.assertLessEqual(len(rows), SMALL.max_niches)

    def test_offspring_come_from_damicore_models(self) -> None:
        origins = {row["origin"] for row in self.result.history if row["generation"] > 0}
        self.assertTrue(origins <= {"niche_sample", "other_niche_sample", "archive_sample", "uniform_fallback"})
        self.assertIn("niche_sample", origins)

    def test_ground_truth_front_is_feasible_and_nondominated(self) -> None:
        truth = ground_truth(self.landscape, SMALL)
        strength = truth.front["strength_mpa"].to_numpy()
        cost = truth.front["cost_usd_m3"].to_numpy()
        self.assertTrue(front_mask(strength, cost).all())
        self.assertTrue(all(solution[2] != str(SMALL.levels - 1) for solution in truth.front["solution"]))
        self.assertGreater(truth.hypervolume, 0)


if __name__ == "__main__":
    unittest.main()
