"""Synthetic checks for the objective, Pareto tools, DAMICORE inputs, and the AED run."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
import zlib

import numpy as np
import pandas as pd

from ..scripts.config import DEFAULT_CONFIG, FEATURES
from ..scripts.fitness import (
    FULL_MASK,
    MaskEvaluator,
    all_masks,
    code_fold,
    exhaustive_scores,
    mask_string,
    score_mask,
    selected_features,
)
from ..scripts.pareto import nondominated_ranks, pareto_front
from ..scripts.search import (
    initial_population,
    probability_tables,
    run_search,
    validate_partition,
    write_feature_series,
)


def synthetic_frame(rows: int, seed: int) -> pd.DataFrame:
    """Field 0 copies the target, field 1 is noise, the rest are constant."""
    rng = np.random.default_rng(seed)
    target = (rng.random(rows) < 0.2).astype(int)
    frame = pd.DataFrame({feature: "SAME" for feature in FEATURES}, index=range(rows))
    frame[FEATURES[0]] = np.where(target == 1, "YES", "NO")
    frame[FEATURES[1]] = rng.choice(["A", "B", "C"], size=rows)
    frame["target"] = target
    return frame


def single(index: int) -> tuple[int, ...]:
    return tuple(int(position == index) for position in range(len(FEATURES)))


class MaskTests(unittest.TestCase):
    def test_selected_features_returns_names_in_order(self) -> None:
        mask = single(0)[:2] + (1,) + (0,) * (len(FEATURES) - 3)
        self.assertEqual(selected_features(mask), [FEATURES[0], FEATURES[2]])
        self.assertEqual(selected_features(FULL_MASK), list(FEATURES))

    def test_all_masks_enumerates_every_nonempty_subset(self) -> None:
        masks = all_masks()
        self.assertEqual(len(masks), 2 ** len(FEATURES) - 1)
        self.assertEqual(len(set(masks)), len(masks))


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fold = code_fold("synthetic", synthetic_frame(4000, 1), synthetic_frame(2000, 2))

    def test_informative_field_gains_close_to_target_entropy(self) -> None:
        gain = score_mask(single(0), [self.fold])["synthetic"]
        rate = self.fold.validation_target.mean()
        entropy = -(rate * np.log2(rate) + (1 - rate) * np.log2(1 - rate))
        self.assertGreater(gain, 0.9 * entropy)

    def test_noise_and_constant_fields_gain_nothing(self) -> None:
        self.assertLess(abs(score_mask(single(1), [self.fold])["synthetic"]), 0.005)
        self.assertLess(abs(score_mask(single(5), [self.fold])["synthetic"]), 1e-9)

    def test_exhaustive_enumeration_matches_direct_scoring(self) -> None:
        table = exhaustive_scores([self.fold], replace(DEFAULT_CONFIG, exhaustive_workers=1))
        self.assertEqual(len(table), 2 ** len(FEATURES) - 1)
        by_mask = table.set_index("mask")["mean_gain_bits"]
        for mask in (single(0), single(1), FULL_MASK, single(0)[:1] + (1,) + (0,) * 13):
            self.assertAlmostEqual(
                by_mask[mask_string(mask)], score_mask(mask, [self.fold])["synthetic"], places=12
            )

    def test_evaluator_matches_direct_scoring(self) -> None:
        evaluator = MaskEvaluator([self.fold])
        for mask in (single(0), single(1), FULL_MASK):
            self.assertAlmostEqual(
                evaluator.quality(mask), score_mask(mask, [self.fold])["synthetic"], places=12
            )


class ParetoTests(unittest.TestCase):
    def test_ranks_and_front(self) -> None:
        points = [(0.5, 1), (0.8, 2), (0.4, 2), (0.9, 5), (0.7, 5)]
        self.assertEqual(list(nondominated_ranks(points)), [0, 0, 1, 0, 1])
        table = pd.DataFrame(
            {
                "mask": ["a", "b", "c", "d", "e"],
                "feature_count": [p[1] for p in points],
                "mean_gain_bits": [p[0] for p in points],
            }
        )
        self.assertEqual(list(pareto_front(table)["mask"]), ["a", "b", "d"])


class DamicoreInputTests(unittest.TestCase):
    def test_series_are_one_byte_per_mask_in_shared_order(self) -> None:
        masks = initial_population(3, replace(DEFAULT_CONFIG, population_size=30))
        with tempfile.TemporaryDirectory() as directory:
            length = write_feature_series(masks, Path(directory))
            for index in range(len(FEATURES)):
                payload = (Path(directory) / f"feature-{index:02d}.txt").read_bytes()
                self.assertEqual(len(payload), length)
                self.assertEqual(payload, bytes(ord("0") + mask[index] for mask in masks))

    def test_ncd_separates_identical_from_independent_series_at_used_lengths(self) -> None:
        def ncd(left: bytes, right: bytes) -> float:
            sizes = [len(zlib.compress(value, 6)) for value in (left, right, left + right)]
            return (sizes[2] - min(sizes[:2])) / max(sizes[:2])

        generator = random.Random(0)
        for length in (120, 336, 500):
            base = bytes(generator.choice(b"01") for _ in range(length))
            other = bytes(generator.choice(b"01") for _ in range(length))
            self.assertLess(ncd(base, base) + 0.3, ncd(base, other))


class SearchTests(unittest.TestCase):
    def test_run_spends_the_budget_with_damicore_blocks(self) -> None:
        weights = np.linspace(1.0, 0.1, len(FEATURES))

        class SyntheticEvaluator:
            def quality(self, mask):
                return float(np.dot(mask, weights) - 0.08 * sum(mask) ** 2)

            def worst_fold(self, mask):
                return self.quality(mask)

        config = replace(
            DEFAULT_CONFIG, population_size=40, generations=2, offspring_per_generation=10
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_search(7, SyntheticEvaluator(), Path(directory), config)
        self.assertEqual(len(result.history), config.evaluations_per_seed)
        self.assertEqual(len({row["mask"] for row in result.history}), len(result.history))
        self.assertEqual(len(result.diagnostics), config.generations)
        for diagnostics in result.diagnostics:
            validate_partition(
                [tuple(FEATURES.index(name) for name in block) for block in diagnostics["blocks"]]
            )

    def test_probability_tables_sum_to_one_per_block(self) -> None:
        masks = initial_population(5, replace(DEFAULT_CONFIG, population_size=30))
        groups = [tuple(range(start, min(start + 4, len(FEATURES)))) for start in range(0, 15, 4)]
        distributions, _ = probability_tables(masks, groups)
        for distribution in distributions:
            self.assertAlmostEqual(float(distribution.sum()), 1.0)
            self.assertTrue((distribution > 0).all())


if __name__ == "__main__":
    unittest.main()
