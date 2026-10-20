"""Concatenated deceptive traps with scattered bits, and how close a grouping is to the truth.

A trap of k bits with u ones is worth k when u = k and k - 1 - u otherwise, so every bit
alone points to zero while the optimum is all ones. Only a search that keeps the k bits of
a trap together finds it reliably. Each instance scatters the bits of every trap over the
string at random, so neighbouring positions say nothing about which bits go together.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score

Groups = list[tuple[int, ...]]


@dataclass(frozen=True)
class TrapInstance:
    block_count: int
    trap_size: int
    # blocks[b] holds the variables of trap b.
    blocks: tuple[tuple[int, ...], ...]

    @property
    def variables(self) -> int:
        return self.block_count * self.trap_size

    @property
    def optimum(self) -> int:
        return self.block_count * self.trap_size

    @property
    def labels(self) -> np.ndarray:
        """Trap of each variable."""
        labels = np.empty(self.variables, dtype=int)
        for block, members in enumerate(self.blocks):
            labels[list(members)] = block
        return labels

    def ones_per_block(self, population: np.ndarray) -> np.ndarray:
        return np.stack([population[:, list(block)].sum(axis=1) for block in self.blocks], axis=1)

    def fitness(self, population: np.ndarray) -> np.ndarray:
        ones = self.ones_per_block(population)
        k = self.trap_size
        return np.where(ones == k, k, k - 1 - ones).sum(axis=1)

    def solved_blocks(self, solution: np.ndarray) -> int:
        return int((self.ones_per_block(solution[None, :])[0] == self.trap_size).sum())


def make_instance(block_count: int, trap_size: int, seed: int) -> TrapInstance:
    order = np.random.default_rng([seed, block_count, trap_size]).permutation(block_count * trap_size)
    blocks = tuple(
        tuple(sorted(int(v) for v in order[b * trap_size : (b + 1) * trap_size]))
        for b in range(block_count)
    )
    return TrapInstance(block_count, trap_size, blocks)


def linkage_scores(groups: Groups, instance: TrapInstance) -> dict:
    """Agreement between DAMICORE's groups and the true traps.

    - ``ari``: adjusted Rand index of the two partitions (1 = identical, ~0 = chance).
    - ``pair_precision``: share of pairs grouped together that belong to the same trap.
    - ``pair_recall``: share of same-trap pairs that were grouped together.
    - ``exact_blocks``: traps that DAMICORE returned exactly as one group.
    """
    found = np.empty(instance.variables, dtype=int)
    for group_id, group in enumerate(groups):
        found[list(group)] = group_id
    truth = instance.labels
    grouped = {pair for group in groups for pair in combinations(sorted(group), 2)}
    true_pairs = {pair for block in instance.blocks for pair in combinations(block, 2)}
    hits = len(grouped & true_pairs)
    return {
        "ari": float(adjusted_rand_score(truth, found)),
        "pair_precision": hits / len(grouped) if grouped else float("nan"),
        "pair_recall": hits / len(true_pairs),
        "exact_blocks": len({tuple(sorted(g)) for g in groups} & set(instance.blocks)),
    }
