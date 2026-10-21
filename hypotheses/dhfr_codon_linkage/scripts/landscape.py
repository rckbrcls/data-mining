"""The measured DHFR landscape of Papkou et al. (2023) and the structure it is known to have.

Nine nucleotides of the E. coli folA gene (codons 26, 27, and 28 of DHFR) were varied over
all 4^9 = 262,144 combinations, and the growth of each variant under trimethoprim was
measured (fitness relative to the wild type, log scale). 261,333 sequences were measured;
the rest count as dead (the lowest measured fitness).

The known structure: positions 1-3, 4-6, and 7-9 are three codons, each read as one amino
acid, so the bases inside a codon only make sense together.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH

BASES = "ACGT"
POSITIONS = 9
CODONS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
_POWERS = 4 ** np.arange(POSITIONS - 1, -1, -1)


def encode(sequences: pd.Series) -> np.ndarray:
    """Strings such as "GCCGATCTC" -> integer matrix (n, 9) with A=0, C=1, G=2, T=3."""
    return np.stack([sequences.str[p].map(BASES.index).to_numpy() for p in range(POSITIONS)], axis=1)


def decode(solution: np.ndarray) -> str:
    return "".join(BASES[int(v)] for v in solution)


@dataclass(frozen=True)
class Landscape:
    # fitness[index] for every one of the 4^9 sequences; unmeasured ones hold `dead`.
    fitness: np.ndarray
    measured: np.ndarray
    dead: float

    def index(self, solutions: np.ndarray) -> np.ndarray:
        return solutions @ _POWERS

    def evaluate(self, solutions: np.ndarray) -> np.ndarray:
        return self.fitness[self.index(solutions)]

    @property
    def optimum(self) -> float:
        return float(self.fitness.max())

    def top_threshold(self, k: int) -> float:
        """Fitness of the k-th best measured sequence."""
        return float(np.sort(self.fitness[self.measured])[-k])

    def table(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured sequences (n, 9) and their fitness."""
        indices = np.flatnonzero(self.measured)
        solutions = (indices[:, None] // _POWERS) % 4
        return solutions, self.fitness[indices]


def load_landscape(path: Path = DATA_PATH) -> Landscape:
    data = pd.read_csv(path)
    solutions = encode(data["sequence"])
    fitness = np.full(4**POSITIONS, np.nan)
    fitness[solutions @ _POWERS] = data["fitness"].to_numpy()
    measured = ~np.isnan(fitness)
    dead = float(np.nanmin(fitness))
    return Landscape(np.where(measured, fitness, dead), measured, dead)


def pair_epistasis(landscape: Landscape, alive_threshold: float) -> pd.DataFrame:
    """How much each pair of positions interacts, from the full measured table.

    Among the sequences above ``alive_threshold``, fit the additive model (one effect per
    base per position) and keep the residuals. For a pair (i, j), the residual mean of
    each of the 16 base combinations is what the pair adds beyond the two positions alone;
    its share of the residual variance is the pair's epistasis.
    """
    solutions, fitness = landscape.table()
    alive = fitness > alive_threshold
    solutions, fitness = solutions[alive], fitness[alive]

    residual = fitness - fitness.mean()
    effects = np.zeros((POSITIONS, 4))
    for _ in range(20):  # backfitting converges in a few sweeps
        for p in range(POSITIONS):
            residual += effects[p][solutions[:, p]]
            counts = np.bincount(solutions[:, p], minlength=4)
            effects[p] = np.bincount(solutions[:, p], residual, 4) / np.maximum(counts, 1)
            residual -= effects[p][solutions[:, p]]

    total = residual.var()
    matrix = np.zeros((POSITIONS, POSITIONS))
    for i, j in combinations(range(POSITIONS), 2):
        cell = solutions[:, i] * 4 + solutions[:, j]
        means = np.bincount(cell, residual, 16) / np.maximum(np.bincount(cell, minlength=16), 1)
        matrix[i, j] = matrix[j, i] = float((means[cell] ** 2).mean() / total)
    labels = [f"p{p + 1}" for p in range(POSITIONS)]
    return pd.DataFrame(matrix, index=labels, columns=labels)
