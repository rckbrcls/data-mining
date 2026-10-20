# Deceptive traps — AED built by DAMICORE

**Hypothesis.** On concatenated deceptive traps whose bits are scattered at random,
DAMICORE (NCD → Neighbor Joining → FastGreedy), applied to the promising strings of each
generation, recovers the true traps, and the AED built from its groups solves problems of
20 to 80 bits.

A trap of k = 5 bits with u ones is worth 5 when u = 5 and 4 − u otherwise. Every bit alone
points to zero while the optimum is all ones, so a search that treats bits one at a time is
misled. Only a model that finds which bits belong together and samples them as a unit finds
the optimum reliably. The true traps are known, so the evidence is direct: how close
DAMICORE's groups are to them (adjusted Rand index, ARI). No other method is run.

| Path | Purpose |
| --- | --- |
| `notebooks/01_deceptive_trap_linkage.ipynb` | Didactic walkthrough (Portuguese); loads the latest run. |
| `scripts/config.py` | Problem sizes, population sizes, seeds, selection, stopping. |
| `scripts/trap.py` | Trap instances with scattered bits; ARI, pair precision/recall, exact traps. |
| `scripts/search.py` | The DAMICORE AED: variable series, groups, pattern tables, sampling. |
| `scripts/experiment.py` | Runs the grid in parallel and writes the artifacts. |
| `scripts/build_notebooks.py` | Rebuilds the notebook. |
| `tests/` | Trap values, linkage scores, encoding, planted groups, a small solved run. |
| `artifacts/` | Generated runs; ignored by Git. |

## The AED

Each generation:

1. The best 30% of the population are the promising strings.
2. One file per variable lists the ids of the promising strings with a 0, then those with a
   1 (3 printable bytes per id). Two bits of the same trap split the strings the same way and
   write similar runs, which NCD can see; writing the bits themselves hides that.
3. DAMICORE groups the variables. Constant variables stay alone; identical files are exact
   copies and go together. If FastGreedy returns one community over every file, it is not
   used (a few unrelated leaves cannot be split).
4. Each group gets a table of the patterns seen among the promising strings.
5. Each offspring draws every group whole from its table. Parents and offspring compete and
   the best N survive.

There is no uniform mixing and there are no fixed blocks: DAMICORE is the only source of
structure.

**Grid.** m ∈ {4, 6, 8, 10, 12, 16} traps (20–80 bits) × N ∈ {1k, 2k, 4k, 8k, 16k, 32k} ×
10 seeds = 360 runs. Each run stops at the optimum, after 10 generations without
improvement, when the population converges, or after 60 generations. A population size
*solves* a problem size when at least 9 of the 10 seeds reach the optimum.

```bash
PYTHONPATH=. venv/bin/python -c "from hypotheses.deceptive_trap_linkage.scripts.experiment import execute_experiment; print(execute_experiment().output_dir)"
venv/bin/python -m unittest hypotheses.deceptive_trap_linkage.tests.test_deceptive_trap_linkage
```

## Results (run `20261020T212703Z`, 341 s on 7 workers)

All six problem sizes were solved; 244 of the 360 runs reached the optimum.

| Traps | Bits | Smallest N that solves | Success | Generations | Evaluations | Best ARI | Traps found exactly |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 20 | 1,000 | 10/10 | 4.5 | 5,500 | 0.95 | 82% |
| 6 | 30 | 2,000 | 10/10 | 5.4 | 12,800 | 0.99 | 97% |
| 8 | 40 | 4,000 | 10/10 | 7.1 | 32,400 | 1.00 | 100% |
| 10 | 50 | 8,000 | 10/10 | 7.3 | 66,400 | 1.00 | 100% |
| 12 | 60 | 8,000 | 9/10 | 10.3 | 84,444 | 0.99 | 98% |
| 16 | 80 | 16,000 | 10/10 | 11.1 | 193,600 | 1.00 | 100% |

**DAMICORE finds the traps.** In the first generation the promising strings are almost
random and the ARI is near zero (0.05–0.16). It then climbs as selection concentrates the
traps. Mean ARI per generation at the smallest solving N:

| Traps | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 | Gen 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.13 | 0.40 | 0.78 | 0.95 | 0.99 | 1.00 |
| 10 | 0.10 | 0.35 | 0.84 | 0.98 | 1.00 | 1.00 |
| 16 | 0.05 | 0.15 | 0.43 | 0.74 | 0.92 | 0.99 |

Across all solved configurations, the best ARI averages 0.99.

**Failure is a population that is too small, and it is linkage that fails.** Below the
needed N, the runs converge to a wrong string, and the best ARI drops with the share of
traps solved. For 16 traps it goes from 0.06 (N=1k) to 0.29 (4k), 0.80 (8k), and 1.00
(16k). When DAMICORE does not recover the traps, the AED does not solve them.

**Large samples make DAMICORE see the traps at once.** With N=32k (9,600 promising
strings), the ARI is already 0.90–1.00 in generation 1 for 6–16 traps, before selection
has done much.

**Cost.** The evaluations to the optimum grow roughly as bits^2.6 (log-log slope over the
six solved sizes). The coarse population grid (doubling) makes this exponent approximate.

DAMICORE runs: 2,751. Of these, 117 had fewer than 3 distinct series (nearly converged
populations) and 12 returned a single community, which was not used.

## Encoding finding

The ids of the promising strings were first written with 2 bytes. With 3 bytes, recovery
improved: 236 → 244 solved runs, and 16 traps were solved with N=16k instead of 32k. With
7,200 promising strings in generation 1, the ARI went from 0.12 (2 bytes) to 0.95
(3 bytes). The likely cause is that zlib copies only matches of 3 bytes or more, so a 2-byte
id shared by two files cannot be recognised on its own.

## Limits

- No other method is compared; the claim is structure recovery against a known truth, not
  that DAMICORE beats ECGA, BOA, or LTGA. The numbers of evaluations are not directly
  comparable with the literature (different selection, replacement, and population grid).
- Only non-overlapping 5-bit traps. Overlapping or hierarchical dependencies were not
  tested.
- Each DAMICORE call compares every pair of variables with zlib. Its cost grows with the
  square of the number of variables and with the population size.
- The selection fraction (30%), plus replacement, and the stopping rules came from a pilot
  on 10 traps and were not tuned further.
