# DHFR codon linkage

AED from the "DAMICORE gerando modelos para AEDs" slide, run on a real, fully measured
fitness landscape: 9 nucleotides of the *E. coli* *folA* gene (DHFR codons 26–28),
all 4⁹ = 262,144 combinations, growth under trimethoprim (Papkou et al. 2023).

## Questions

1. **Does it optimize?** With 1,000 measured sequences per run, does it reach the top 70?
2. **Does DAMICORE find meaningful structure?** Without being told, does it group the
   positions of the same codon (1-2-3, 4-5-6, 7-8-9) and the pairs that really interact?

There are no baseline comparisons: the evidence is the structure DAMICORE recovers.

## Method

Each generation, the top 30% of a population of 200 are the promising sequences. One
series per position lists their ids grouped by base (A, C, G, T; 3-byte ids so zlib can
match them). DAMICORE (NCD → Neighbor Joining → FastGreedy) groups the positions; each
group gets a joint table of its observed base patterns, offspring draw whole groups from
it, and parents and offspring compete (μ+λ). Only distinct sequences count against the
budget; if a generation measures nothing new, half the population restarts at random.

## Results (30 seeds, run `20261021T010809Z`)

| | |
|---|---|
| Runs reaching the top 70 (0.03% best) | 100% |
| Runs finding the global optimum `AAGGAAATG` | 27% |
| Mean best fitness / optimum | 97.6% |
| Median sequences measured until the top 70 | 766 |
| Most grouped pairs | 1-2 (52%), 7-8 (48%), 6-7, 4-5 (31%) |
| Codon split among the 280 splits into triples | 3rd, exact p = 0.011 |
| Best split | 1-2-3 \| 4-5-9 \| 6-7-8 |
| Spearman, co-grouping vs epistasis (36 pairs) | ρ = 0.22, p = 0.19 |

7-8 is by far the strongest epistatic pair in the full table and one of the two pairs
DAMICORE groups most; 6-8, the second strongest, is also high. The overall correlation
with epistasis is weak: DAMICORE also groups some pairs with no interaction.

## Layout

- `data/papkou2023_dhfr_fitness.csv.gz`: sequence, fitness, fitness_se (CC BY 4.0).
- `scripts/landscape.py`: loading, encoding, pair epistasis from the full table.
- `scripts/search.py`: position series, DAMICORE groups, sampling, the AED.
- `scripts/experiment.py`: all seeds, pair co-grouping, codon test, audit.
- `scripts/build_notebooks.py`: builds `notebooks/01_dhfr_codon_linkage.ipynb` (Portuguese).
- `artifacts/<run_id>/` (ignored by git): `audit.json`, `runs.csv`, `generations.csv`,
  `pair_frequency.csv`, `epistasis.csv`, `pairs.csv`, and `example/` with the NCD matrix,
  tree, and groups of every generation of seed 1.

## Running

```bash
venv/bin/python -m unittest hypotheses.dhfr_codon_linkage.tests.test_dhfr_codon_linkage
venv/bin/python -c "from hypotheses.dhfr_codon_linkage.scripts.experiment import execute_experiment; execute_experiment()"
venv/bin/python -m hypotheses.dhfr_codon_linkage.scripts.build_notebooks
```
