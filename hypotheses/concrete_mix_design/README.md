# Concrete mix design — AED built by DAMICORE

**Hypothesis.** An estimation-of-distribution algorithm (AED) whose whole probabilistic
model is built by DAMICORE — which mixes form each region of the front, which components
go together, and one table per group — approaches the true strength × cost Pareto front
while evaluating 0.06% of the possible mixes.

This is the extruder example of `docs/AED Proposito Geral.pdf` with concrete instead of the
extruder: 7 components (kg/m³) are the variables, the 28-day compressive strength is the
quality, and the material cost is the cost.

| Path | Purpose |
| --- | --- |
| `notebooks/01_concrete_mix_design.ipynb` | End-to-end didactic pipeline (Portuguese). |
| `data/concrete.csv` | UCI Concrete Compressive Strength (Yeh 1998), converted from the original `.xls`. |
| `scripts/config.py` | Components, illustrative prices, search space, and the AED budget. |
| `scripts/landscape.py` | Levels, strength surrogate, feasibility, and the scored grid (cached). |
| `scripts/pareto.py` | Constrained Pareto order, fast front, 2-D hypervolume. |
| `scripts/search.py` | The DAMICORE AED: niches, variable groups, tables, sampling. |
| `scripts/experiment.py` | Runs every seed, compares it with the ground truth, writes artifacts. |
| `scripts/build_notebooks.py` | Rebuilds the notebook. |
| `tests/` | Synthetic checks (no surrogate, small grid). |
| `artifacts/` | Generated runs and the landscape cache; ignored by Git. |

**Search space.** Each component takes one of 8 levels (quantiles of the 428 real mixes;
slag, fly ash, and superplasticizer also have 0), so there are 8⁷ = 2,097,152 mixes.

**Objectives.** Strength is predicted at 28 days by a monotone-constrained gradient-boosting
model trained on the 1,030 laboratory tests (grouped 5-fold CV: R² 0.877, MAE 4.0 MPa).
Cost uses illustrative US$/kg prices in `config.py`.

**Feasibility.** A mix is feasible only if its standardized distance to the nearest real mix
is ≤ 1.0; 363,486 of the 2.1M mixes qualify. Infeasible mixes cost an evaluation and rank behind every feasible one
(Deb's rule).

**The AED.** 20 seeds; each starts from 200 random mixes and runs 20 generations of 50
offspring (1,200 evaluations, 0.06% of the space). Each generation:

1. Promising mixes: the best 10% (at most 100) of everything evaluated, in constrained
   Pareto order.
2. Niches: DAMICORE over one file per mix (each component's letter repeated by its level,
   `A.BBB.C…`). Its FastGreedy cut is kept if it gives at most 6 niches of at least 10 mixes;
   otherwise DAMICORE is asked for fewer clusters until it does.
3. Variable groups: DAMICORE over one file per component, once over all promising mixes and
   again inside each niche. Each file lists the mix ids grouped by level, so two components
   that split the mixes the same way write the same runs whatever the level labels (also
   "much cement, little water"). The previous encoding (one level character per mix) hides
   those relations: on planted dependencies DAMICORE recovered them in 20/20 cases with the
   new encoding and 1/20 with the old one.
4. One joint table per group in each niche; each offspring picks a niche at random and draws
   every group from that niche's table.

There is no uniform mixing and no fixed block structure. If a niche only reproduces mixes
already evaluated, the draw moves to the other niches and then to the model of all
promising mixes; a uniform draw is the last resort and is logged (it never happened). If
DAMICORE cannot build a tree, the last DAMICORE groups are reused (3 of 1,945 runs).

**Ground truth.** All 2.1M mixes are scored once and cached; the searches only read what they
evaluate. The full table gives the true Pareto front, the hypervolume reference, and the
cheapest mix for 30/40/50/60 MPa. No other search method is run.

## Results (run `20261020T204734Z`)

| Metric (20 seeds) | Mean ± std | Range |
| --- | --- | --- |
| Hypervolume / truth | 0.845 ± 0.054 | 0.745 – 0.926 |
| Extra cost at 30 MPa (US$/m³) | 5.81 ± 4.75 | 0.36 – 12.75 |
| Extra cost at 40 MPa | 7.42 ± 1.90 | 2.36 – 11.86 |
| Extra cost at 50 MPa | 3.13 ± 1.68 | 0.24 – 6.64 |
| Extra cost at 60 MPa | 5.29 ± 4.97 | 1.26 – 20.14 |

Pooling the seeds, the cheapest mixes found cost US$0.24–2.36/m³ more than the true cheapest
for 30–60 MPa. Offspring came from the chosen niche (72.7%), another niche (25.3%), or all
promising mixes (1.9%); none were uniform.

What DAMICORE found:

- Niches split the front: on average they leave apart 47% of the promising mixes' strength
  range (0 = every niche spans it all). 90% of generations used 2–6 niches.
- Variable groups are now above chance (23.5% for any pair): cement + superplasticizer
  40.0%, water + coarse aggregate 35.8%, fly ash + superplasticizer 34.3% (binomial
  Bonferroni p < 0.001; generations of a seed are not independent, so this is a screen).
- Cement and water share a group in only 10.3% of generations: DAMICORE does not recover the
  water–cement relation here.

For reference only, the previous run (`20261009T045430Z`, old encoding, same seeds and
initial populations) gave 0.830 for its DAMICORE AED. Per seed the new AED is +0.014 above it
(12/20 seeds, Wilcoxon p = 0.23), level with that run's shuffled-block control (p = 0.65),
above its UMDA (+0.036, p = 0.012), and above its random search (+0.074, p < 0.001).

**Reading.** The AED built only by DAMICORE reaches 84% of the true hypervolume and lands
within a few dollars of the cheapest mix for each strength, and DAMICORE now finds structure
(niches and above-chance groups) that it did not find before. Without a control in this
run, it cannot show that the gain comes from DAMICORE's choices rather than from any AED.

**Limits.** Strength is a surrogate prediction; nearest real mixes show the model can be
optimistic. Prices are illustrative. The niche limits (6 niches, 10 mixes) were chosen in a
pilot on 7 of the 20 seeds. The comparison with the earlier run is historical, not part of
this experiment.

Run the tests with:

```bash
venv/bin/python -m unittest hypotheses.concrete_mix_design.tests.test_concrete_mix_design -v
```

Data: I-Cheng Yeh, "Concrete Compressive Strength", UCI Machine Learning Repository (CC BY 4.0),
https://doi.org/10.24432/C5PK67.
