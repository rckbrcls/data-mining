# Emergency feature selection — DAMICORE-guided multiobjective AED

**Hypothesis.** An estimation-of-distribution algorithm (AED) whose probabilistic model is
built by DAMICORE finds the Disque 100 forms with the best trade-off between cost (how many
of the 15 report fields are asked) and quality (how much information they carry about the
emergency label).

The method follows the "DAMICORE gerando modelos para AEDs" slide in
`docs/AED Proposito Geral.pdf`, with the extruder replaced by the form. Every generation:

1. **Selection.** The best 30% of the forms evaluated so far, in Pareto order (quality, cost).
2. **Variables as samples.** Each field becomes a file with its 0/1 inclusion across those
   forms, all in the same order.
3. **DAMICORE.** NCD matrix → Neighbor-Joining tree → FastGreedy groups of fields.
4. **Probabilistic model.** Each group gets a Laplace-smoothed joint probability table.
5. **Sampling.** New forms are drawn from the tables, evaluated, and the loop repeats.

| Path | Purpose |
| --- | --- |
| `notebooks/01_emergency_feature_selection.ipynb` | End-to-end didactic pipeline. |
| `scripts/config.py` | Fields, target, data profiles, and the AED budget. |
| `scripts/data.py` | PostgreSQL report aggregation, status audit, and temporal folds. |
| `scripts/fitness.py` | Quality of a form and the exhaustive ground truth. |
| `scripts/pareto.py` | Non-dominated ranks, crowding, Pareto front. |
| `scripts/search.py` | The DAMICORE-guided AED. |
| `scripts/experiment.py` | Runs the AED seeds, the ground-truth check, and writes artifacts. |
| `scripts/build_notebooks.py` | Rebuilds the notebook. |
| `tests/` | Synthetic checks (no database). |
| `artifacts/` | Generated run outputs and the ground-truth cache; ignored by Git. |

**Quality.** The selected fields split reports into cells (value combinations). Training
gives each cell an emergency rate shrunk toward the base rate (pseudo-count 50); quality is
the log-loss reduction on the following validation period in bits per report, averaged over
three expanding-window folds. Overly sparse combinations lose quality.

**Budget.** Each run evaluates 1,200 forms (400 initial + 10 generations × 80), scored
directly on the data; 10 seeds run in parallel.

**Recommendation.** The fewest fields on the AED front reaching 95% of its best quality,
also measured on the reserved period next to the all-fields form.

**Ground-truth check.** All 32,767 subsets are scored once (cached) only to check whether
the AED reached the same recommendation; the AED never reads this table.

**Data profiles.** `EMERGENCY_DATA_PROFILE=full` (default) uses 2024-01..2026-06 from
`DISQUE100_DATABASE_URL` with folds 2025-Q3/Q4/2026-Q1 and reserved period 2026-Q2.
`local_2026` uses the local PostgreSQL copy of 2026-S1 with monthly folds and test 2026-06.

**Limits.** The label reproduces the historical emergency status, not verified urgency. NCD
on binary series recognizes fields that enter good forms together, not substitutes.

Run the tests with:

```bash
venv/bin/python -m unittest hypotheses.emergency_feature_selection.tests.test_emergency_feature_selection -v
```
