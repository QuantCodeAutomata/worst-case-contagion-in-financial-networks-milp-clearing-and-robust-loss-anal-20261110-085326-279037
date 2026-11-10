# Results: Worst-Case Contagion in Financial Networks: MILP Clearing and Robust Loss Analysis

Status: in progress. This file is updated as experiments are run; numbers
below are taken from the JSON files saved by the corresponding `exp.exp_N`
run in `results/`.

## exp_1 — Random Pointwise Clearing Validation (D-val, Table 2)

Source file: `results/exp1_results.json`.

| epsilon/epsilon_ub | avg LP loss | avg exact loss | avg insolvent | LP feasible | paper avg LP | paper avg exact | paper LP feasible |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.0172 | 0.0172 | 0.00 | 1.00 | 0.0173 | 0.0173 | 1.00 |
| 1.0 | 0.0784 | 0.0784 | 0.00 | 1.00 | 0.0524 | 0.0524 | 1.00 |
| 1.2 | 0.0716 | 0.1108 | 0.13 | 0.95 | 0.0650 | 0.0662 | 0.99 |
| 2.0 | 0.0731 | 0.3020 | 1.39 | 0.73 | 0.0854 | 0.1810 | 0.72 |

Verdicts: under review (see below).

## exp_2, exp_3, exp_4, exp_5

Not yet compared; the remaining experiment scripts are still being completed
and validated.

## Deviations from the paper

The exact parameters of the synthetic-network generator that the paper does
not specify (log-normal liability location/scale, the magnitude of the
no-default buffer, the core/periphery link-probability multipliers) are
documented assumptions in `data/generate_synthetic.py` and `src/config.py`.
This section will be completed once every experiment has been run and
compared against the paper.

## Not reproduced

To be completed.
