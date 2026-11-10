# Worst-Case Contagion in Financial Networks: MILP Clearing and Robust Loss Analysis

Reproduction of selected numerical experiments from "Worst-Case Contagion in
Financial Networks: MILP Clearing and Robust Loss Analysis" (paper indexed for
this replication; see `results/RESULTS.md` for the exact page references used
for every implementation detail).

This repository reproduces the five numerical experiments of the paper's
Section 7 ("Numerical experiments"):

1. `exp_1` — Random Pointwise Clearing Validation and the Role of the
   Insolvency-Resilience Threshold (D-val instance, Table 2).
2. `exp_2` — Pre-Insolvency Worst-Case Loss Curve: Piecewise Linearity and
   Breakpoint Diagnostics (D-pre instance, Table 3; growth experiment, Table 4).
3. `exp_3` — Post-Insolvency Scalar Geometry: Exact Regime Enumeration on a
   Single-Asset Instance (D-post instance, Table 5).
4. `exp_4` — Certified Bounds and Regime Generation for Post-Insolvency
   Worst-Case Losses (M-bound instance, Tables 6-9).
5. `exp_5` — Scenario Lower Bounds and Localized Certificates on a 353-Bank
   Network (L-353 instance, Tables 10-11).

All instances are synthetic financial networks generated as described in the
paper's Section 7.1; no external market data is used anywhere in this
replication.

## Repository layout

```
src/    clearing (LP/MILP/exact regime enumeration), robust pre-insolvency
        loss curve, post-insolvency certificates, shared configuration
exp/    one script per experiment, exp_1 .. exp_5
data/   data/generate_synthetic.py builds the synthetic network instances
tests/  pytest unit tests for src/ components
results/ saved metrics (JSON) and plots (PNG) for every experiment
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Running the experiments

Each experiment is run as a module from the repository root and writes its
results directly into `results/`:

```bash
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
```

`exp_2` and `exp_4`/`exp_5` accept a `--full` flag to run the paper's full
configuration where a reduced configuration was used to fit the available
compute time (see `results/RESULTS.md`, "Deviations from the paper", for the
exact counts and the estimated runtime of the full configuration).

```bash
python -m exp.exp_2 --full
```

Or run everything, including the tests, with:

```bash
bash reproduce.sh
```

## Results

See `results/RESULTS.md` for the comparison of every reproduced metric against
the paper's reported values, with a verdict (reproduced / partially
reproduced / not reproduced) for each, plus the "Deviations from the paper"
and "Not reproduced" sections.
