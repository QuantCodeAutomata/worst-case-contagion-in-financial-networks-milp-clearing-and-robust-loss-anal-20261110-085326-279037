# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of the numerical experiments (Section 7) of "Worst-Case
Contagion in Financial Networks: MILP Clearing and Robust Loss Analysis":
clearing-vector computation (LP, MILP, exact regime enumeration), the
insolvency-resilience threshold, the pre-insolvency robust worst-case loss
curve, post-insolvency scalar geometry, and certified lower/upper bounds for
post-insolvency worst-case losses (scenario bounds, fixed-pattern dual bounds,
regime generation, localized certificates). All experiments use synthetic
financial-network instances; there is no market-data dependency.

## Layout

- `src/` — flat library modules: `config.py` (all parameters, with a source
  comment per value), `clearing.py`, `robust_loss.py`, `certificates.py`.
- `exp/` — one script per experiment, `exp_1.py` .. `exp_5.py`, run as
  `python -m exp.<name>`. Each writes its outputs to `results/`.
- `data/` — `generate_synthetic.py` builds the paper's synthetic network
  instances (Erdos-Renyi and core-periphery liability graphs). No market data
  is loaded anywhere in this repository.
- `tests/` — pytest tests for the `src/` components.
- `results/` — JSON metrics and plots produced by the experiment scripts, plus
  `RESULTS.md` with the paper comparison, deviations, and verdicts.

## Commands

Setup:

```bash
pip install -r requirements.txt
```

Tests (all synthetic / in-memory, no market-data marker is used in this
repository since no experiment touches market data):

```bash
python -m pytest
```

Experiments (paper's full configuration):

```bash
python -m exp.exp_1
python -m exp.exp_2 --full
python -m exp.exp_3
python -m exp.exp_4 --full
python -m exp.exp_5 --full
```

Or everything at once:

```bash
bash reproduce.sh
```

## Conventions

- All experiment parameters live only in `src/config.py`, as module-level
  UPPER_CASE constants, each with a one-line comment stating its meaning,
  unit, and the paper section/table it comes from, or "not specified in the
  paper" when it is a documented assumption.
- Instance generation lives only in `data/generate_synthetic.py`; `src/` and
  `exp/` never generate networks directly.
- Deviations from the paper (generator assumptions, reduced configurations,
  gaps to reported numbers) are recorded only in `results/RESULTS.md`, never
  in code comments or commit messages.
- Running an experiment script overwrites its files in `results/`. After a
  rerun, the numbers and verdicts in `results/RESULTS.md` must be checked
  against the new files and updated by hand.
- No API keys are used by this repository (no market data); if any are ever
  introduced they must only be read from an uncommitted `.env`.
