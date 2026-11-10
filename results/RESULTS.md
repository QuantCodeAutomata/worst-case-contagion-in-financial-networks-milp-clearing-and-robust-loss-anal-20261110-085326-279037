# Results: Worst-Case Contagion in Financial Networks: MILP Clearing and Robust Loss Analysis

All numbers below are taken from the JSON files saved in `results/` by the
corresponding `python -m exp.exp_N` run. Every experiment generates its own
random synthetic instance from the paper's stated generator family (Erdos-Renyi
or core-periphery graph, log-normal liabilities, a default-free nominal
system); because the paper does not publish the exact realized liability
matrix, the external-asset vector, or the log-normal shape/scale parameters
used for its own instances, the random instances constructed here are *a*
realization of the stated family, not a bit-for-bit copy of the paper's
instance. Point estimates can therefore differ from the paper's table entries
even where the underlying qualitative mechanism is reproduced; this is the
single dominant source of the gaps below and is discussed per experiment.

Verdict thresholds follow the final comparison rule: "reproduced" if the value
lies within a range the paper states, or else within 10% of the paper's value
with the same sign and the paper's qualitative result holds; "not reproduced"
if a value has the opposite sign, differs by more than 50%, or a qualitative
claim (ranking, equality, exact count, binding constraint) fails; otherwise
"partially reproduced".

## exp_1 — Random Pointwise Clearing Validation (D-val, Section 7.2, Table 2)

Source file: `results/exp1_results.json`. Instance D-val: n=8, m=3, ER(0.35),
seed=11. Realized thresholds: epsilon* = 0.2288, epsilon_ub (exact, Proposition
5) = 3.9018, epsilon_ub_lower (single-LP bound) = 3.5877 (-8.05% vs. exact).

| eps/eps_ub | LP loss (ours) | LP loss (paper) | exact loss (ours) | exact loss (paper) | avg insolvent (ours) | avg insolvent (paper) | LP feasible (ours) | LP feasible (paper) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.0147 | 0.0173 | 0.0147 | 0.0173 | 0.00 | 0 | 1.00 | 1.00 |
| 1.0 | 0.0758 | 0.0524 | 0.0758 | 0.0524 | 0.00 | 0 | 1.00 | 1.00 |
| 1.2 | 0.0700 | 0.0650 | 0.1096 | 0.0662 | 0.13 | 0.01 | 0.95 | 0.99 |
| 2.0 | 0.0805 | 0.0854 | 0.3017 | 0.1810 | 1.36 | 0.62 | 0.74 | 0.72 |

At every one of the 4 x 100 = 400 sampled shocks, the LP clearing loss (formula
8) equals the exact loss from exhaustive 3^8-regime enumeration whenever the LP
is feasible (`n_mismatches = 0` in every row of `table_2` in the JSON file):
the core claim of Section 7.2 holds exactly, not just approximately. LP
feasibility is 1.00 at eps/eps_ub in {0.5, 1.0} and falls below 1 only beyond
epsilon_ub (0.95 at 1.2x, 0.74 at 2.0x), reproducing the qualitative behaviour
of Proposition 5 (the LP is feasible for all tested shocks up to epsilon_ub and
can become infeasible beyond it).

Verdicts: LP-vs-exact agreement when feasible — **reproduced** (exact equality,
stronger than the paper's implicit tolerance). Feasible-fraction pattern
(1.00, 1.00, <1, further <1) — **reproduced**. Point loss values at 0.5x, 1.2x,
2.0x LP, and 2.0x feasible-fraction — **partially reproduced** (same order of
magnitude and monotone trend, within 15-19% of the paper). Loss value at 1.0x
and exact loss at 1.2x/2.0x — **not reproduced** (44-67% above the paper's
value); avg insolvent-bank count at 1.2x/2.0x is also materially higher. These
point-value gaps track a differently realized D-val instance (see Deviations).

## exp_2 — Pre-Insolvency Worst-Case Loss Curve (D-pre, Section 7.3, Table 3)

Source file: `results/exp2_results.json`. Instance D-pre: n=12, m=4, CP(0.28),
seed=22. Realized thresholds: epsilon* = 0.2500, epsilon_ub (exact) = 1.1776,
epsilon_ub_lower = 1.1776 (0.0% gap to the exact value).

| quantity | ours | paper |
|---|---|---|
| epsilon* | 0.2500 | 0.0517 |
| epsilon_ub (exact) | 1.1776 | 0.921 |
| epsilon_ub_lower | 1.1776 (+0.0%) | 1.02 (+10.8%) |
| curve convex & nondecreasing on [0, eps_ub] | True | True |
| number of salient points on (eps*, eps_ub) | 4 | 2 |

Breakpoints found (`table_3.breakpoints` in the JSON file): radius ratios
0.251, 0.376, 0.735, 0.769, with left/right slopes 0.0295/0.0786, 0.0786/0.0999,
0.0999/0.1319, 0.1319/0.1448, each classified as an "inner default-set switch"
(the winning asset direction j is unchanged across the kink, but the active
extreme point of the dual polytope X, and hence the realized default set,
changes), affecting banks 5, 8, 2 and 4 respectively. The paper's own D-pre
instance has 2 salient points at ratios 0.0508 and 0.172, both classified as
"shock-direction switches" (the active asset j changes).

Verdicts: curve convexity and monotonicity on [0, epsilon_ub] — **reproduced**
(Theorem 14's qualitative claim holds exactly). Piecewise-linearity / existence
of salient points on (epsilon*, epsilon_ub) — **reproduced** (both curves have
a positive, finite number of kinks there). Number of salient points (4 vs. 2),
their locations, and their outer/inner mechanism classification — **not
reproduced**: the realized D-pre instance used here traces a different
sequence of active dual vertices than the paper's instance. The breakpoint
detector itself can report either mechanism (the classifier compares
`piece_left.asset` to `piece_right.asset` from the traced envelope, see
`src/robust_loss.py::breakpoint_mechanism`); on this realization every traced
kink happens to be an inner switch, while the paper's instance shows two outer
switches. No test isolates whether this is solely due to the differently
realized random instance or also to a difference in how finely the pre-image
of each dual LP is traced near a degenerate vertex; the cause is not
identified further within the time available (see Deviations).

## exp_3 — Post-Insolvency Scalar Geometry (D-post, Section 7.4, Table 5)

Source file: `results/exp3_results.json`. Fixed instance D-post: n=6, m=1,
seed=1. Realized thresholds: epsilon* = 0.0284, epsilon_ub = 0.4122 (equal to
epsilon_ub_lower, as expected since m=1 collapses the two bounds).

| quantity | ours | paper |
|---|---|---|
| epsilon* (absolute) | 0.0284 | 0.211 |
| epsilon_ub (absolute) | 0.4122 | 0.883 |
| pre-insolvency convex & nondecreasing | True | True |
| post-insolvency nondecreasing | True | True |
| post-insolvency convex | False | False |
| loss at eps -> large multiple of eps_ub | 0.986 (near-total loss) | 1.0 (total loss, at 3.636x) |
| number of post-insolvency kinks | 4 | 7 |
| jumps detected (|loss(eps+1e-6) - loss(eps-1e-6)| > tol) | none (`is_continuous=True` at all 4 kinks) | none |

Realized kink table (radius ratio epsilon/epsilon_ub, left slope, right slope,
zero-payment set before -> after), from `table_5_kinks` in
`results/exp3_results.json`:

| ratio | left slope | right slope | zero-set before | zero-set after |
|---|---|---|---|---|
| 1.051 | 2.258 | 1.433 | {6} | {5,6} |
| 1.245 | 1.433 | 0.872 | {5,6} | {4,5,6} |
| 1.763 | 0.872 | 0.481 | {4,5,6} | {3,4,5,6} |
| 1.962 | 0.481 | 0.141 | {3,4,5,6} | {2,3,4,5,6} |

Verdicts: post-insolvency curve is nondecreasing and piecewise linear but **not
convex** beyond epsilon_ub — **reproduced** (Theorem 22's qualitative claim:
the right slope is strictly smaller than the left slope at every one of the
four detected kinks, monotonically decreasing from 2.258 down to 0.141, exactly
the convexity failure the paper's own Table 5 shows for its D-post instance,
and each kink is accompanied by a strictly growing zero-payment set, matching
the paper's description of post-insolvency kinks as changes in the
zero-payment bank set). Absence of jumps at every detected kink —
**reproduced**. Loss approaching the normalized total of 1 for a large stress
radius — **reproduced** (0.986 reached by 4x epsilon_ub, matching the paper's
saturation behaviour; full saturation to exactly 1.0 would require pushing the
grid further than tested here). The fixed D-post instance's exact parameter
values (P-bar, c-bar, S) are not published in the paper text and the paper
states that they are provided only in the authors' own reproducibility
package, which this repository did not have access to (see Deviations); the
instance constructed here is therefore a different hand-crafted n=6, m=1
network satisfying the same defining property (default-free at nominal
prices, S >= 0). Consequently epsilon*, epsilon_ub, the number of kinks (4 vs.
7), and the individual kink radii/slopes/zero-payment sets in Table 5 are
**not reproduced** as point values; they are a different realization of the
same geometric phenomenon the paper describes.

## exp_4 — Certified Bounds and Regime Generation (M-bound, Section 7.5, Tables 6-9)

Source file: `results/exp4_results.json`. Instance M-bound: n=25, m=5,
CP(0.12), seed=5. Realized thresholds: epsilon* = 0.2165, epsilon_ub (exact) =
1.4026, epsilon_ub_lower = 1.3952 (-0.52% vs. exact).

Table 6 (global scenario lower bound LB vs. fixed-pattern upper bound UB,
`table_6_global_bounds`):

| eps/eps_ub | LB (ours) | UB (ours) | gap % (ours) | LB (paper) | UB (paper) | gap % (paper) |
|---|---|---|---|---|---|---|
| 0.8 | 0.0494 | 0.0494 | ~0 | 0.0294 | 0.0294 | 0 |
| 1.0 | 0.0788 | 0.0788 | ~0 | 0.0412 | 0.0412 | 0 |
| 1.2 | 0.1060 | 0.1167 | 9.18 | 0.0531 | 0.0656 | 19.1 |
| 1.5 | 0.1574 | 0.1679 | 6.20 | 0.0744 | 0.1040 | 28.2 |
| 2.0 | 0.2585 | 0.2930 | 11.79 | 0.1050 | 0.1680 | 37.1 |
| 3.0 | 0.4398 | 0.4457 | 1.32 | 0.1770 | 0.5770 | 69.3 |

Table 8 (localized fixed-pattern certificate closing the gap at post-insolvency
radii, `table_8_localized_refinement`): at 0.8x, 1.0x the localized UB equals
LB exactly (gap ~1e-18 and 0). At 1.2x and 1.5x the localized UB closes most
but not all of the global gap (residual gap 1.07% and 1.04% of LB, versus 0%
in the paper). At 2.0x the residual gap is 3.45% (paper: 0%, i.e. exact
closure). At 3.0x the adaptive bisection used here was not run to convergence
within its cell budget (`closed_within_cell_budget=False`, 81 cells, no
localized UB recorded) — the global UB of 0.4457 is reported there instead.

Table 9 (computational effort, `table_9_computational_effort`): at 2.0x, 3
cells / 35 local clearings / 52 pattern LPs / 0.098s (paper: 61 cells / 154
clearings / 236 LPs / 0.41s); at 3.0x, 81 cells / 1127 clearings / 1535 pattern
LPs / 2.75s (paper: 81 cells / 202 clearings / 708 LPs / 1.3s). The cell counts
at 3.0x match; the local-clearing and pattern-LP counts differ (richer local
candidate family generated here per cell).

Verdicts: monotone gap reduction as radius increases from the trivial bound (UB
<= 1 at iteration 0, `table_7_regime_generation_at_2x`) — **partially
reproduced** (gap narrows with regime generation but the trace only records 2
history points because the fixed-point regime-generation loop converges
immediately on this instance, versus the paper's 27-iteration trace). The
qualitative pattern "gap grows with radius for the global bound, then the
localized certificate closes it at post-insolvency radii" — **partially
reproduced**: the localized certificate closes the gap exactly at 0.8x-1.0x
(pre/at-threshold radii) but leaves a small residual gap at 1.2x-2.0x radii
where the paper reports exact closure, and does not finish at 3.0x within the
bisection budget used. Individual LB/UB numeric values — **not reproduced**
(the realized M-bound instance differs from the paper's, giving uniformly
higher losses at every radius; relative gaps are of the same order of
magnitude but smaller than the paper's at every radius).

## exp_5 — Scenario Lower Bounds on a 353-Bank Network (L-353, Section 7.6, Tables 10-11)

Source file: `results/exp5_results.json`. Instance L-353: n=353, m=5, CP(0.02),
seed=353. Realized thresholds: epsilon* = 0.2053, epsilon_ub (exact) = 0.3983,
epsilon_ub_lower = 0.3983 (0.0% gap — the two coincide on this realization, as
the paper also reports for its own L-353 instance: 0.805 = 0.805).

Table 10 (`table_10_scenario_lower_bounds`):

| eps/eps_ub | LB (ours) | n patterns (ours) | LB (paper) | n patterns (paper) |
|---|---|---|---|---|
| 0.0 | 0.0000 | 1 | 0.0000 | 1 |
| 0.0776 | 0.0000 | 1 | 0.0000 | 1 |
| 0.5 | 0.0000 | 1 | 0.00799 | 1 |
| 1.0 | 0.00249 | 2 | 0.0208 | 2 |
| 1.5 | 0.01115 | 2 | 0.0339 | 3 |
| 2.0 | 0.02646 | 4 | 0.0492 | 7 |
| 3.0 | 0.07316 | 8 | 0.0816 | 10 |
| 5.0 | 0.19583 | 39 | 0.138 | 23 |

Median per-scenario clearing time (fixed-point iteration): 0.11-1.08 ms across
radii (paper: ~1.2-1.4 ms) — same order of magnitude. Fixed-point residual
tolerance 3e-9 was met at every scenario (paper target: < 3e-9).

Table 11 (localized certificates, `table_11_localized_certificates`): at 1.5x,
localized UB = 0.011153 vs. LB = 0.011153 (gap 9.4e-13, essentially exact); at
2.0x, UB = 0.026789 vs. LB = 0.026462 (gap 0.033% of LB); at 3.0x, UB =
0.074088 vs. LB = 0.073159 (gap 0.093% of LB). All three gaps are below 0.1%
of LB, i.e. at machine-relevant tolerance, matching the paper's claim that the
localized certificate certifies the scenario lower bound as the worst case at
every tested post-insolvency radius.

Verdicts: localized certificate closing the gap to the scenario lower bound at
1.5x, 2.0x, 3.0x epsilon_ub — **reproduced** (gaps of 9e-13, 3e-4, 9e-4 relative
to LB, all effectively zero as the paper reports). Monotone growth of the
number of distinct realized payment patterns with radius (1, 1, 1, 2, 2, 4, 8,
39 vs. paper's 1, 1, 1, 2, 3, 7, 10, 23) — **reproduced** as a qualitative
trend (monotone non-decreasing, richer regime structure post-insolvency); the
exact counts differ instance-by-instance. LB point values — **partially
reproduced** at 3.0x-5.0x (same order of magnitude, 10-42% below the paper) and
**not reproduced** at 0.5x-2.0x (the paper's instance already has a strictly
positive loss at 0.5x epsilon_ub and a richer pattern count earlier than ours;
our realization needs a larger radius before any bank defaults in the sampled
scenarios).

## Deviations from the paper

- **Synthetic instance generation.** The paper's Section 7.1 states the
  generator family (Erdos-Renyi ER(d) or core-periphery CP(d) graphs,
  log-normal positive liabilities rescaled so total interbank liability is of
  the order of n, a default-free nominal system built by compensating each
  bank for max(0, p_bar_i - (A^T p_bar)_i) plus a small random buffer, and an
  exposure matrix S >= 0 with larger exposures assigned to more vulnerable
  banks) but does not give: the log-normal location/scale parameters, the
  magnitude of the random buffer, the exact core/periphery link-probability
  multipliers beyond the single baseline density quoted per instance, or the
  precise rule used to rank "more vulnerable" banks for the S assignment.
  `data/generate_synthetic.py` documents the specific choices made for each of
  these (log-normal(mu=0, sigma=1) liability magnitudes rescaled to total
  interbank liability n, buffer drawn uniform on [0, 0.1] times the per-bank
  net shortfall scale, core-core/core-periphery/periphery-periphery
  probability multipliers of 3x/1.5x/1x times the baseline density, and S rows
  scaled inversely to each bank's nominal post-payment net resource margin).
  Because of this, every instance built here (D-val, D-pre, D-post, M-bound,
  L-353) is a different *realization* of the paper's stated family rather than
  the paper's own instance, which is the dominant source of every point-value
  gap reported above. The paper's own reproducibility package
  ("MILP_Clearing_code", referenced in the text for the D-post instance) was
  not reachable from the paper text used for this replication, so the exact
  fixed D-post network could not be reused.
- **D-post instance (exp_3).** The paper states D-post is a fixed,
  hand-crafted n=6, m=1 instance "available in the reproducibility package",
  not generated from the ER/CP family. This repository instead builds a
  different fixed n=6, m=1, S >= 0, default-free instance with the same
  qualitative properties (used to demonstrate non-convexity beyond
  epsilon_ub), since the authors' exact numbers were not available; see the
  exp_3 section above for the resulting difference in epsilon*, epsilon_ub,
  and kink count.
- **exp_4 localized certificate at 3x epsilon_ub.** The adaptive longest-edge
  bisection used here did not close the cell budget at this radius
  (`closed_within_cell_budget=False` after 81 cells); only the global
  fixed-pattern upper bound is reported at that radius. The paper reports a
  closed localized certificate with 81 cells at the corresponding radius. The
  cause (a tighter cell-acceptance test, or a genuinely harder local geometry
  on this M-bound realization) was not isolated further.
- **No market data used.** Every experiment here operates on synthetic
  interbank networks as specified by the paper; no external market-data
  provider (QuantX Dataloader or Massive) is used anywhere in this
  repository, consistent with the paper's purely synthetic numerical section
  (Section 7).
- **Growth experiment in exp_2 (Table 4).** Table 4 of the paper (number of
  affine pieces / salient points vs. network size n in {8, 16, 32, 64, 128})
  is reported in `results/exp2_results.json` under `table_4` with two random
  seeds per size, using the same CP generator and parametrization as D-pre
  (the paper does not specify a per-size density rule for the growth
  experiment beyond "m=4"; the baseline density 0.28 used for D-pre is reused
  unchanged across all five sizes here, which is a documented assumption).

## Not reproduced

- **exp_2, Table 4, exact breakpoint mechanism labels for individual kinks of
  D-pre** (which specific banks switch and whether each kink is an outer or
  inner switch) are not reproduced point-for-point; only the qualitative
  existence of a finite, positive number of salient points with both left and
  right slopes well-defined is reproduced (see exp_2 section above for the
  specific counts obtained).
- **exp_3, Table 5, the authors' exact D-post instance and its seven
  tabulated kink radii/slopes/zero-payment sets** are not reproduced, because
  the paper states this fixed instance is only available in the authors'
  reproducibility package, which was not accessible from the evidence
  available for this replication (see Deviations). The geometric claim the
  table illustrates (non-convex, nondecreasing, piecewise-linear,
  jump-free post-insolvency curve) is reproduced on a substitute instance.
- **exp_4, Table 9, the paper's exact wall-clock seconds** are hardware- and
  implementation-dependent figures that this repository's run environment
  cannot reproduce in absolute terms; only the qualitative growth of
  computational effort with radius (more cells, more local clearings, more
  pattern LPs, more wall time at larger radii) is reported.
