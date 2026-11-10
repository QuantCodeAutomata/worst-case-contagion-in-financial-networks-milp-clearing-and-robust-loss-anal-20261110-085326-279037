"""Model and experiment parameters, kept as module-level constants.

Every value carries the paper section or table that specifies it, or is
marked "not specified in the paper" when the generator choice is a documented
assumption (see results/RESULTS.md, "Deviations from the paper").
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Common synthetic-instance generator parameters (Sec. 7.1, Table 1)
# ---------------------------------------------------------------------------
LOGNORMAL_MU = 0.0          # location of the underlying normal law for edge liabilities; not specified in the paper
LOGNORMAL_SIGMA = 1.0       # scale of the underlying normal law for edge liabilities; not specified in the paper
BUFFER_SCALE = 0.05         # relative size of the random no-default buffer added to c_bar; not specified in the paper
CORE_FRACTION = 0.2         # fraction of nodes treated as "core" in CP graphs; paper Sec. 7.1 states "~20%"
CORE_CORE_MULT = 3.0        # core-core density multiplier relative to the baseline CP density d; not specified in the paper
CORE_PERIPHERY_MULT = 1.0   # core-periphery density multiplier relative to d; not specified in the paper
PERIPHERY_PERIPHERY_MULT = 0.3  # periphery-periphery density multiplier relative to d; not specified in the paper

# ---------------------------------------------------------------------------
# exp_1: Random Pointwise Clearing Validation (D-val instance, Sec. 7.2)
# ---------------------------------------------------------------------------
EXP1_N = 8                  # number of banks, paper Table 1 (D-val)
EXP1_M = 3                  # number of external assets, paper Table 1 (D-val)
EXP1_GRAPH_TYPE = "ER"      # Erdos-Renyi liability graph, paper Table 1
EXP1_DENSITY = 0.35         # ER edge density, paper Table 1
EXP1_SEED = 11              # random seed, paper Table 1
EXP1_RADIUS_RATIOS = (0.5, 1.0, 1.2, 2.0)  # epsilon / epsilon_ub tested, paper Table 2
EXP1_N_SHOCKS = 100         # independent random shocks per radius, paper Sec. 7.2

# ---------------------------------------------------------------------------
# exp_2: Pre-Insolvency Worst-Case Loss Curve (D-pre instance, Sec. 7.3)
# ---------------------------------------------------------------------------
EXP2_N = 12                 # number of banks, paper Table 1 (D-pre)
EXP2_M = 4                  # number of external assets, paper Table 1 (D-pre)
EXP2_GRAPH_TYPE = "CP"      # core-periphery liability graph, paper Table 1
EXP2_DENSITY = 0.28         # CP baseline edge density, paper Table 1
EXP2_SEED = 22              # random seed, paper Table 1

EXP2_GROWTH_SIZES_PAPER = (8, 16, 32, 64, 128)  # network sizes n, paper Table 4
EXP2_GROWTH_SIZES_RUN = (8, 16, 32, 64, 128)    # executed sizes (unchanged: growth itself is the result)
EXP2_GROWTH_SEEDS = (101, 202)  # two random seeds per size, paper Sec. 7.3; seed values not specified in the paper
EXP2_GROWTH_M = 4            # external assets in the growth experiment, paper Sec. 7.3
# The paper does not specify a per-size density rule for the growth experiment.
# We hold the expected mean in-degree constant across n (density(n) = d0*n0/n,
# anchored at the D-pre baseline d0=0.28, n0=12), rather than holding the edge
# density itself constant, since a constant density would make each network's
# in-degree grow linearly with n and is not the usual convention for network
# scalability studies.
EXP2_GROWTH_DENSITY_ANCHOR_N = 12    # anchor network size for the mean-degree rule
EXP2_GROWTH_DENSITY_ANCHOR_D = 0.28  # anchor density at the anchor size

# ---------------------------------------------------------------------------
# exp_3: Post-Insolvency Scalar Geometry (D-post instance, Sec. 7.4)
# ---------------------------------------------------------------------------
EXP3_N = 6                   # number of banks, paper Table 1 (D-post)
EXP3_M = 1                   # single external asset, paper Table 1 (D-post)
EXP3_SEED = 1                 # paper Table 1 (D-post)
EXP3_GRID_POINTS = 2000       # fine grid resolution over epsilon in [0, ~4 epsilon_ub], paper Sec. 7.4
EXP3_RADIUS_MULTIPLE = 4.0    # grid upper end as a multiple of epsilon_ub, paper Sec. 7.4 ("~4*epsilon_ub")
EXP3_KINK_PROBE = 1e-6        # +/- probe step around each detected kink, paper Sec. 7.4

# ---------------------------------------------------------------------------
# exp_4: Certified Bounds and Regime Generation (M-bound instance, Sec. 7.5)
# ---------------------------------------------------------------------------
EXP4_N = 25                  # number of banks, paper Table 1 (M-bound)
EXP4_M = 5                   # number of external assets, paper Table 1 (M-bound)
EXP4_GRAPH_TYPE = "CP"
EXP4_DENSITY = 0.12           # paper Table 1 (M-bound)
EXP4_SEED = 5                 # paper Table 1 (M-bound)
EXP4_RADIUS_RATIOS = (0.8, 1.0, 1.2, 1.5, 2.0, 3.0)  # epsilon/epsilon_ub tested, paper Table 6
EXP4_N_SCENARIOS_PAPER = 1536      # total scenarios for the lower bound, paper Sec. 7.5
EXP4_N_SCENARIOS_RUN = 1536        # executed scenario count (kept at the paper's value)
EXP4_DIRICHLET_CONCENTRATIONS = (0.5, 1.0, 5.0)  # concentration parameters for boundary sampling; not specified in the paper
EXP4_MAX_CELLS_PAPER = 81          # largest reported cell count (radius 3*epsilon_ub), paper Table 8
EXP4_MAX_CELLS_RUN = 81            # executed bisection cap (unchanged)
EXP4_REGIME_GEN_MAX_ITER = 40      # cap on the regime-generation loop, paper reports 26 iterations at 2*epsilon_ub (Table 7)

# ---------------------------------------------------------------------------
# exp_5: Scenario Lower Bounds on a 353-Bank Network (L-353 instance, Sec. 7.6)
# ---------------------------------------------------------------------------
EXP5_N = 353                  # number of banks, paper Table 1 (L-353)
EXP5_M = 5                    # number of external assets, paper Table 1 (L-353)
EXP5_GRAPH_TYPE = "CP"
EXP5_DENSITY = 0.02            # paper Table 1 (L-353)
EXP5_SEED = 353                # paper Table 1 (L-353)
EXP5_RADIUS_RATIOS = (0.0, 0.0776, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)  # epsilon/epsilon_ub tested, paper Table 10
EXP5_N_SCENARIOS_PAPER = 42        # scenarios per radius, paper Sec. 7.6
EXP5_N_SCENARIOS_RUN = 42          # executed scenario count (kept at the paper's value)
EXP5_LOCALIZED_RADII = (1.5, 2.0, 3.0)  # post-insolvency radii certified with localized bounds, paper Table 11
EXP5_MAX_CELLS_PAPER = 66          # largest reported cell count (radius 3*epsilon_ub), paper Table 11
EXP5_MAX_CELLS_RUN = 66            # executed bisection cap (unchanged)
EXP5_FIXED_POINT_TOL = 3e-9        # fixed-point residual tolerance, paper Sec. 7.6

# ---------------------------------------------------------------------------
# Numerical tolerances shared across experiments
# ---------------------------------------------------------------------------
LP_FEASIBILITY_TOL = 1e-7   # tolerance used to classify LP feasibility/infeasibility; not specified in the paper
BREAKPOINT_TRACE_TOL = 1e-9  # adaptive-bisection affine-fit tolerance for loss-curve tracing; not specified in the paper
