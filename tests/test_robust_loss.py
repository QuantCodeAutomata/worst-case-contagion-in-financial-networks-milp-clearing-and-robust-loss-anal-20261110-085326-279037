"""Unit tests for src/robust_loss.py (pre-insolvency robust dual, Theorem 14)."""
from __future__ import annotations

import numpy as np

from src.clearing import (
    default_resilience_margin,
    insolvency_resilience_threshold,
    maximal_clearing_milp,
)
from src.robust_loss import eta_wc_point, trace_loss_curve


def test_eta_wc_zero_below_default_resilience_margin(small_instance):
    """Below epsilon_star the system cannot default (Eq. 7), so the
    worst-case loss must be (numerically) zero."""
    inst = small_instance
    eps_star = default_resilience_margin(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    val, _ = eta_wc_point(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], 0.5 * eps_star)
    assert val < 1e-6


def test_eta_wc_matches_worst_case_milp_clearing(small_instance):
    """eta_wc(eps) (robust dual, Eq. 43) must equal the loss of the maximal
    clearing vector at the realized worst-case single-asset shock
    delta* = -eps * e_j for the winning direction j (Eq. 42)."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    eps = 0.6 * eps_ub
    val, piece = eta_wc_point(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], eps)
    delta = -eps * np.eye(inst["m"])[piece.asset]
    c = inst["c_bar"] + inst["S"] @ delta
    clearing = maximal_clearing_milp(inst["p_bar"], inst["A"], c)
    assert clearing.feasible
    assert abs(val - clearing.loss) < 1e-5 * max(1.0, clearing.loss)


def test_loss_curve_nondecreasing_and_convex_on_pre_insolvency_range(small_instance):
    """Theorem 14: on [0, epsilon_ub] the worst-case loss curve is convex and
    nondecreasing (checked via the tangent-line / convexity inequality on a
    uniform grid of radii)."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    grid = np.linspace(0.0, eps_ub, 9)
    values = [eta_wc_point(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], e)[0] for e in grid]
    diffs = np.diff(values)
    assert np.all(diffs >= -1e-7), "loss curve must be nondecreasing on [0, epsilon_ub]"
    second_diffs = np.diff(diffs)
    assert np.all(second_diffs >= -1e-6), "loss curve must be convex (piecewise-linear slopes nondecreasing)"


def test_trace_loss_curve_breakpoints_within_range(small_instance):
    """Every identified breakpoint must lie strictly inside (0, epsilon_ub)
    and have a right slope at least as large as the left slope (convexity)."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    trace = trace_loss_curve(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], eps_ub)
    for bp in trace["breakpoints"]:
        assert 0.0 <= bp["epsilon"] <= eps_ub + 1e-6
        assert bp["right_slope"] >= bp["left_slope"] - 1e-8
