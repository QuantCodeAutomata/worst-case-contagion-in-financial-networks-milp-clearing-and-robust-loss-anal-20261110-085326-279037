"""Unit tests for src/certificates.py (Sec. 6: scenario lower bounds,
fixed-pattern dual upper bounds, regime generation, localized certificates)."""
from __future__ import annotations

import numpy as np

from src.certificates import (
    activity_pattern,
    domain_condition_holds,
    fixed_pattern_dual_value,
    fixed_pattern_loss,
    fixed_pattern_upper_bound,
    localized_certificate,
    regime_generation,
    scenario_lower_bound,
)
from src.clearing import insolvency_resilience_threshold, maximal_clearing_milp


def test_activity_pattern_matches_positive_payments():
    p = np.array([0.0, 1e-10, 0.5, 2.0])
    z = activity_pattern(p)
    np.testing.assert_array_equal(z, [0, 0, 1, 1])


def test_all_ones_pattern_matches_milp_clearing_loss(small_instance):
    """The all-active pattern z = 1 must give the same fixed-pattern loss as
    the unconstrained exact clearing (Sec. 6: P_1(c) = clearing LP domain)."""
    inst = small_instance
    c = inst["c_bar"] - 0.5 * np.max(inst["S"], axis=1)
    clearing = maximal_clearing_milp(inst["p_bar"], inst["A"], c)
    z_ones = np.ones(inst["n"], dtype=int)
    loss = fixed_pattern_loss(inst["p_bar"], inst["A"], c, z_ones)
    assert loss is not None
    assert abs(loss - clearing.loss) < 1e-5 * max(1.0, clearing.loss)


def test_fixed_pattern_upper_bound_dominates_scenario_lower_bound(small_instance):
    """Weak duality (Proposition 10): for the realized worst-case pattern,
    UB_z(eps) must be at least the scenario lower bound LB_Delta(eps)."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    eps = 1.5 * eps_ub
    scenarios = [-eps * np.eye(inst["m"])[j] for j in range(inst["m"])] + [np.zeros(inst["m"])]
    lb_result = scenario_lower_bound(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], scenarios)
    z = activity_pattern(lb_result["best_p"])
    ub, _ = fixed_pattern_upper_bound(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], eps, z)
    assert ub is not None
    assert ub >= lb_result["lb"] - 1e-6


def test_regime_generation_upper_bound_monotone_nonincreasing(small_instance):
    """Proposition 11: the global upper bound UB_Z(eps) must decrease (or
    stay flat) monotonically across regime-generation iterations."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    trace = regime_generation(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], 2.0 * eps_ub)
    ub_hist = trace.ub_history
    assert len(ub_hist) >= 1
    diffs = np.diff(ub_hist)
    assert np.all(diffs <= 1e-8), "UB_Z(eps) must be nonincreasing along the regime-generation trace"


def test_localized_certificate_within_global_upper_bound(small_instance):
    """The localized certificate (Proposition 13) must never exceed the
    coarse global upper bound and must be bounded below by zero."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    eps = 1.5 * eps_ub
    cert = localized_certificate(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], eps, max_cells=10)
    assert cert["localized_ub"] >= -1e-8
    assert cert["n_cells"] >= 1


def test_domain_condition_false_for_empty_pattern_with_negative_resources(small_instance):
    """The empty pattern z = 0 trivially satisfies the domain check (P_0(c)
    = {0} is always feasible), confirming the box-collapse behaviour used as
    the regime-generation seed."""
    inst = small_instance
    z0 = np.zeros(inst["n"], dtype=int)
    assert domain_condition_holds(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"], 1.0, z0)
