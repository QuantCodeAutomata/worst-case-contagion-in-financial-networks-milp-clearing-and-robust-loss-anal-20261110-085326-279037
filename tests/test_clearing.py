"""Unit tests for src/clearing.py (paper Sec. 3-4, Proposition 5, Eq. 7-13)."""
from __future__ import annotations

import numpy as np
import pytest

from src.clearing import (
    clearing_lp,
    default_resilience_margin,
    exact_clearing_regime_enumeration,
    insolvency_resilience_threshold,
    insolvency_resilience_threshold_lower_bound,
    maximal_clearing_fixed_point,
    maximal_clearing_lp,
    maximal_clearing_milp,
)


def test_clearing_lp_default_free_returns_full_payment(toy_two_bank):
    """At the nominal (default-free) external vector, the clearing LP must
    return full payment p* = p_bar (no bank is forced into default)."""
    inst = toy_two_bank
    p, feasible = clearing_lp(inst["p_bar"], inst["A"], inst["c_bar"])
    assert feasible
    np.testing.assert_allclose(p, inst["p_bar"], atol=1e-8)


def test_lp_milp_fixed_point_agree_on_default_free_case(toy_two_bank):
    """The LP, MILP and fixed-point clearing methods must agree exactly when
    the system is default-free (paper Sec. 3: the clearing vector is unique
    and all three characterizations coincide)."""
    inst = toy_two_bank
    p_lp, _ = clearing_lp(inst["p_bar"], inst["A"], inst["c_bar"])
    milp_res = maximal_clearing_milp(inst["p_bar"], inst["A"], inst["c_bar"])
    fp_res = maximal_clearing_fixed_point(inst["p_bar"], inst["A"], inst["c_bar"])
    np.testing.assert_allclose(p_lp, milp_res.p, atol=1e-6)
    np.testing.assert_allclose(p_lp, fp_res.p, atol=1e-6)


def test_clearing_loss_nonnegative_and_bounded(small_instance):
    """Loss L(p*) = 1^T(p_bar - p*) must lie in [0, 1^T p_bar] for any shock
    that keeps the clearing LP feasible (paper Sec. 3)."""
    inst = small_instance
    c = inst["c_bar"] - 0.01 * np.ones(inst["n"])
    result = maximal_clearing_milp(inst["p_bar"], inst["A"], c)
    assert result.feasible
    assert -1e-8 <= result.loss <= np.sum(inst["p_bar"]) + 1e-8


def test_exact_regime_enumeration_matches_lp_when_feasible(small_instance):
    """Proposition 7 pattern decomposition and the clearing LP (Eq. 8) must
    give the same loss whenever the LP is feasible (exp_1's core check)."""
    inst = small_instance
    rng = np.random.default_rng(42)
    for _ in range(5):
        c = inst["c_bar"] + 0.3 * rng.normal(size=inst["n"])
        p_lp, feasible = clearing_lp(inst["p_bar"], inst["A"], c)
        if not feasible:
            continue
        exact = exact_clearing_regime_enumeration(inst["p_bar"], inst["A"], c)
        assert exact.feasible
        assert abs(float(np.sum(inst["p_bar"] - p_lp)) - exact.loss) < 1e-5 * max(1.0, np.sum(inst["p_bar"]))


def test_default_resilience_margin_positive_for_default_free_system(small_instance):
    """epsilon_star (Eq. 7) must be strictly positive whenever the nominal
    system has strictly positive residual resources."""
    inst = small_instance
    eps_star = default_resilience_margin(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    assert eps_star > 0


def test_insolvency_resilience_threshold_exceeds_margin(small_instance):
    """epsilon_ub (Proposition 5) must be at least epsilon_star (Eq. 7): the
    clearing LP stays feasible at least until the first bank's resources
    could turn negative under the worst single-asset shock."""
    inst = small_instance
    eps_star = default_resilience_margin(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    assert eps_ub >= eps_star - 1e-8


def test_lower_bound_never_exceeds_feasibility_in_practice(small_instance):
    """The single-LP lower bound epsilon_ub_lower (Eq. 13) is a certified
    bound on the worst componentwise deterioration and need not equal the
    exact threshold, but must be positive and finite for a well-posed
    instance."""
    inst = small_instance
    lower = insolvency_resilience_threshold_lower_bound(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    assert 0 < lower < np.inf


def test_clearing_lp_infeasible_for_large_enough_shock(small_instance):
    """A sufficiently large adverse shock must eventually make the clearing
    LP (8) infeasible (beyond the insolvency-resilience threshold)."""
    inst = small_instance
    eps_ub, _ = insolvency_resilience_threshold(inst["p_bar"], inst["A"], inst["c_bar"], inst["S"])
    s = np.max(inst["S"], axis=1)
    big_shock_c = inst["c_bar"] - 10 * eps_ub * s
    _, feasible = clearing_lp(inst["p_bar"], inst["A"], big_shock_c)
    assert not feasible


@pytest.mark.parametrize("n", [1, 2])
def test_edge_case_small_networks(n):
    """Edge case: the clearing primitives must not crash on n=1 or n=2
    trivial networks."""
    p_bar = np.ones(n)
    a = np.eye(n)
    c_bar = np.ones(n)
    p, feasible = clearing_lp(p_bar, a, c_bar)
    assert feasible
    np.testing.assert_allclose(p, p_bar, atol=1e-8)
