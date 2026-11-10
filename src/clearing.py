"""Clearing-vector computation: LP (Eq. 8), MILP (Theorem 1), monotone
fixed-point iteration, and exact regime enumeration (Proposition 7 pattern
decomposition), plus the insolvency-resilience threshold (Proposition 5), its
single-LP lower bound (Eq. 13), and the default-resilience margin (Eq. 7).

All LPs use scipy.optimize.linprog with the HiGHS backend and all MILPs use
scipy.optimize.milp, as stated in the paper's numerical section (Sec. 7).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


@dataclass
class ClearingResult:
    p: np.ndarray            # clearing payment vector
    loss: float               # L(p) = 1^T (p_bar - p), not normalized
    feasible: bool            # whether the underlying LP/MILP solved feasible
    method: str               # "lp", "milp", or "fixed_point"


def default_resilience_margin(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                               s_matrix: np.ndarray) -> float:
    """epsilon_star, Eq. (7): min_{i: p_bar_i>0, s_i>0} r_bar_i / s_i, with
    s_i = ||sigma_i||_inf (the l1-perturbation dual norm) since S >= 0.
    """
    r_bar = c_bar + a.T @ p_bar - p_bar
    s = np.max(s_matrix, axis=1)  # ||sigma_i||_inf for S >= 0
    mask = (p_bar > 0) & (s > 0)
    if not np.any(mask):
        return np.inf
    return float(np.min(r_bar[mask] / s[mask]))


def clearing_lp(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray):
    """Solve the clearing LP (8): max 1^T p s.t. 0<=p<=p_bar, (I-A^T)p <= c.

    Returns (p, feasible). If infeasible, p is None.
    """
    n = len(p_bar)
    # linprog minimizes; maximize 1^T p -> minimize -1^T p
    obj = -np.ones(n)
    a_ub = (np.eye(n) - a.T)
    bounds = Bounds(lb=np.zeros(n), ub=p_bar)
    res = linprog(obj, A_ub=a_ub, b_ub=c, bounds=list(zip(bounds.lb, bounds.ub)),
                   method="highs")
    if not res.success:
        return None, False
    return np.clip(res.x, 0, p_bar), True


def maximal_clearing_lp(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray) -> ClearingResult:
    """Maximal clearing vector via the LP (8), valid only when it is feasible
    (i.e. r := c + A^T p_bar - p_bar >= 0 is not required; feasibility of the
    clearing LP is the actual condition, cf. Sec. 3).
    """
    p, feasible = clearing_lp(p_bar, a, c)
    if not feasible:
        return ClearingResult(p=None, loss=np.nan, feasible=False, method="lp")
    loss = float(np.sum(p_bar - p))
    return ClearingResult(p=p, loss=loss, feasible=True, method="lp")


def maximal_clearing_milp(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray) -> ClearingResult:
    """Exact MILP characterization of the maximal clearing vector (Theorem 1,
    MILP_c): valid for every c, including the post-insolvency regime.

        max 1^T p
        s.t. 0 <= p_i <= p_bar_i z_i
             p_i <= c_i + (A^T p)_i + M_i (1 - z_i),  M_i = max(-c_i, 0)
             z in {0,1}^n
    """
    n = len(p_bar)
    m_i = np.maximum(-c, 0.0)
    # Variables: [p (n), z (n)]
    num_vars = 2 * n
    integrality = np.concatenate([np.zeros(n), np.ones(n)])
    lb = np.concatenate([np.zeros(n), np.zeros(n)])
    ub = np.concatenate([p_bar, np.ones(n)])
    bounds = Bounds(lb=lb, ub=ub)

    # box constraint p_i - p_bar_i z_i <= 0
    a_box = np.hstack([np.eye(n), -np.diag(p_bar)])
    b_box = np.zeros(n)

    # resource constraint: p_i - (A^T p)_i - M_i (1 - z_i) <= c_i
    # => p_i - (A^T p)_i + M_i z_i <= c_i + M_i
    a_res = (np.eye(n) - a.T)
    a_res = np.hstack([a_res, np.diag(m_i)])
    b_res = c + m_i

    a_ub = np.vstack([a_box, a_res])
    b_ub = np.concatenate([b_box, b_res])
    constraints = LinearConstraint(a_ub, -np.inf, b_ub)

    obj = np.concatenate([-np.ones(n), np.zeros(n)])
    res = milp(obj, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success:
        return ClearingResult(p=None, loss=np.nan, feasible=False, method="milp")
    p = np.clip(res.x[:n], 0, p_bar)
    loss = float(np.sum(p_bar - p))
    return ClearingResult(p=p, loss=loss, feasible=True, method="milp")


def maximal_clearing_fixed_point(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray,
                                  tol: float = 3e-9, max_iter: int = 10_000) -> ClearingResult:
    """Monotone fixed-point iteration (Sec. 7.6):
        p^(0) = p_bar, p^(k+1) = T_c(p^(k)) = [min(p_bar, c + A^T p^(k))]_+.
    T_c is monotone and T_c(p_bar) <= p_bar, so the sequence decreases
    componentwise and converges to the maximal clearing vector.
    """
    p = p_bar.copy()
    residual = np.inf
    for _ in range(max_iter):
        p_next = np.clip(np.minimum(p_bar, c + a.T @ p), 0, None)
        residual = np.max(np.abs(p_next - p))
        p = p_next
        if residual < tol:
            break
    loss = float(np.sum(p_bar - p))
    return ClearingResult(p=p, loss=loss, feasible=True, method="fixed_point")


def exact_clearing_regime_enumeration(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray):
    """Exact maximal clearing vector by exhaustive enumeration of all 3^n
    bank states (zero payment, full payment, resource-bound partial payment),
    used as an independent oracle in exp_1 (D-val, n=8 -> 3^8 = 6561 regimes).

    For a fixed assignment of states, banks in the "partial" group satisfy
    p_i = c_i + (A^T p)_i exactly: a linear system in those variables (with
    the zero/full entries substituted). We solve this linear system for every
    one of the 3^n combinations in a vectorized sweep and keep the best
    feasible (box- and resource-consistent) solution, which by the
    Eisenberg-Noe lattice property is the maximal clearing vector.
    """
    n = len(p_bar)
    best_total = -np.inf
    best_p = None
    states = [0, 1, 2]  # 0 = zero payment, 1 = full payment, 2 = partial payment
    for combo in product(states, repeat=n):
        combo = np.array(combo)
        zero_idx = np.where(combo == 0)[0]
        full_idx = np.where(combo == 1)[0]
        partial_idx = np.where(combo == 2)[0]

        p = np.zeros(n)
        p[full_idx] = p_bar[full_idx]

        if partial_idx.size:
            # p_i = c_i + sum_j a_ji p_j for i in partial_idx
            # => (I - A_partial^T) p_partial = c_partial + A_full_contrib
            a_pp = a[np.ix_(partial_idx, partial_idx)]
            rhs = c[partial_idx] + (a[np.ix_(full_idx, partial_idx)].T @ p[full_idx]
                                     if full_idx.size else np.zeros(partial_idx.size))
            lin_sys = np.eye(partial_idx.size) - a_pp.T
            try:
                p_partial = np.linalg.solve(lin_sys, rhs)
            except np.linalg.LinAlgError:
                continue
            p[partial_idx] = p_partial

        if not np.all(p >= -1e-7):
            continue
        if not np.all(p <= p_bar + 1e-7):
            continue
        resources = c + a.T @ p
        # Zero-payment banks must have nonpositive resources (or zero payment
        # is consistent trivially when resources >= 0 too, but to keep this a
        # valid clearing *candidate* consistent with (4) we require:
        if zero_idx.size and np.any(resources[zero_idx] > 1e-6):
            continue
        if full_idx.size and np.any(p_bar[full_idx] - 1e-6 > resources[full_idx]):
            continue
        if partial_idx.size and np.any(resources[partial_idx] < -1e-6):
            continue
        total = float(np.sum(np.clip(p, 0, p_bar)))
        if total > best_total:
            best_total = total
            best_p = np.clip(p, 0, p_bar)
    if best_p is None:
        return ClearingResult(p=None, loss=np.nan, feasible=False, method="regime_enum")
    loss = float(np.sum(p_bar - best_p))
    return ClearingResult(p=best_p, loss=loss, feasible=True, method="regime_enum")


def insolvency_resilience_threshold(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                                     s_matrix: np.ndarray) -> tuple[float, dict]:
    """Exact insolvency-resilience threshold epsilon_ub (Proposition 5),
    for the l1-ball with S >= 0: only vertices q = -e_j, j=1..m, are needed.

        epsilon_q = sup{eps >= 0 : 0<=p<=p_bar, c_bar + eps S q + A^T p >= p}

    Each epsilon_q is obtained from the LP
        max_{p,eps} eps  s.t. 0<=p<=p_bar, (I-A^T)p - eps S q <= c_bar, eps>=0
    Returns (epsilon_ub, {"epsilon_q": array of per-asset values}).
    """
    n = len(p_bar)
    m = s_matrix.shape[1]
    eps_q = np.empty(m)
    for j in range(m):
        q = -np.eye(m)[j]
        s_q = s_matrix @ q  # = -S[:, j]
        # variables: [p (n), eps (1)], maximize eps
        obj = np.zeros(n + 1)
        obj[-1] = -1.0
        a_ub = np.hstack([np.eye(n) - a.T, -s_q.reshape(-1, 1)])
        b_ub = c_bar
        bounds = [(0, p_bar[i]) for i in range(n)] + [(0, None)]
        res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        eps_q[j] = res.x[-1] if res.success else np.inf
    epsilon_ub = float(np.min(eps_q))
    return epsilon_ub, {"epsilon_q": eps_q}


def insolvency_resilience_threshold_lower_bound(p_bar: np.ndarray, a: np.ndarray,
                                                 c_bar: np.ndarray, s_matrix: np.ndarray) -> float:
    """Certified lower bound epsilon_ub_lower (Eq. 13), a single LP using the
    componentwise worst deterioration s_i = ||sigma_i||_inf = max_j S_ij:

        max_{p,eps} eps  s.t. 0<=p<=p_bar, c_bar - eps*s + A^T p >= p, eps>=0
    """
    n = len(p_bar)
    s = np.max(s_matrix, axis=1)
    obj = np.zeros(n + 1)
    obj[-1] = -1.0
    a_ub = np.hstack([np.eye(n) - a.T, s.reshape(-1, 1)])
    b_ub = c_bar
    bounds = [(0, p_bar[i]) for i in range(n)] + [(0, None)]
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return float(res.x[-1]) if res.success else np.inf
