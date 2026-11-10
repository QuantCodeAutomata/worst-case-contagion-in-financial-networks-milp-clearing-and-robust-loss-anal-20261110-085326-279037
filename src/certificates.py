"""Certified lower and upper bounds for the post-insolvency worst-case loss
(Sec. 6): scenario lower bounds, fixed-pattern dual upper bounds
(Proposition 10), regime generation (Proposition 11), and localized
fixed-pattern certificates via adaptive simplex bisection (Proposition 13).

A payment pattern z in {0,1}^n marks which banks are "active" (may make a
positive payment); for z_i = 0 the clearing LP forces p_i = 0, which is
implemented here by zeroing the nominal liability of bank i, i.e. solving the
clearing LP (8) with p_bar replaced by z * p_bar (Sec. 6, P_z(c)).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from src.clearing import clearing_lp, maximal_clearing_milp, maximal_clearing_fixed_point


def activity_pattern(p: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """z_i = 1 if bank i makes a positive payment (p_i > tol)."""
    return (p > tol).astype(int)


def fixed_pattern_primal_feasible(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray,
                                   z: np.ndarray) -> tuple[bool, np.ndarray | None]:
    """Domain check: is P_z(c) nonempty? Solve the fixed-pattern clearing LP

        max 1^T p  s.t.  0 <= p <= Z p_bar,  Z (I - A^T) p <= Z c   (Sec. 6)

    The resource constraint is scaled by Z = diag(z) so it binds only on
    active banks (z_i = 1); for an inactive bank (z_i = 0, hence p_i = 0 via
    the box bound) the row collapses to the trivial 0 <= 0, matching the
    big-M relaxation used in the exact MILP (Theorem 1) since A, p >= 0
    implies (A^T p)_i >= 0 always. Imposing the unmasked constraint
    (I - A^T) p <= c on inactive rows (as in ``clearing_lp`` with p_bar
    replaced by z*p_bar) would spuriously reject patterns whenever c_i < 0
    for an inactive bank i.
    """
    n = len(p_bar)
    z = np.asarray(z, dtype=float)
    obj = -np.ones(n)
    a_ub = np.diag(z) @ (np.eye(n) - a.T)
    b_ub = z * c
    bounds = list(zip(np.zeros(n), z * p_bar))
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return False, None
    return True, np.clip(res.x, 0, p_bar)


def fixed_pattern_loss(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray, z: np.ndarray):
    """Exact fixed-pattern clearing loss 1^T (p_bar - p*) for pattern z, or
    None if P_z(c) is empty."""
    feasible, p = fixed_pattern_primal_feasible(p_bar, a, c, z)
    if not feasible:
        return None
    return float(np.sum(p_bar - p))


def fixed_pattern_dual_value(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                              s_matrix: np.ndarray, eps: float, asset: int,
                              z: np.ndarray) -> float | None:
    """One asset-direction LP of the fixed-pattern dual (Proposition 10):

        max_{beta,lambda>=0} 1^T p_bar - (Z p_bar)^T beta - (Z c_bar)^T lambda
                              + eps (S^T Z lambda)_asset
        s.t.  beta + (I - A) Z lambda >= 1
    """
    n = len(p_bar)
    z_diag = np.diag(z)
    c_vec = np.concatenate([z * p_bar, z * (c_bar - eps * s_matrix[:, asset])])
    a_ub = np.hstack([-np.eye(n), -(np.eye(n) - a) @ z_diag])
    b_ub = -np.ones(n)
    bounds = [(0, None)] * (2 * n)
    res = linprog(c_vec, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    return float(np.sum(p_bar) - res.fun)


def fixed_pattern_upper_bound(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                               s_matrix: np.ndarray, eps: float, z: np.ndarray):
    """UB_z(eps) = max over m asset directions of the fixed-pattern dual LP.
    Returns (UB_z, best_asset) or (None, None) if unbounded for every
    direction (z fails the robust-domain condition)."""
    m = s_matrix.shape[1]
    best_val, best_j = None, None
    for j in range(m):
        val = fixed_pattern_dual_value(p_bar, a, c_bar, s_matrix, eps, j, z)
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val, best_j = val, j
    return best_val, best_j


def domain_condition_holds(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                            s_matrix: np.ndarray, eps: float, z: np.ndarray) -> bool:
    """Checks P_z(c_bar + S delta) nonempty at every vertex of the adverse
    l1 simplex {0, -eps e_1, ..., -eps e_m} (Sec. 6: sufficient for the
    long-only l1 case since the simplex is the convex hull of these points).
    """
    m = s_matrix.shape[1]
    feasible, _ = fixed_pattern_primal_feasible(p_bar, a, c_bar, z)
    if not feasible:
        return False
    for j in range(m):
        delta = -eps * np.eye(m)[j]
        feasible, _ = fixed_pattern_primal_feasible(p_bar, a, c_bar + s_matrix @ delta, z)
        if not feasible:
            return False
    return True


@dataclass
class RegimeGenerationTrace:
    patterns: list = field(default_factory=list)
    ub_history: list = field(default_factory=list)
    n_candidate: int = 0
    n_finite: int = 0


def regime_generation(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                       s_matrix: np.ndarray, eps: float, max_iter: int = 40,
                       tol: float = 1e-10) -> RegimeGenerationTrace:
    """Regime generation loop (Proposition 11): iteratively refine the global
    upper bound UB_Z(eps) = min_{z in Z} UB_z(eps) by solving the current
    best fixed-pattern robust problem, extracting its worst-case shock, and
    adding the resulting realized clearing pattern to the candidate set.
    """
    n = len(p_bar)
    candidates = [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    seen = {tuple(z) for z in candidates}
    trace = RegimeGenerationTrace()
    ub_global = np.inf
    for _ in range(max_iter):
        best_z, best_val, best_j = None, np.inf, None
        n_finite_this_round = 0
        for z in candidates:
            val, j = fixed_pattern_upper_bound(p_bar, a, c_bar, s_matrix, eps, z)
            if val is None:
                continue
            n_finite_this_round += 1
            if val < best_val:
                best_val, best_z, best_j = val, z, j
        trace.n_finite = max(trace.n_finite, n_finite_this_round)
        trace.n_candidate = len(candidates)
        if best_z is None:
            break
        trace.ub_history.append(best_val)
        if best_val >= ub_global - tol:
            break
        ub_global = best_val
        delta_worst = -eps * np.eye(s_matrix.shape[1])[best_j]
        c_worst = c_bar + s_matrix @ delta_worst
        clearing = maximal_clearing_milp(p_bar, a, c_worst)
        if not clearing.feasible:
            break
        z_hat = activity_pattern(clearing.p)
        key = tuple(z_hat)
        if key in seen:
            break
        seen.add(key)
        candidates.append(z_hat)
    trace.patterns = candidates
    return trace


def global_upper_bound(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                        s_matrix: np.ndarray, eps: float, patterns: list) -> float:
    """UB_Z(eps) = min_{z in Z} UB_z(eps) over a finite candidate family."""
    best = np.inf
    for z in patterns:
        val, _ = fixed_pattern_upper_bound(p_bar, a, c_bar, s_matrix, eps, z)
        if val is not None and val < best:
            best = val
    return best


def scenario_lower_bound(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                          s_matrix: np.ndarray, scenarios: list,
                          clearing_method: str = "milp") -> dict:
    """LB_Delta(eps) = max_{delta in Delta} eta*(c_bar + S delta) (Sec. 6),
    the scenario lower bound evaluated via exact clearing (MILP, Theorem 1,
    or the monotone fixed-point iteration for large instances).
    """
    best_loss, best_p, best_delta = -np.inf, None, None
    patterns = set()
    times = []
    lp_feasible_best = None
    for delta in scenarios:
        c = c_bar + s_matrix @ delta
        t0 = time.perf_counter()
        if clearing_method == "milp":
            result = maximal_clearing_milp(p_bar, a, c)
        else:
            result = maximal_clearing_fixed_point(p_bar, a, c)
        times.append(time.perf_counter() - t0)
        if not result.feasible:
            continue
        patterns.add(tuple(activity_pattern(result.p)))
        _, lp_feas = clearing_lp(p_bar, a, c)
        if result.loss > best_loss:
            best_loss = result.loss
            best_p = result.p
            best_delta = delta
            lp_feasible_best = lp_feas
    return {
        "lb": best_loss,
        "best_p": best_p,
        "best_delta": best_delta,
        "n_scenarios": len(scenarios),
        "n_patterns": len(patterns),
        "lp_feasible_at_best": lp_feasible_best,
        "median_time": float(np.median(times)) if times else np.nan,
    }


def adverse_simplex_vertices(m: int, eps: float) -> list[np.ndarray]:
    """Vertices of the adverse l1 simplex U_eps^- = conv{0, -eps e_1, ..., -eps e_m}
    for S >= 0 (long-only exposures), Sec. 6."""
    vertices = [np.zeros(m)]
    for j in range(m):
        vertices.append(-eps * np.eye(m)[j])
    return vertices


@dataclass
class Cell:
    vertices: list  # list of np.ndarray, the cell's simplex vertices
    inherited_patterns: list = field(default_factory=list)
    ub: float = np.inf
    best_pattern: np.ndarray | None = None


def _cell_candidates(p_bar, a, c_bar, s_matrix, cell: Cell, n_banks: int):
    """Local candidate patterns: exact clearing activity at each vertex and
    the centroid, inherited parent patterns, and pairwise componentwise
    intersections of these (Proposition 13)."""
    cands = list(cell.inherited_patterns)
    local = []
    for v in cell.vertices:
        c = c_bar + s_matrix @ v
        result = maximal_clearing_milp(p_bar, a, c)
        if result.feasible:
            local.append(activity_pattern(result.p))
    centroid = np.mean(cell.vertices, axis=0)
    c = c_bar + s_matrix @ centroid
    result = maximal_clearing_milp(p_bar, a, c)
    if result.feasible:
        local.append(activity_pattern(result.p))
    cands.extend(local)
    # componentwise intersections of local regimes
    for i in range(len(local)):
        for j in range(i + 1, len(local)):
            cands.append(np.minimum(local[i], local[j]))
    # de-duplicate
    uniq = {}
    for z in cands:
        uniq[tuple(z)] = np.array(z)
    n_local_clearings = len(cell.vertices) + 1
    return list(uniq.values()), n_local_clearings


def _evaluate_cell(p_bar, a, c_bar, s_matrix, cell: Cell, n_banks: int):
    """Cell upper bound: min over accepted patterns of the max-vertex
    fixed-pattern loss, where a pattern is accepted only if its restricted
    LP is feasible at every cell vertex (Proposition 13)."""
    candidates, n_local_clearings = _cell_candidates(p_bar, a, c_bar, s_matrix, cell, n_banks)
    n_pattern_lps = 0
    best_ub, best_z = np.inf, None
    for z in candidates:
        losses = []
        ok = True
        for v in cell.vertices:
            c = c_bar + s_matrix @ v
            n_pattern_lps += 1
            loss = fixed_pattern_loss(p_bar, a, c, z)
            if loss is None:
                ok = False
                break
            losses.append(loss)
        if not ok:
            continue
        cell_ub = max(losses)
        if cell_ub < best_ub:
            best_ub, best_z = cell_ub, z
    cell.ub = best_ub
    cell.best_pattern = best_z
    return candidates, n_local_clearings, n_pattern_lps


def localized_certificate(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                           s_matrix: np.ndarray, eps: float, max_cells: int = 80,
                           gap_tol: float = 1e-7) -> dict:
    """Localized fixed-pattern certificates (Proposition 13): adaptive
    longest-edge bisection of the adverse l1 simplex U_eps^-. On each cell, a
    local candidate family of payment patterns is formed and the cell with
    the largest local upper bound is bisected, until ``max_cells`` cells are
    reached or the certificate stabilizes (two consecutive rounds with no
    improvement in the global localized upper bound).
    """
    n = len(p_bar)
    m = s_matrix.shape[1]
    t0 = time.perf_counter()
    root = Cell(vertices=adverse_simplex_vertices(m, eps))
    _evaluate_cell(p_bar, a, c_bar, s_matrix, root, n)
    cells = [root]
    total_local_clearings = len(root.vertices) + 1
    total_pattern_lps = 0
    history = [max(c.ub for c in cells)]
    while len(cells) < max_cells:
        idx = int(np.argmax([c.ub for c in cells]))
        cell = cells[idx]
        verts = cell.vertices
        # longest-edge bisection
        best_pair, best_len = None, -1
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                d = np.linalg.norm(verts[i] - verts[j])
                if d > best_len:
                    best_len, best_pair = d, (i, j)
        if best_len < 1e-12:
            break
        i, j = best_pair
        midpoint = 0.5 * (verts[i] + verts[j])
        verts_a = [midpoint if k == j else verts[k] for k in range(len(verts))]
        verts_b = [midpoint if k == i else verts[k] for k in range(len(verts))]
        inherited = [cell.best_pattern] if cell.best_pattern is not None else []
        inherited += cell.inherited_patterns
        cell_a = Cell(vertices=verts_a, inherited_patterns=inherited)
        cell_b = Cell(vertices=verts_b, inherited_patterns=inherited)
        _, lc_a, lp_a = _evaluate_cell(p_bar, a, c_bar, s_matrix, cell_a, n)
        _, lc_b, lp_b = _evaluate_cell(p_bar, a, c_bar, s_matrix, cell_b, n)
        total_local_clearings += lc_a + lc_b
        total_pattern_lps += lp_a + lp_b
        cells.pop(idx)
        cells.extend([cell_a, cell_b])
        current = max(c.ub for c in cells)
        history.append(current)
        if len(history) > 2 and abs(history[-1] - history[-2]) < gap_tol and abs(history[-2] - history[-3]) < gap_tol:
            break
    wall_time = time.perf_counter() - t0
    global_ub = max(c.ub for c in cells)
    return {
        "localized_ub": global_ub,
        "n_cells": len(cells),
        "n_local_clearings": total_local_clearings,
        "n_pattern_lps": total_pattern_lps,
        "wall_time": wall_time,
        "history": history,
    }
