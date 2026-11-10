"""Pre-insolvency worst-case loss curve on [0, epsilon_ub]: robust dual
representation (Theorem 14), finite-envelope tracing, candidate breakpoints
(Corollary 16), and breakpoint diagnostics (Sec. 5.3: outer shock-direction
switches vs. inner default-set switches).

For the l1 perturbation set with S >= 0 (long-only exposures, used throughout
the paper's numerical section), the worst-case loss is

    eta_wc(eps) = max_{j=1,...,m} max_{(beta,lambda) in X}
                  { (1-beta)^T p_bar - c_bar^T lambda + eps (S^T lambda)_j }

with X = {(beta,lambda) in R_+^n x R_+^n : beta - 1 + (I_n - A) lambda >= 0}
(Eq. 38, 43). Each inner problem is an LP in (beta, lambda); solving it for a
given eps and j returns one extreme point of X, hence one affine piece
(gamma, mu) of the envelope (Eq. 44-47).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog


@dataclass
class AffinePiece:
    gamma: float
    mu: float
    asset: int


def robust_dual_lp(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                    s_matrix: np.ndarray, eps: float, asset: int) -> AffinePiece | None:
    """Solve the robust dual LP (Eq. 43 inner problem) for one asset direction
    j = ``asset`` at radius ``eps``. Returns the affine piece (gamma, mu)
    attained at the optimal extreme point of X, or None if unbounded
    (eps > epsilon_ub for this direction).
    """
    n = len(p_bar)
    s_j = s_matrix[:, asset]
    # minimize p_bar^T beta + (c_bar - eps*s_j)^T lambda  (negative of objective)
    c_vec = np.concatenate([p_bar, c_bar - eps * s_j])
    # constraint: beta + (I - A) lambda >= 1  =>  -beta - (I-A) lambda <= -1
    a_ub = np.hstack([-np.eye(n), -(np.eye(n) - a)])
    b_ub = -np.ones(n)
    bounds = [(0, None)] * (2 * n)
    res = linprog(c_vec, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    beta, lam = res.x[:n], res.x[n:]
    gamma = float(np.sum(p_bar) - p_bar @ beta - c_bar @ lam)
    mu = float(s_j @ lam)
    return AffinePiece(gamma=gamma, mu=mu, asset=asset)


def eta_wc_point(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                  s_matrix: np.ndarray, eps: float) -> tuple[float, AffinePiece]:
    """eta_wc(eps) = max over m asset-direction LPs (Eq. 43), returning the
    value and the winning affine piece.
    """
    m = s_matrix.shape[1]
    best_val, best_piece = -np.inf, None
    for j in range(m):
        piece = robust_dual_lp(p_bar, a, c_bar, s_matrix, eps, j)
        if piece is None:
            continue
        val = piece.gamma + piece.mu * eps
        if val > best_val:
            best_val, best_piece = val, piece
    return best_val, best_piece


def _pieces_close(p1: AffinePiece, p2: AffinePiece, tol: float = 1e-6) -> bool:
    return abs(p1.mu - p2.mu) < tol and abs(p1.gamma - p2.gamma) < tol


def trace_loss_curve(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                      s_matrix: np.ndarray, eps_ub: float, tol: float = 1e-9,
                      max_depth: int = 30) -> dict:
    """Adaptive bisection trace of eta_wc on [0, eps_ub] (Theorem 14):
    identifies the finite affine pieces and the breakpoints between them by
    recursively testing whether the curve is affine over an interval (value
    at the midpoint matches linear interpolation of the endpoints) and, if
    not, splitting the interval. This implements the paper's Remark 17
    ("generate active pieces by solving the LPs at selected radii").

    Returns a dict with: "segments" (list of (lo, hi, AffinePiece)) and
    "breakpoints" (list of dicts with eps, left_slope, right_slope).
    """
    val_0, piece_0 = eta_wc_point(p_bar, a, c_bar, s_matrix, 0.0)
    val_ub, piece_ub = eta_wc_point(p_bar, a, c_bar, s_matrix, eps_ub)
    scale = max(1.0, abs(val_ub))
    segments: list[tuple[float, float, AffinePiece]] = []

    def recurse(lo, hi, val_lo, piece_lo, val_hi, piece_hi, depth):
        if depth >= max_depth or (hi - lo) < 1e-9 * max(eps_ub, 1.0):
            segments.append((lo, hi, piece_lo))
            return
        mid = 0.5 * (lo + hi)
        val_mid, piece_mid = eta_wc_point(p_bar, a, c_bar, s_matrix, mid)
        predicted = val_lo + (val_hi - val_lo) * (mid - lo) / (hi - lo)
        if abs(val_mid - predicted) < tol * scale:
            segments.append((lo, hi, piece_lo))
            return
        recurse(lo, mid, val_lo, piece_lo, val_mid, piece_mid, depth + 1)
        recurse(mid, hi, val_mid, piece_mid, val_hi, piece_hi, depth + 1)

    recurse(0.0, eps_ub, val_0, piece_0, val_ub, piece_ub, 0)
    segments.sort(key=lambda s: s[0])

    # merge adjacent segments sharing (numerically) the same piece
    merged: list[tuple[float, float, AffinePiece]] = []
    for lo, hi, piece in segments:
        if merged and _pieces_close(merged[-1][2], piece):
            prev_lo, _, prev_piece = merged[-1]
            merged[-1] = (prev_lo, hi, prev_piece)
        else:
            merged.append((lo, hi, piece))

    breakpoints = []
    for k in range(len(merged) - 1):
        _, hi, piece_left = merged[k]
        _, _, piece_right = merged[k + 1]
        if abs(piece_left.mu - piece_right.mu) < 1e-9:
            continue
        eps0 = (piece_right.gamma - piece_left.gamma) / (piece_left.mu - piece_right.mu)
        if not (0.0 < eps0 < eps_ub):
            eps0 = hi  # fall back to the traced boundary
        left_slope = min(piece_left.mu, piece_right.mu)
        right_slope = max(piece_left.mu, piece_right.mu)
        breakpoints.append({
            "epsilon": eps0,
            "left_slope": left_slope,
            "right_slope": right_slope,
            "left_asset": piece_left.asset,
            "right_asset": piece_right.asset,
        })

    return {"segments": merged, "breakpoints": breakpoints,
            "value_at_0": val_0, "value_at_eps_ub": val_ub}


def clearing_activity_sets(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                            s_matrix: np.ndarray, eps: float, asset: int,
                            tol: float = 1e-7) -> dict:
    """Breakpoint diagnostics (Sec. 5.3): for the worst-case perturbation
    delta* = -eps * e_asset (long-only l1 case, Eq. 42), solve the clearing
    LP (8) and partition banks into F (full payment), D (default,
    p* < p_bar), and R (resource constraint tight).
    """
    from src.clearing import clearing_lp

    delta = -eps * np.eye(s_matrix.shape[1])[asset]
    c = c_bar + s_matrix @ delta
    p, feasible = clearing_lp(p_bar, a, c)
    if not feasible:
        return {"feasible": False}
    resources = c + a.T @ p
    full_payment = set(np.where(p >= p_bar - tol)[0].tolist())
    default = set(np.where(p < p_bar - tol)[0].tolist())
    resource_tight = set(np.where(np.abs(p - resources) < tol)[0].tolist())
    return {
        "feasible": True,
        "p": p,
        "full_payment": full_payment,
        "default": default,
        "resource_tight": resource_tight,
    }


def breakpoint_mechanism(p_bar: np.ndarray, a: np.ndarray, c_bar: np.ndarray,
                          s_matrix: np.ndarray, breakpoint: dict, step: float = 1e-4) -> dict:
    """Diagnose whether a breakpoint is an outer shock-direction switch
    (the active asset direction j changes) or an inner default-set switch
    (the active asset is the same but the default set D(eps) changes),
    following Sec. 5.3.
    """
    eps0 = breakpoint["epsilon"]
    j_left, j_right = breakpoint["left_asset"], breakpoint["right_asset"]
    outer_switch = j_left != j_right
    left = clearing_activity_sets(p_bar, a, c_bar, s_matrix, max(eps0 - step, 0.0), j_left)
    right = clearing_activity_sets(p_bar, a, c_bar, s_matrix, eps0 + step, j_right)
    default_change = None
    if left.get("feasible") and right.get("feasible"):
        default_change = sorted(left["default"].symmetric_difference(right["default"]))
    return {
        "epsilon": eps0,
        "mechanism": "outer shock-direction switch" if outer_switch else "inner default-set switch",
        "default_set_left": sorted(left.get("default", set())),
        "default_set_right": sorted(right.get("default", set())),
        "banks_changed": [b + 1 for b in default_change] if default_change is not None else [],
    }
