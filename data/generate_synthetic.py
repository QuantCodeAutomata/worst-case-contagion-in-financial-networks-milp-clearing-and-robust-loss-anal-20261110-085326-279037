"""Synthetic financial-network instance generation.

Implements the random-network generator described in Section 7.1 ("Experimental
instances and reported quantities") of the paper: directed Erdos-Renyi ER(d) and
core-periphery CP(d) liability graphs, log-normal positive liabilities rescaled
so that total interbank liability is of the same order as the number of banks
n, a nominal external vector c_bar constructed so the nominal system is
default-free, and a nonnegative exposure matrix S with larger exposures
assigned to banks that are relatively more vulnerable under full payment.

The paper does not specify the parameters of the log-normal distribution, the
exact core/periphery link-probability multipliers, or the magnitude of the
random no-default buffer; the choices below are documented assumptions (see
results/RESULTS.md, "Deviations from the paper").
"""
from __future__ import annotations

import numpy as np


def relative_liability_matrix(p_bar_matrix: np.ndarray) -> np.ndarray:
    """Row-stochastic matrix A of relative liabilities, paper Eq. (1).

    a_ij = p_bar_ij / p_bar_i if p_bar_i > 0; a_ii = 1 if p_bar_i = 0; else 0.
    """
    n = p_bar_matrix.shape[0]
    p_bar = p_bar_matrix.sum(axis=1)
    a = np.zeros((n, n))
    for i in range(n):
        if p_bar[i] > 0:
            a[i, :] = p_bar_matrix[i, :] / p_bar[i]
        else:
            a[i, i] = 1.0
    row_sums = a.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "A must be row-stochastic (A @ 1 = 1)"
    return a


def _ensure_min_out_degree(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Guarantee every bank owes at least one interbank liability (out-degree
    >= 1). A liability graph with an isolated zero-liability bank is a
    degenerate instance outside the "mutual liabilities" setting the paper's
    model is built on (Sec. 2): such a bank is excluded from the
    default-resilience margin (7) by construction, yet, if it carries nonzero
    external exposure, it can still drive the insolvency-resilience threshold
    (Proposition 5) below the margin, which is a generator artifact rather
    than a property the paper's experiments exhibit (Table 1 has eps* <
    eps_ub on every reported instance). Not specified explicitly in the
    paper; documented assumption."""
    n = adjacency.shape[0]
    out_degree = adjacency.sum(axis=1)
    for i in np.flatnonzero(out_degree == 0):
        candidates = [j for j in range(n) if j != i]
        adjacency[i, rng.choice(candidates)] = True
    return adjacency


def generate_er_graph(n: int, density: float, seed: int) -> np.ndarray:
    """Directed Erdos-Renyi adjacency ER(density), no self-loops."""
    rng = np.random.default_rng(seed)
    adjacency = rng.random((n, n)) < density
    np.fill_diagonal(adjacency, False)
    return _ensure_min_out_degree(adjacency, rng)


def generate_cp_graph(
    n: int,
    density: float,
    seed: int,
    core_fraction: float = 0.2,
    core_core_mult: float = 3.0,
    core_periphery_mult: float = 1.0,
    periphery_periphery_mult: float = 0.3,
) -> np.ndarray:
    """Directed core-periphery adjacency CP(density).

    The first ~20% of nodes are core nodes (paper Sec. 7.1). Core-core links
    are sampled with the largest probability, periphery-periphery links with
    the smallest, and core-periphery links with an intermediate probability.
    The paper does not specify the exact multipliers beyond this ordering;
    core_core_mult > core_periphery_mult > periphery_periphery_mult is a
    documented assumption enforcing that ordering while keeping density as
    the baseline (not specified in the paper beyond the table values).
    """
    rng = np.random.default_rng(seed)
    n_core = max(1, int(round(core_fraction * n)))
    is_core = np.zeros(n, dtype=bool)
    is_core[:n_core] = True

    prob = np.empty((n, n))
    core_mask = np.outer(is_core, is_core)
    periphery_mask = np.outer(~is_core, ~is_core)
    mixed_mask = ~core_mask & ~periphery_mask
    prob[core_mask] = min(1.0, density * core_core_mult)
    prob[periphery_mask] = min(1.0, density * periphery_periphery_mult)
    prob[mixed_mask] = min(1.0, density * core_periphery_mult)

    adjacency = rng.random((n, n)) < prob
    np.fill_diagonal(adjacency, False)
    return _ensure_min_out_degree(adjacency, rng)


def sample_liabilities(
    adjacency: np.ndarray,
    seed: int,
    lognormal_mu: float = 0.0,
    lognormal_sigma: float = 1.0,
) -> np.ndarray:
    """Positive liabilities on the edges of ``adjacency`` from a log-normal law,
    rescaled so total interbank liability 1^T P_bar 1 equals n (paper Sec. 7.1:
    "rescaled so that the total interbank liability is of the same order as
    the number of banks"). The log-normal parameters are not given in the
    paper; mu=0, sigma=1 (standard log-normal) is a documented assumption.
    """
    n = adjacency.shape[0]
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(mean=lognormal_mu, sigma=lognormal_sigma, size=(n, n))
    p_bar = raw * adjacency
    total = p_bar.sum()
    if total > 0:
        p_bar = p_bar * (n / total)
    return p_bar


def build_network(
    n: int,
    m: int,
    graph_type: str,
    density: float,
    seed: int,
    buffer_scale: float = 0.05,
    exposure_scale: float | None = None,
) -> dict:
    """Build a random instance (P_bar, A, c_bar, S) following paper Sec. 7.1.

    Parameters
    ----------
    n : number of banks.
    m : number of external assets.
    graph_type : "ER" or "CP".
    density : nominal edge density parameter d.
    seed : random seed (identifies the realization, as in Table 1).
    buffer_scale : relative size of the random no-default buffer added to
        c_bar (not specified in the paper; documented assumption).
    exposure_scale : overall scale of the exposure matrix S (not specified in
        the paper; chosen, if None, so that the median single-asset exposure
        is comparable to the median nominal residual resource, giving a
        stress radius of order 1).

    Returns a dict with keys: n, m, P_bar, A, p_bar, c_bar, S, r_bar, seed.
    """
    if graph_type == "ER":
        adjacency = generate_er_graph(n, density, seed)
    elif graph_type == "CP":
        adjacency = generate_cp_graph(n, density, seed)
    else:
        raise ValueError(f"Unknown graph_type {graph_type!r}; expected 'ER' or 'CP'")

    p_bar_matrix = sample_liabilities(adjacency, seed)
    a_matrix = relative_liability_matrix(p_bar_matrix)
    p_bar = p_bar_matrix.sum(axis=1)

    # Nominal external vector: compensate the positive part of
    # p_bar_i - (A^T p_bar)_i, then add a small random buffer, so that the
    # nominal system is default-free (paper Sec. 7.1): c_bar + A^T p_bar >= p_bar.
    at_pbar = a_matrix.T @ p_bar
    deficit = np.maximum(p_bar - at_pbar, 0.0)
    # Uniform(0, x) buffers have a minimum order statistic that shrinks toward
    # zero as n grows, which drives the default-resilience margin (7) to a
    # near-degenerate value for the most exposed bank. The paper's reported
    # instances (Table 1) all have a visibly positive eps* < eps_ub; drawing
    # the buffer from a range bounded away from zero keeps every bank's
    # no-default buffer non-trivial while remaining "small" relative to
    # p_bar.max() (not specified in the paper; documented assumption).
    rng = np.random.default_rng(seed + 10_000)
    scale = buffer_scale * max(p_bar.max(), 1.0)
    buffer = rng.uniform(0.3 * scale, scale, size=n)
    c_bar = deficit + buffer
    r_bar = c_bar + at_pbar - p_bar
    assert np.all(r_bar >= -1e-9), "nominal system must be default-free (r_bar >= 0)"

    # Exposure matrix S >= 0: larger exposures for banks that are relatively
    # more vulnerable under full payment (smaller r_bar means more vulnerable).
    rng2 = np.random.default_rng(seed + 20_000)
    order = np.argsort(r_bar)  # ascending: most vulnerable first
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(1, n + 1)
    vulnerability_weight = (n - rank + 1.0) / n  # in (0, 1], largest for most vulnerable
    base = rng2.uniform(0.5, 1.5, size=(n, m))
    s_matrix = vulnerability_weight[:, None] * base
    if exposure_scale is None:
        positive_r = r_bar[r_bar > 1e-12]
        reference = np.median(positive_r) if positive_r.size else 1.0
        current_median = np.median(s_matrix)
        exposure_scale = 0.5 * reference / max(current_median, 1e-12)
    s_matrix = s_matrix * exposure_scale
    assert np.all(s_matrix >= 0), "exposure matrix S must be nonnegative"

    return {
        "n": n,
        "m": m,
        "P_bar": p_bar_matrix,
        "A": a_matrix,
        "p_bar": p_bar,
        "c_bar": c_bar,
        "S": s_matrix,
        "r_bar": r_bar,
        "seed": seed,
        "graph_type": graph_type,
        "density": density,
    }


def build_d_post() -> dict:
    """Hand-crafted scalar (m=1, n=6) instance 'D-post' used for exact
    post-insolvency regime-enumeration experiments (paper Sec. 7.4). The
    paper states this instance is fixed (not random) and "chosen to make the
    post-insolvency geometry visible," but its exact numerical parameters are
    not given in the retrieved text (stated to be in the authors'
    reproducibility package, not accessible here). The instance below is
    constructed from scratch to reproduce the same qualitative structure:
    n=6 banks, m=1 asset, long-only exposure (S >= 0), a nominally
    default-free system, and enough heterogeneity across banks that
    successive price declines trigger a sequence of distinct zero-payment
    patterns.
    """
    n = 6
    rng = np.random.default_rng(1)
    # Directed cyclic/ring liability structure with extra cross-links, fixed
    # by construction (seed=1 only fixes the illustrative liability weights).
    p_bar_matrix = np.zeros((n, n))
    edges = [
        (0, 1, 1.0), (1, 2, 1.4), (2, 3, 0.8), (3, 4, 1.2), (4, 5, 0.9),
        (5, 0, 1.1), (1, 3, 0.6), (2, 4, 0.5), (0, 2, 0.4), (3, 5, 0.7),
    ]
    for i, j, w in edges:
        p_bar_matrix[i, j] = w
    a_matrix = relative_liability_matrix(p_bar_matrix)
    p_bar = p_bar_matrix.sum(axis=1)

    at_pbar = a_matrix.T @ p_bar
    deficit = np.maximum(p_bar - at_pbar, 0.0)
    buffer = rng.uniform(0.0, 0.05, size=n)
    c_bar = deficit + buffer
    r_bar = c_bar + at_pbar - p_bar
    assert np.all(r_bar >= -1e-9)

    # A single long-only exposure, heterogeneous in magnitude across banks so
    # that the price decline hits different banks at different radii.
    s_vector = np.array([0.9, 1.6, 0.5, 1.3, 1.1, 0.7])
    s_matrix = s_vector.reshape(n, 1)

    return {
        "n": n,
        "m": 1,
        "P_bar": p_bar_matrix,
        "A": a_matrix,
        "p_bar": p_bar,
        "c_bar": c_bar,
        "S": s_matrix,
        "r_bar": r_bar,
        "seed": 1,
        "graph_type": "fixed",
        "density": None,
    }
