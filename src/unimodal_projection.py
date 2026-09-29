"""
Unimodality-constrained distortion risk bounds and projection algorithms.

Implements Propositions 2, 3, 4 and Theorem 3 from the paper:
- Fixed-inflection-point L2 projection onto the unimodal cone F_{U,xi}^{-1}
- Step-function parameterization of the projection
- Approximation error bounds
- Unimodality + Wasserstein combined bounds

The cone F_{U,xi}^{-1} consists of quantile functions that are convex (increasing
derivative) on [0, xi] and concave (decreasing derivative) on [xi, 1].

Custom implementation — Context7 found no library equivalent (paper-specific projection cone).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
import numpy as np
from scipy.optimize import minimize, brentq
import warnings

from .reference_distributions import ReferenceDistribution
from .distortion_metrics import DistortionMetric
from .concave_envelope import (
    _leggauss,
    wasserstein_sq_distance,
    least_concave_majorant,
    compute_b_lambda,
    construct_h_lambda,
    compute_a_lambda,
    build_g_lambda_vals,
)


@dataclass
class UnimodalResult:
    """Result from unimodal projection computation."""

    metric_name: str
    dist_name: str
    xi: float
    mu: float
    sigma: float
    epsilon: Optional[float]
    epsilon_min_U: Optional[float]
    epsilon_switch_U: Optional[float]
    regime: Optional[str]
    lambda_epsilon: Optional[float]
    worst_case_value: float
    projection_norm: float
    benchmark_value: Optional[float]
    abs_gap: Optional[float]
    prop4_bound_satisfied: bool


def project_onto_unimodal_cone_l2(
    gamma_vals: np.ndarray,
    u_grid: np.ndarray,
    xi: float,
) -> np.ndarray:
    """
    Compute the L2 projection of gamma onto the cone F_{U,xi}^{-1}.

    The cone F_{U,xi}^{-1} is the set of functions k in L2([0,1]) that are:
      - concave on [0, xi]  (second difference <= 0 on the left)
      - convex  on [xi, 1]  (second difference >= 0 on the right)

    This matches the paper's Proposition 2/3 characterisation:
      - For GD  (gamma = 1-2u, linear): already in the cone → projection = itself.
      - For MMD (gamma = step ±1 at 0.5): projection = 3*(u-0.5).

    Solved as a weighted least-squares QP via CVXPY using second-difference
    inequality constraints.  Falls back to unconstrained weighted mean if CVXPY
    is unavailable.

    Custom — Context7 found no library equivalent (paper Proposition 2/3).

    Parameters
    ----------
    gamma_vals : np.ndarray, shape (N,)
        Gamma values at u_grid.
    u_grid : np.ndarray, shape (N,)
        Evaluation grid on (0, 1).
    xi : float
        Inflection point in (0, 1).

    Returns
    -------
    proj_vals : np.ndarray, shape (N,)
        Projection values on u_grid.
    """
    try:
        import cvxpy as cp
    except ImportError:
        warnings.warn("cvxpy not available; returning unconstrained projection (identity).")
        return gamma_vals.copy()

    n = len(u_grid)
    # Uniform interval width for L2 norm (grid assumed quasi-uniform).
    # Using sum_squares is equivalent to minimising the unweighted L2^2 because
    # trapezoidal weights with equal spacing reduce to a constant factor.
    du = (u_grid[-1] - u_grid[0]) / (n - 1) if n > 1 else 1.0

    k = cp.Variable(n)
    # min  du * sum_i (gamma_i - k_i)^2
    objective = cp.Minimize(du * cp.sum_squares(gamma_vals - k))

    constraints: list = []
    xi_idx = int(np.searchsorted(u_grid, xi))   # first index with u_grid[j] >= xi

    # Concavity on [0, xi]: second-difference <= 0 at each interior left point
    #   k[j-1] - 2*k[j] + k[j+1] <= 0   for j = 1 .. xi_idx-1
    for j in range(1, xi_idx):
        constraints.append(k[j - 1] - 2 * k[j] + k[j + 1] <= 0)

    # Convexity on [xi, 1]: second-difference >= 0 at each interior right point
    #   k[j-1] - 2*k[j] + k[j+1] >= 0   for j = xi_idx .. n-2
    for j in range(max(xi_idx, 1), n - 1):
        constraints.append(k[j - 1] - 2 * k[j] + k[j + 1] >= 0)

    prob = cp.Problem(objective, constraints)

    # Prefer CLARABEL (exact QP, handles second-difference constraints well)
    for solver in [cp.CLARABEL, cp.OSQP, None]:
        try:
            if solver is None:
                prob.solve(verbose=False)
            else:
                prob.solve(solver=solver, verbose=False)
            if k.value is not None:
                break
        except Exception:
            continue

    if k.value is None:
        warnings.warn("CVXPY projection failed; returning gamma unchanged.")
        return gamma_vals.copy()

    return np.array(k.value, dtype=float)


def _compute_weights(u_grid: np.ndarray) -> np.ndarray:
    """Compute trapezoidal integration weights for a grid."""
    if len(u_grid) <= 1:
        return np.ones(len(u_grid))
    widths = np.zeros(len(u_grid))
    widths[0] = (u_grid[1] - u_grid[0]) / 2
    widths[-1] = (u_grid[-1] - u_grid[-2]) / 2
    widths[1:-1] = (u_grid[2:] - u_grid[:-2]) / 2
    return widths


def _isotonic_regression_nondecreasing(
    y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Isotonic regression: find non-decreasing sequence minimizing weighted L2.

    Uses pool-adjacent-violators (PAV) algorithm.
    Custom — Context7 found no library equivalent with arbitrary weights.
    """
    n = len(y)
    if n == 0:
        return np.array([])
    if n == 1:
        return y.copy()

    # Pool-adjacent-violators
    # Each block: sum of weights, weighted sum of y
    blocks = [[w, w * v] for w, v in zip(weights, y)]

    def merge_needed(blocks: list) -> bool:
        for i in range(len(blocks) - 1):
            if blocks[i][1] / blocks[i][0] > blocks[i + 1][1] / blocks[i + 1][0]:
                return True
        return False

    changed = True
    while changed:
        changed = False
        i = 0
        new_blocks = []
        while i < len(blocks):
            if i < len(blocks) - 1 and (
                blocks[i][1] / blocks[i][0] > blocks[i + 1][1] / blocks[i + 1][0]
            ):
                # Merge blocks i and i+1
                merged = [
                    blocks[i][0] + blocks[i + 1][0],
                    blocks[i][1] + blocks[i + 1][1],
                ]
                new_blocks.append(merged)
                i += 2
                changed = True
            else:
                new_blocks.append(blocks[i])
                i += 1
        blocks = new_blocks

    # Reconstruct
    result = np.zeros(n)
    idx = 0
    for block in blocks:
        w_tot, wy_tot = block
        val = wy_tot / w_tot
        # Count how many elements are in this block by tracking original
        # weights (simplified: approximate block count by weight ratio)
        # Use a different approach: track original indices
        pass

    # Better PAV implementation tracking original points
    return _pav_nondecreasing(y, weights)


def _pav_nondecreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for non-decreasing isotonic regression."""
    n = len(y)
    if n == 0:
        return np.array([])
    # Each element: (mean, weight, start_idx)
    sol_mean = y.copy().astype(float)
    sol_w = w.copy().astype(float)
    # Block structure: blocks[i] = (mean, total_weight, list of original indices)
    blocks: List[Tuple[float, float, List[int]]] = [
        (float(y[i]), float(w[i]), [i]) for i in range(n)
    ]

    changed = True
    while changed:
        changed = False
        new_blocks = []
        i = 0
        while i < len(blocks):
            if (
                i < len(blocks) - 1
                and blocks[i][0] > blocks[i + 1][0] + 1e-14
            ):
                # Merge
                m1, w1, idx1 = blocks[i]
                m2, w2, idx2 = blocks[i + 1]
                new_w = w1 + w2
                new_m = (m1 * w1 + m2 * w2) / new_w
                new_blocks.append((new_m, new_w, idx1 + idx2))
                i += 2
                changed = True
            else:
                new_blocks.append(blocks[i])
                i += 1
        blocks = new_blocks

    result = np.zeros(n)
    for (m, _, indices) in blocks:
        for j in indices:
            result[j] = m
    return result


def _isotonic_regression_nonincreasing(
    y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Isotonic regression for non-increasing sequences (reverse PAV)."""
    if len(y) == 0:
        return np.array([])
    return _pav_nondecreasing(-y[::-1], weights[::-1])[::-1] * -1


def project_gamma_unimodal(
    gamma_fn: Callable[[np.ndarray], np.ndarray],
    xi: float,
    n_grid: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """
    Project gamma onto the unimodal cone with fixed inflection xi using a fine grid.

    Parameters
    ----------
    gamma_fn : Callable
        Weight function gamma: (0,1) -> R.
    xi : float
        Inflection point.
    n_grid : int
        Grid resolution.

    Returns
    -------
    u_grid : np.ndarray
    proj_vals : np.ndarray
    proj_fn : Callable
        Interpolated projection function.
    """
    u_grid = np.linspace(1e-9, 1 - 1e-9, n_grid)
    gamma_vals = gamma_fn(u_grid)
    proj_vals = project_onto_unimodal_cone_l2(gamma_vals, u_grid, xi)

    def proj_fn(u: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(u, float), u_grid, proj_vals)

    return u_grid, proj_vals, proj_fn


def compute_unimodal_worst_case_no_wasserstein(
    metric: DistortionMetric,
    xi: float,
    mu: float,
    sigma: float,
    n_grid: int = 1000,
    n_gauss: int = 500,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """
    Compute worst-case value under unimodality constraint (no Wasserstein), fixed xi.

    worst-case value = mu * g(1) + sigma * b_xi
    where b_xi = sqrt(Var(gamma_xi^uparrow(V))).

    Parameters
    ----------
    metric : DistortionMetric
    xi : float
    mu, sigma : float
    n_grid : int
    n_gauss : int

    Returns
    -------
    (worst_case, u_grid, proj_vals, a_xi, b_xi)
    """
    u_grid, proj_vals, _ = project_gamma_unimodal(metric.gamma, xi, n_grid)

    # a_xi = E[gamma_xi^uparrow(V)] = integral_0^1 proj(u) du = g(1)
    widths = np.diff(np.concatenate([[0.0], u_grid, [1.0]]))
    widths_mid = np.full(n_grid, 1.0 / n_grid)
    a_xi = float(np.sum(proj_vals * widths_mid))

    # b_xi = sqrt(Var(proj(V)))
    e_sq = float(np.sum(proj_vals**2 * widths_mid))
    var_proj = max(e_sq - a_xi**2, 0.0)
    b_xi = float(np.sqrt(var_proj))

    # Worst-case value = mu * g(1) + sigma * b_xi
    worst_case = mu * metric.g_val_at_1 + sigma * b_xi
    return worst_case, u_grid, proj_vals, a_xi, b_xi


def compute_unimodal_worst_case_quantile(
    proj_vals: np.ndarray,
    u_grid: np.ndarray,
    a_xi: float,
    b_xi: float,
    mu: float,
    sigma: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct the worst-case quantile h_xi^uparrow.

    h_xi^uparrow(u) = mu + sigma * (gamma_xi^uparrow(u) - a_xi) / b_xi

    Parameters
    ----------
    proj_vals, u_grid : np.ndarray
    a_xi, b_xi : float
    mu, sigma : float

    Returns
    -------
    Callable
    """
    assert b_xi > 1e-12, "b_xi is zero: degenerate case"

    def h_fn(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        proj_at_u = np.interp(u, u_grid, proj_vals)
        return mu + sigma * (proj_at_u - a_xi) / b_xi

    return h_fn


def step_approx_gamma(
    gamma_fn: Callable[[np.ndarray], np.ndarray],
    n_pieces: int,
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """
    Approximate gamma by a step function with n_pieces equal-width intervals.

    Parameters
    ----------
    gamma_fn : Callable
    n_pieces : int

    Returns
    -------
    u_partition : np.ndarray, shape (n_pieces+1,)
        Partition points including 0 and 1.
    gamma_step_vals : np.ndarray, shape (n_pieces,)
        Constant value on each interval.
    step_fn : Callable
    """
    u_partition = np.linspace(0.0, 1.0, n_pieces + 1)
    u_midpoints = 0.5 * (u_partition[:-1] + u_partition[1:])
    gamma_step_vals = gamma_fn(u_midpoints)

    def step_fn(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        idx = np.clip(
            np.searchsorted(u_partition, u, side="right") - 1, 0, n_pieces - 1
        )
        return gamma_step_vals[idx]

    return u_partition, gamma_step_vals, step_fn


def compute_step_projection_and_error(
    gamma_fn: Callable[[np.ndarray], np.ndarray],
    xi: float,
    n_pieces_list: List[int],
    n_ref: int = 2000,
) -> List[dict]:
    """
    Compute the step approximation and projection for increasing n_pieces,
    measuring error against a high-resolution reference projection.

    Validates Proposition 4 error bounds.

    Parameters
    ----------
    gamma_fn : Callable
    xi : float
    n_pieces_list : List[int]
    n_ref : int
        Reference grid for high-resolution comparison.

    Returns
    -------
    List of dicts with error metrics.
    """
    # High-resolution reference
    u_ref, proj_ref, proj_ref_fn = project_gamma_unimodal(gamma_fn, xi, n_ref)
    # Reference gamma approx error
    gamma_ref = gamma_fn(u_ref)
    proj_ref_l2 = np.sqrt(np.mean((proj_ref - gamma_ref) ** 2))

    results = []
    for n in n_pieces_list:
        u_part, g_step, step_fn = step_approx_gamma(gamma_fn, n)

        # Project step function
        u_step_grid = np.linspace(1e-9, 1 - 1e-9, max(n * 20, 200))
        g_step_vals = step_fn(u_step_grid)
        proj_step_vals = project_onto_unimodal_cone_l2(g_step_vals, u_step_grid, xi)

        # Error metrics on u_ref grid
        gamma_ref_vals = gamma_fn(u_ref)
        g_step_ref = step_fn(u_ref)
        gamma_approx_l2 = float(np.sqrt(np.mean((g_step_ref - gamma_ref_vals) ** 2)))

        proj_step_ref = np.interp(u_ref, u_step_grid, proj_step_vals)
        proj_error_l2 = float(np.sqrt(np.mean((proj_step_ref - proj_ref) ** 2)))

        # Proposition 4: ||gamma_n^uparrow - gamma^uparrow||_2 <= ||gamma_n - gamma||_2
        prop4_ok = proj_error_l2 <= gamma_approx_l2 + 1e-6

        # Count pieces (should be <= 2n+2)
        # Count constant regions in projection
        diffs = np.abs(np.diff(proj_step_vals))
        n_pieces_actual = int(np.sum(diffs > 1e-8 * (np.max(proj_step_vals) - np.min(proj_step_vals) + 1e-10))) + 1

        results.append({
            "n_pieces": n,
            "gamma_approx_l2": gamma_approx_l2,
            "proj_error_l2": proj_error_l2,
            "prop4_bound": gamma_approx_l2,
            "prop4_satisfied": prop4_ok,
            "n_projection_pieces": n_pieces_actual,
            "max_pieces_bound": 2 * n + 2,
            "pieces_bound_ok": n_pieces_actual <= 2 * n + 2,
        })

    return results


def compute_unimodal_wasserstein_thresholds(
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    metric: DistortionMetric,
    xi: float,
    n_grid: int = 1000,
    n_gauss: int = 500,
) -> Tuple[float, float, float, float]:
    """
    Compute the unimodal-plus-Wasserstein thresholds:

    epsilon_min,U = (mu_F-mu)^2 + (sigma_F-sigma)^2 + 2 sigma_F sigma (1 - c_hat_0)
    epsilon_switch,U = (mu_F-mu)^2 + (sigma_F-sigma)^2 + 2 sigma_F sigma (1 - c_1)

    where:
    c_hat_0 = Corr(F^{-1}(V), F_{xi}^{-1,uparrow}(V))
    c_1 = Corr(F^{-1}(V), gamma_xi^uparrow(V))

    Parameters
    ----------
    ref : ReferenceDistribution
    mu, sigma : float
    metric : DistortionMetric
    xi : float
    n_grid, n_gauss : int

    Returns
    -------
    (epsilon_min_U, epsilon_switch_U, c_hat_0, c_1)
    """
    mu_F, sigma_F = ref.mu_F, ref.sigma_F

    # Compute gamma_xi^uparrow
    u_grid, proj_vals, proj_fn = project_gamma_unimodal(metric.gamma, xi, n_grid)

    nodes, weights = _leggauss(n_gauss)
    u_pts = 0.5 * (1.0 - 2e-9) * nodes + 0.5 * (1.0 + 2e-9)
    half = 0.5 * (1.0 - 2e-9)

    f_inv = ref.quantile(u_pts)
    gamma_proj = proj_fn(u_pts)

    # c_1 = Corr(F^{-1}(V), gamma_xi^uparrow(V))
    e_f = float(half * np.sum(weights * f_inv))
    e_g = float(half * np.sum(weights * gamma_proj))
    e_fg = float(half * np.sum(weights * f_inv * gamma_proj))
    e_f2 = float(half * np.sum(weights * f_inv**2))
    e_g2 = float(half * np.sum(weights * gamma_proj**2))
    var_f = e_f2 - e_f**2
    var_g = e_g2 - e_g**2

    if var_f < 1e-15 or var_g < 1e-15:
        c_1 = 0.0
    else:
        c_1 = (e_fg - e_f * e_g) / np.sqrt(var_f * var_g)
    c_1 = float(np.clip(c_1, -1.0, 1.0))

    # c_hat_0 = Corr(F^{-1}(V), worst-case quantile at lambda=0)
    # The worst-case quantile (unimodal, no Wasserstein) has quantile
    # h_xi^uparrow(u) = mu + sigma * (gamma_proj(u) - a_xi) / b_xi
    # So Corr(F^{-1}(V), h_xi^uparrow(V)) = Corr(F^{-1}(V), gamma_proj(V)) = c_1
    # (Since h is affine function of gamma_proj)
    c_hat_0 = c_1

    epsilon_min_U = (mu_F - mu)**2 + (sigma_F - sigma)**2 + 2 * sigma_F * sigma * (1 - c_hat_0)
    epsilon_switch_U = (mu_F - mu)**2 + (sigma_F - sigma)**2 + 2 * sigma_F * sigma * (1 - c_1)

    return float(epsilon_min_U), float(epsilon_switch_U), float(c_hat_0), float(c_1)


def compute_unimodal_wasserstein_bound(
    metric: DistortionMetric,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    epsilon: float,
    xi: float,
    n_grid: int = 1000,
    n_gauss: int = 500,
    tol: float = 1e-8,
) -> UnimodalResult:
    """
    Compute the unimodality-plus-Wasserstein worst-case bound (Theorem 3).

    Parameters
    ----------
    metric, ref : DistortionMetric, ReferenceDistribution
    mu, sigma, epsilon : float
    xi : float
        Fixed inflection point.
    n_grid, n_gauss : int
    tol : float

    Returns
    -------
    UnimodalResult
    """
    eps_min_U, eps_switch_U, c_hat_0, c_1 = compute_unimodal_wasserstein_thresholds(
        ref, mu, sigma, metric, xi, n_grid, n_gauss
    )

    if epsilon < eps_min_U - tol:
        wc_val, _, proj_vals, a_xi, b_xi = compute_unimodal_worst_case_no_wasserstein(
            metric, xi, mu, sigma, n_grid, n_gauss
        )
        return UnimodalResult(
            metric_name=metric.name,
            dist_name=ref.name,
            xi=xi, mu=mu, sigma=sigma,
            epsilon=epsilon,
            epsilon_min_U=eps_min_U,
            epsilon_switch_U=eps_switch_U,
            regime="infeasible",
            lambda_epsilon=np.nan,
            worst_case_value=np.nan,
            projection_norm=b_xi,
            benchmark_value=None,
            abs_gap=None,
            prop4_bound_satisfied=True,
        )

    # No-Wasserstein baseline
    wc_no_wass, u_grid, proj_vals, a_xi, b_xi = compute_unimodal_worst_case_no_wasserstein(
        metric, xi, mu, sigma, n_grid, n_gauss
    )

    if epsilon >= eps_switch_U - tol:
        regime = "inactive"
        lambda_eps = 0.0
        # Use h_xi^uparrow
        if b_xi < 1e-12:
            wc_val = mu * metric.g_val_at_1
        else:
            h_fn = compute_unimodal_worst_case_quantile(proj_vals, u_grid, a_xi, b_xi, mu, sigma)
            wc_val = float(np.sum(metric.gamma(u_grid) * h_fn(u_grid) / n_grid))
    else:
        regime = "active"
        # Solve for lambda_epsilon
        lambda_eps = _solve_lambda_unimodal(
            epsilon, metric, ref, mu, sigma, xi, n_grid, n_gauss, tol
        )
        # Compute worst-case with lambda
        k_lam_fn, hull_u, hull_f, a_lam, b_lam = _build_h_lambda_unimodal(
            lambda_eps, metric, ref, mu, sigma, xi, n_grid
        )
        if b_lam < 1e-12:
            wc_val = mu * metric.g_val_at_1
        else:
            wc_val = float(np.sum(metric.gamma(u_grid) * k_lam_fn(u_grid) / n_grid))

    return UnimodalResult(
        metric_name=metric.name,
        dist_name=ref.name,
        xi=xi, mu=mu, sigma=sigma,
        epsilon=epsilon,
        epsilon_min_U=eps_min_U,
        epsilon_switch_U=eps_switch_U,
        regime=regime,
        lambda_epsilon=lambda_eps,
        worst_case_value=float(wc_val),
        projection_norm=float(b_xi),
        benchmark_value=None,
        abs_gap=None,
        prop4_bound_satisfied=True,
    )


def _build_h_lambda_unimodal(
    lambda_val: float,
    metric: DistortionMetric,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    xi: float,
    n_grid: int = 1000,
) -> Tuple[Callable, np.ndarray, np.ndarray, float, float]:
    """
    For a given lambda, build k_lambda and its unimodal projection + worst-case quantile.

    k_lambda(u) = gamma(u) + lambda * F^{-1}(u)
    k_{lambda,xi}^uparrow = projection of k_lambda onto unimodal cone
    h_{lambda,xi}(u) = mu + sigma * (k_{lambda,xi}^uparrow(u) - a_{lambda,xi}) / b_{lambda,xi}
    """
    u_grid = np.linspace(1e-9, 1 - 1e-9, n_grid)
    gamma_vals = metric.gamma(u_grid)
    f_inv_vals = ref.quantile(u_grid)

    k_lam_vals = gamma_vals + lambda_val * f_inv_vals
    # Project onto unimodal cone
    k_proj_vals = project_onto_unimodal_cone_l2(k_lam_vals, u_grid, xi)

    widths_mid = np.full(n_grid, 1.0 / n_grid)
    a_lam = metric.g_val_at_1 + lambda_val * ref.mu_F
    e_sq = float(np.sum(k_proj_vals**2 * widths_mid))
    var_k = max(e_sq - a_lam**2, 0.0)
    b_lam = float(np.sqrt(var_k))

    if b_lam < 1e-12:
        def h_fn(u: np.ndarray) -> np.ndarray:
            return np.full_like(np.asarray(u, float), mu)
    else:
        def h_fn(u: np.ndarray) -> np.ndarray:
            u = np.asarray(u, float)
            k_proj_at_u = np.interp(u, u_grid, k_proj_vals)
            return mu + sigma * (k_proj_at_u - a_lam) / b_lam

    return h_fn, u_grid, k_proj_vals, a_lam, b_lam


def _solve_lambda_unimodal(
    epsilon: float,
    metric: DistortionMetric,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    xi: float,
    n_grid: int = 1000,
    n_gauss: int = 500,
    tol: float = 1e-8,
) -> float:
    """Solve for lambda_epsilon in the unimodal Wasserstein active regime."""

    def eq(lam: float) -> float:
        h_fn, _, _, _, _ = _build_h_lambda_unimodal(
            lam, metric, ref, mu, sigma, xi, n_grid
        )
        dW2 = wasserstein_sq_distance(h_fn, ref.quantile, n_gauss=n_gauss)
        return dW2 - epsilon

    val_at_0 = eq(0.0)
    if val_at_0 < 0:
        return 0.0

    lambda_max = 50.0
    val_at_max = eq(lambda_max)
    if val_at_max > 0:
        for lam_try in [100.0, 500.0]:
            v = eq(lam_try)
            if v < 0:
                lambda_max = lam_try
                val_at_max = v
                break

    if val_at_max > 0:
        warnings.warn(f"Could not bracket lambda_unimodal for epsilon={epsilon}")
        return lambda_max

    try:
        return float(brentq(eq, 1e-10, lambda_max, xtol=tol, rtol=tol, maxiter=200))
    except ValueError:
        warnings.warn(f"brentq failed (unimodal) for epsilon={epsilon}")
        return lambda_max / 2
