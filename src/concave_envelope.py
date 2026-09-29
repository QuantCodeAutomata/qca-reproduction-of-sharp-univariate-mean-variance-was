"""
Concave envelope (least concave majorant) computation.

The least concave majorant (LCM) of a function f on [0,1] is the smallest
concave function g with g >= f everywhere. It is computed via the upper
convex hull of the (u, f(u)) point set.

Custom implementation — Context7 found no library equivalent.
Paper references: Sections involving g^* = concave envelope of g,
and g_lambda^* = concave envelope of g_lambda(t) = g(t) + lambda * integral_{1-t}^1 F^{-1}(s) ds.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Callable, Tuple
import numpy as np


@lru_cache(maxsize=16)
def _leggauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre nodes and weights for n points."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def least_concave_majorant(
    u_grid: np.ndarray,
    f_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the least concave majorant (LCM) of a function on a grid.

    Uses upper convex hull on the point set {(u_i, f_i)}.

    Parameters
    ----------
    u_grid : np.ndarray, shape (N,)
        Grid points in [0,1], sorted ascending.
    f_vals : np.ndarray, shape (N,)
        Function values at grid points.

    Returns
    -------
    hull_u : np.ndarray
        u-coordinates of the concave hull vertices (including endpoints).
    hull_f : np.ndarray
        f-values at hull vertices.
    lcm_vals : np.ndarray, shape (N,)
        LCM evaluated at all u_grid points by linear interpolation.
    """
    u_grid = np.asarray(u_grid, float)
    f_vals = np.asarray(f_vals, float)
    n = len(u_grid)
    assert n >= 2

    # Upper convex hull: Graham scan variant for concavity
    # A function is concave iff its epigraph (from above) is convex.
    # Equivalently: keep points where slopes are non-increasing.
    hull_idx = [0]
    for i in range(1, n):
        # Add point i
        while len(hull_idx) >= 2:
            a, b = hull_idx[-2], hull_idx[-1]
            # Check if b is below line from a to i (then remove b)
            slope_ai = (f_vals[i] - f_vals[a]) / (u_grid[i] - u_grid[a])
            slope_ab = (f_vals[b] - f_vals[a]) / (u_grid[b] - u_grid[a])
            if slope_ab <= slope_ai:
                hull_idx.pop()
            else:
                break
        hull_idx.append(i)

    hull_idx = np.array(hull_idx)
    hull_u = u_grid[hull_idx]
    hull_f = f_vals[hull_idx]

    # Interpolate LCM at all grid points
    lcm_vals = np.interp(u_grid, hull_u, hull_f)
    return hull_u, hull_f, lcm_vals


def lcm_derivative(
    hull_u: np.ndarray,
    hull_f: np.ndarray,
    u_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the derivative (slope) of the piecewise-linear LCM at each grid point.

    The LCM is piecewise linear between hull vertices, so the derivative is
    piecewise constant on each interval between consecutive hull points.

    At a grid point u, the derivative is the slope of the hull segment
    containing u.

    Parameters
    ----------
    hull_u : np.ndarray
        u-coordinates of hull vertices.
    hull_f : np.ndarray
        f-values at hull vertices.
    u_grid : np.ndarray
        Grid points where derivative is evaluated.

    Returns
    -------
    deriv : np.ndarray, shape as u_grid
        Piecewise-constant derivative values.
    """
    u_grid = np.asarray(u_grid, float)
    n_hull = len(hull_u)
    assert n_hull >= 2

    slopes = np.diff(hull_f) / np.diff(hull_u)

    # For each u in u_grid, find which segment it falls in
    # Segment i: [hull_u[i], hull_u[i+1])
    seg_idx = np.searchsorted(hull_u, u_grid, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(slopes) - 1)

    return slopes[seg_idx]


def build_g_lambda_vals(
    u_grid: np.ndarray,
    g_vals: np.ndarray,
    lambda_val: float,
    ref_quantile_fn: Callable[[np.ndarray], np.ndarray],
    ref_mu_F: float,
) -> np.ndarray:
    """
    Compute values of g_lambda(t) = g(t) + lambda * integral_{1-t}^1 F^{-1}(s) ds.

    Note: d/dt [integral_{1-t}^1 F^{-1}(s)ds] = F^{-1}(1-(1-t)) = F^{-1}(t).
    So g_lambda'(t) = g'(t) + lambda * F^{-1}(t).

    The integral integral_{1-t}^1 F^{-1}(s)ds = mu_F - integral_0^{1-t} F^{-1}(s)ds
    = mu_F - E[F^{-1}(U) * 1_{U <= 1-t}].

    We compute this at each grid point t using numerical integration.

    Parameters
    ----------
    u_grid : np.ndarray
        Grid of t values in [0,1].
    g_vals : np.ndarray
        g(t) values at u_grid.
    lambda_val : float
        Lagrange multiplier >= 0.
    ref_quantile_fn : Callable
        F^{-1}: (0,1) -> R.
    ref_mu_F : float
        Mean of reference distribution (= integral_0^1 F^{-1}(u) du).

    Returns
    -------
    g_lam_vals : np.ndarray
        g_lambda(t) at each t in u_grid.
    """
    if abs(lambda_val) < 1e-15:
        return g_vals.copy()

    # Compute integral_{1-t}^1 F^{-1}(s) ds for each t
    # = mu_F - integral_0^{1-t} F^{-1}(s) ds
    t_vals = u_grid

    # Use cumulative trapezoid for integral_0^{1-t} F^{-1}(s) ds
    # Evaluate at a fine grid, then interpolate to 1-t
    fine = np.linspace(1e-12, 1 - 1e-12, 4000)
    f_fine = ref_quantile_fn(fine)
    # Cumulative integral from 0 to each u
    cum_int = np.zeros(len(fine))
    cum_int[1:] = np.cumsum(np.diff(fine) * 0.5 * (f_fine[:-1] + f_fine[1:]))
    # integral_0^{1-t} F^{-1}(s)ds:
    s_pts = np.clip(1.0 - t_vals, 1e-12, 1 - 1e-12)
    int_0_to_1mt = np.interp(s_pts, fine, cum_int)
    int_1mt_to_1 = ref_mu_F - int_0_to_1mt

    g_lam_vals = g_vals + lambda_val * int_1mt_to_1
    return g_lam_vals


def compute_a_lambda(
    g_lam_star_deriv_at_0: float,
    g_val_at_1: float,
    lambda_val: float,
    mu_F: float,
) -> float:
    """
    Compute a_lambda = g(1) + lambda * mu_F.

    This is E[g_lambda^{*}'(1-V)] where V~U[0,1], equals the mean of the
    derivative field. For the concave envelope g_lambda^*, this simplifies
    to g(1) + lambda * mu_F.

    Parameters
    ----------
    g_lam_star_deriv_at_0 : float
        Not used in this formula (kept for interface compatibility).
    g_val_at_1 : float
        g(1) from the distortion metric.
    lambda_val : float
        Lagrange multiplier.
    mu_F : float
        Mean of reference distribution.

    Returns
    -------
    float
    """
    # Custom — Context7 found no library equivalent (paper notation a_lambda)
    return g_val_at_1 + lambda_val * mu_F


def compute_b_lambda(
    hull_u: np.ndarray,
    hull_f: np.ndarray,
    a_lambda: float,
    n_integration: int = 2000,
) -> float:
    """
    Compute b_lambda = sqrt(Var((g_lambda^*)'(V))) where V ~ U[0,1].

    Since (g_lambda^*)' is piecewise constant on the hull intervals,
    Var((g_lambda^*)'(V)) = E[(g_lambda^*)'(V)^2] - (E[(g_lambda^*)'(V)])^2
    = E[(g_lambda^*)'(V)^2] - a_lambda^2.

    E[(g_lambda^*)'(V)^2] = sum over segments: slope_i^2 * (width of segment i).

    Parameters
    ----------
    hull_u : np.ndarray
        Hull vertices (u-coordinates), length K.
    hull_f : np.ndarray
        Hull values (f-coordinates), length K.
    a_lambda : float
        Mean of the derivative = E[(g_lambda^*)'(V)].
    n_integration : int
        Not used (analytical computation on hull segments).

    Returns
    -------
    float
        b_lambda >= 0.
    """
    slopes = np.diff(hull_f) / np.diff(hull_u)
    widths = np.diff(hull_u)

    # E[slope^2]
    e_sq = float(np.sum(slopes**2 * widths))
    var_deriv = e_sq - a_lambda**2

    # Clip small negatives from numerical error
    var_deriv = max(var_deriv, 0.0)
    return float(np.sqrt(var_deriv))


def construct_h_lambda(
    hull_u: np.ndarray,
    hull_f: np.ndarray,
    a_lambda: float,
    b_lambda: float,
    mu: float,
    sigma: float,
    u_eval: np.ndarray,
) -> np.ndarray:
    """
    Construct the worst-case quantile h_lambda(t) evaluated at u_eval.

    h_lambda(t) = mu + sigma * ((g_lambda^*)'(1-t) - a_lambda) / b_lambda.

    The derivative of g_lambda^* at (1-t) is the piecewise-constant slope
    of the LCM at position (1-t).

    Parameters
    ----------
    hull_u, hull_f : np.ndarray
        LCM hull vertices.
    a_lambda, b_lambda : float
        Shift and scale parameters.
    mu, sigma : float
        Target mean and standard deviation.
    u_eval : np.ndarray
        t values in [0,1] at which h_lambda is evaluated.

    Returns
    -------
    h_vals : np.ndarray
    """
    assert b_lambda > 1e-15, "b_lambda is zero: degenerate case"
    # Derivative of LCM at 1-t
    one_minus_t = np.clip(1.0 - u_eval, 0.0, 1.0)
    deriv_at_1mt = lcm_derivative(hull_u, hull_f, one_minus_t)
    h_vals = mu + sigma * (deriv_at_1mt - a_lambda) / b_lambda
    return h_vals


def _make_g_grid_local(
    gamma_fn: Callable[[np.ndarray], np.ndarray],
    g_val_at_1: float,
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a distortion function g on a uniform grid from its weight function gamma.

    g(t) = integral_0^t gamma(1-s) ds  (for absolutely continuous distortions).
    Rescaled so g(1) matches the provided g_val_at_1.

    Parameters
    ----------
    gamma_fn : Callable
        Weight function gamma: (0,1) -> R.
    g_val_at_1 : float
        Known value of g at 1.
    n : int
        Number of interior grid points.

    Returns
    -------
    (u_grid, g_vals) : Tuple[np.ndarray, np.ndarray]
    """
    u_grid = np.linspace(0.0, 1.0, n + 2)

    def g_prime(t: np.ndarray) -> np.ndarray:
        return gamma_fn(1.0 - t)

    g_vals = np.zeros_like(u_grid)
    for i in range(1, len(u_grid)):
        g_vals[i] = np.trapz(g_prime(u_grid[:i+1]), u_grid[:i+1])

    # Rescale to match g(1) = g_val_at_1 if numeric value differs
    if abs(g_vals[-1]) > 1e-10:
        g_vals = g_vals * (g_val_at_1 / g_vals[-1])
    elif abs(g_val_at_1) < 1e-10:
        pass  # both zero, fine
    # else leave as is

    return u_grid, g_vals


def wasserstein_sq_distance(
    h_lambda_fn: Callable[[np.ndarray], np.ndarray],
    ref_quantile_fn: Callable[[np.ndarray], np.ndarray],
    n_gauss: int = 500,
) -> float:
    """
    Compute d_W^2(F, H_lambda) = integral_0^1 |F^{-1}(u) - H_lambda^{-1}(u)|^2 du.

    Parameters
    ----------
    h_lambda_fn : Callable
        Quantile function of H_lambda.
    ref_quantile_fn : Callable
        Quantile function of reference F.
    n_gauss : int
        Gauss-Legendre nodes.

    Returns
    -------
    float
    """
    nodes, weights = _leggauss(n_gauss)
    eps = 1e-9

    def integrand(u: np.ndarray) -> np.ndarray:
        f_inv = ref_quantile_fn(u)
        h_inv = h_lambda_fn(u)
        return (f_inv - h_inv) ** 2

    # Main interval + endpoint corrections
    def gl_on(a: float, b: float, nn: int = n_gauss) -> float:
        nd, wt = _leggauss(nn)
        u_pts = 0.5 * (b - a) * nd + 0.5 * (b + a)
        return float(0.5 * (b - a) * np.sum(wt * integrand(u_pts)))

    val = gl_on(eps, 1.0 - eps)
    val += gl_on(1e-14, eps, nn=100)
    val += gl_on(1.0 - eps, 1.0 - 1e-14, nn=100)
    return val
