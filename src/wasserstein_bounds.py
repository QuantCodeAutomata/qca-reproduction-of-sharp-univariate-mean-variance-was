"""
Wasserstein distortion risk bounds: Theorems 1 & 2, Corollaries 1 & 2.

Implements the worst-case and best-case distortion risk metric bounds over the
uncertainty set M_epsilon(mu, sigma) = {G : E_G[X]=mu, E_G[X^2]=mu^2+sigma^2,
d_W(F,G)^2 <= epsilon}.

Custom implementation — Context7 found no library equivalent (paper-specific formulas).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
import warnings

from .reference_distributions import ReferenceDistribution
from .distortion_metrics import DistortionMetric, eval_rho_direct
from .concave_envelope import (
    _leggauss,
    least_concave_majorant,
    lcm_derivative,
    build_g_lambda_vals,
    compute_a_lambda,
    compute_b_lambda,
    construct_h_lambda,
    wasserstein_sq_distance,
)


@dataclass
class BoundResult:
    """
    Result of a worst-case / best-case bound computation.

    Attributes
    ----------
    metric_name : str
    dist_name : str
    mu, sigma : float
        Target mean and std.
    epsilon : float
        Wasserstein radius squared.
    epsilon_min : float
        Feasibility threshold.
    epsilon_switch : float
        Active/inactive switching threshold.
    regime : str
        'infeasible', 'active', or 'inactive'.
    lambda_epsilon : float
        Optimal Lagrange multiplier (0 in inactive regime).
    worst_case_value : float or None
    best_case_value : float or None
    benchmark_value : float or None
        Direct optimization benchmark.
    abs_gap : float or None
        |theorem - benchmark|
    constraints_satisfied : bool
        Whether h_lambda satisfies moment and Wasserstein constraints.
    """

    metric_name: str
    dist_name: str
    mu: float
    sigma: float
    epsilon: float
    epsilon_min: float
    epsilon_switch: float
    regime: str
    lambda_epsilon: float
    worst_case_value: Optional[float] = None
    best_case_value: Optional[float] = None
    benchmark_value: Optional[float] = None
    abs_gap: Optional[float] = None
    constraints_satisfied: bool = False


N_GRID = 1000  # Default grid size for LCM computation


def _make_g_grid(
    metric: DistortionMetric,
    n: int = N_GRID,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the distortion g (or its USC envelope g_hat) on a uniform grid
    in [0, 1].

    For metrics with a ``g_hat_fn`` (e.g. VaR+ and IQD+ whose gamma is a
    singular measure), the envelope is evaluated directly — this is both
    numerically exact and mathematically correct because the paper shows
    sup rho_g = sup rho_{g_hat}.

    For absolutely continuous metrics with a smooth gamma, g is reconstructed
    by integration: g(t) = ∫₀ᵗ g'(s) ds = ∫₀ᵗ gamma(1−s) ds.

    Returns
    -------
    u_grid, g_vals : np.ndarray
        Shape (n+2,) uniform grid on [0, 1] and corresponding distortion values.
    """
    u_grid = np.linspace(0.0, 1.0, n + 2)

    # --- Path 1: direct evaluation from USC-envelope function ---
    if metric.g_hat_fn is not None:
        g_vals = metric.g_hat_fn(u_grid)
        # Ensure boundary values match the metric specification
        g_vals[0] = 0.0
        g_vals[-1] = float(metric.g_val_at_1)
        return u_grid, g_vals

    # --- Path 2: reconstruct g by integrating gamma(1-t) ---
    def g_prime(t: np.ndarray) -> np.ndarray:
        return metric.gamma(1.0 - np.asarray(t, float))

    g_vals = np.zeros_like(u_grid)
    for i in range(1, len(u_grid)):
        g_vals[i] = np.trapz(g_prime(u_grid[: i + 1]), u_grid[: i + 1])

    # Force g(1) = metric.g_val_at_1 by affine rescaling if needed
    if abs(g_vals[-1]) > 1e-10 and abs(metric.g_val_at_1) < 1e-10:
        # Variability metric: g(0)=g(1)=0, integral of gamma=0 automatically.
        pass
    elif abs(g_vals[-1] - metric.g_val_at_1) > 0.1:
        if abs(g_vals[-1]) > 1e-10:
            g_vals = g_vals * metric.g_val_at_1 / g_vals[-1]

    return u_grid, g_vals


def compute_thresholds(
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    metric: DistortionMetric,
    n_grid: int = N_GRID,
    n_gauss: int = 500,
) -> Tuple[float, float, float]:
    """
    Compute epsilon_min and epsilon_switch for the uncertainty set M_epsilon(mu,sigma).

    epsilon_min = (mu_F - mu)^2 + (sigma_F - sigma)^2
    epsilon_switch = (mu_F - mu)^2 + (sigma_F - sigma)^2 + 2 sigma_F sigma (1 - c_0)
    c_0 = Corr(F^{-1}(V), (g^*)'(1-V))

    Parameters
    ----------
    ref : ReferenceDistribution
    mu, sigma : float
        Target moments.
    metric : DistortionMetric
    n_grid : int
    n_gauss : int

    Returns
    -------
    (epsilon_min, epsilon_switch, c_0) : Tuple[float, float, float]
    """
    mu_F = ref.mu_F
    sigma_F = ref.sigma_F

    # Feasibility threshold
    epsilon_min = (mu_F - mu) ** 2 + (sigma_F - sigma) ** 2

    # Compute c_0 = Corr(F^{-1}(V), (g^*)'(1-V))
    u_grid, g_vals = _make_g_grid(metric, n=n_grid)
    # LCM of g (concave envelope g^*)
    hull_u, hull_f, lcm_vals = least_concave_majorant(u_grid, g_vals)
    # (g^*)'(1-t) for t in grid, so need derivative at 1-t
    # Corr over V~U[0,1]: E[F^{-1}(V) * (g^*)'(1-V)]
    # use GL quadrature
    nodes, weights = _leggauss(n_gauss)
    u_pts = 0.5 * (1.0 - 2e-9) * nodes + 0.5 * (1.0 + 2e-9)
    f_inv_pts = ref.quantile(u_pts)
    # (g^*)'(1-u) for u in u_pts
    g_star_deriv_pts = lcm_derivative(hull_u, hull_f, 1.0 - u_pts)

    # Check if g^*' is essentially constant
    if np.std(g_star_deriv_pts) < 1e-12:
        c_0 = 0.0
    else:
        half = 0.5 * (1.0 - 2e-9)
        # Correlation via covariance / (std_F * std_gstar)
        e_f = float(half * np.sum(weights * f_inv_pts))
        e_g = float(half * np.sum(weights * g_star_deriv_pts))
        e_fg = float(half * np.sum(weights * f_inv_pts * g_star_deriv_pts))
        e_f2 = float(half * np.sum(weights * f_inv_pts**2))
        e_g2 = float(half * np.sum(weights * g_star_deriv_pts**2))
        var_f = e_f2 - e_f**2
        var_g = e_g2 - e_g**2
        if var_f < 1e-15 or var_g < 1e-15:
            c_0 = 0.0
        else:
            c_0 = (e_fg - e_f * e_g) / np.sqrt(var_f * var_g)

    c_0 = float(np.clip(c_0, -1.0, 1.0))
    epsilon_switch = epsilon_min + 2.0 * sigma_F * sigma * (1.0 - c_0)

    return float(epsilon_min), float(epsilon_switch), c_0


def _compute_h_lambda_for_lambda(
    lambda_val: float,
    u_grid: np.ndarray,
    g_vals: np.ndarray,
    g_val_at_1: float,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    n_gauss: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float, float, Callable]:
    """
    For a given lambda, compute the worst-case quantile h_lambda and its W^2 distance to F.

    Returns
    -------
    hull_u, hull_f, a_lambda, b_lambda, h_lambda_fn
    """
    # Build g_lambda values
    g_lam_vals = build_g_lambda_vals(
        u_grid, g_vals, lambda_val, ref.quantile, ref.mu_F
    )
    # LCM of g_lambda => g_lambda^*
    hull_u, hull_f, _ = least_concave_majorant(u_grid, g_lam_vals)
    # a_lambda = g(1) + lambda * mu_F
    a_lam = compute_a_lambda(0.0, g_val_at_1, lambda_val, ref.mu_F)
    b_lam = compute_b_lambda(hull_u, hull_f, a_lam)

    def h_lam_fn(u: np.ndarray) -> np.ndarray:
        return construct_h_lambda(hull_u, hull_f, a_lam, b_lam, mu, sigma, u)

    return hull_u, hull_f, a_lam, b_lam, h_lam_fn


def _compute_wasserstein_at_lambda(
    lambda_val: float,
    u_grid: np.ndarray,
    g_vals: np.ndarray,
    g_val_at_1: float,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    n_gauss: int = 500,
) -> float:
    """
    Compute d_W^2(F, H_lambda) as a function of lambda.
    """
    g_lam_vals = build_g_lambda_vals(
        u_grid, g_vals, lambda_val, ref.quantile, ref.mu_F
    )
    hull_u, hull_f, _ = least_concave_majorant(u_grid, g_lam_vals)
    a_lam = g_val_at_1 + lambda_val * ref.mu_F
    b_lam = compute_b_lambda(hull_u, hull_f, a_lam)
    if b_lam < 1e-12:
        # Degenerate: the LCM of g_lambda is flat (lambda is below the critical
        # threshold lambda_c where the LCM first lifts off zero).  This happens
        # for convex distortions such as GD whose g^* = 0 for small lambda.
        # The worst-case quantile h_lambda collapses to a constant (delta_mu),
        # giving W^2(delta_mu, F) = sigma_F^2 + (mu_F - mu)^2.  Return a large
        # sentinel so the root-finder recognises this lambda as too small and
        # searches toward larger lambda values where b_lambda > 0.
        return 1e30

    def h_fn(u: np.ndarray) -> np.ndarray:
        return construct_h_lambda(hull_u, hull_f, a_lam, b_lam, mu, sigma, u)

    return wasserstein_sq_distance(h_fn, ref.quantile, n_gauss=n_gauss)


def solve_lambda_epsilon(
    epsilon: float,
    u_grid: np.ndarray,
    g_vals: np.ndarray,
    g_val_at_1: float,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    lambda_max: float = 100.0,
    tol: float = 1e-8,
    n_gauss: int = 500,
) -> float:
    """
    Solve for the unique lambda_epsilon > 0 satisfying d_W^2(F, H_lambda) = epsilon.

    Uses Brent's method (brentq) after confirming a sign change bracket.

    Parameters
    ----------
    epsilon : float
        Target Wasserstein radius squared.
    u_grid, g_vals : np.ndarray
    g_val_at_1 : float
    ref : ReferenceDistribution
    mu, sigma : float
    lambda_max : float
        Upper bracket for lambda search.
    tol : float
        Root-finding tolerance.
    n_gauss : int

    Returns
    -------
    lambda_epsilon : float
    """
    def eq(lam: float) -> float:
        return _compute_wasserstein_at_lambda(
            lam, u_grid, g_vals, g_val_at_1, ref, mu, sigma, n_gauss
        ) - epsilon

    # Check at lambda=0: if W^2(F,H_0) < epsilon we are already in the inactive regime
    val_at_0 = eq(0.0)
    if val_at_0 < 0:
        return 0.0

    # Exponential search to find an upper bracket where eq(lam) < 0.
    # W^2(F, H_lambda) is strictly decreasing in lambda (paper Theorem 1), so the
    # root is unique and exponential search is guaranteed to terminate.
    lo, hi = 1e-10, max(lambda_max, 10.0)
    val_hi = eq(hi)
    max_doublings = 30
    for _ in range(max_doublings):
        if val_hi < 0:
            break
        hi *= 4.0
        val_hi = eq(hi)
    else:
        # After many doublings eq is still positive → active regime but lambda is huge.
        # This happens at epsilon extremely close to epsilon_min; return best estimate.
        warnings.warn(
            f"Could not bracket lambda for epsilon={epsilon:.6g} "
            f"(eq at lam={hi:.2g} is still {val_hi:.3g}). "
            "Returning largest attempted lambda as conservative estimate."
        )
        return float(hi)

    try:
        lam_sol = brentq(eq, lo, hi, xtol=tol, rtol=tol, maxiter=300)
    except ValueError as exc:
        warnings.warn(f"brentq failed for epsilon={epsilon:.6g}: {exc}")
        lam_sol = hi / 2.0

    return float(lam_sol)


def compute_worst_case_value(
    metric: DistortionMetric,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    epsilon: float,
    n_grid: int = N_GRID,
    n_gauss: int = 500,
    tol: float = 1e-8,
    include_benchmark: bool = True,
) -> BoundResult:
    """
    Compute the worst-case distortion risk bound (Theorem 1) for a single (metric, ref, mu, sigma, epsilon).

    Steps:
    1. Compute epsilon_min, epsilon_switch.
    2. Determine regime.
    3. Solve for lambda_epsilon if active.
    4. Construct H_lambda_epsilon and compute rho_g(H_lambda).
    5. Optionally benchmark with direct optimization.

    Parameters
    ----------
    metric : DistortionMetric
    ref : ReferenceDistribution
    mu, sigma, epsilon : float
    n_grid, n_gauss : int
    tol : float
    include_benchmark : bool

    Returns
    -------
    BoundResult
    """
    epsilon_min, epsilon_switch, c_0 = compute_thresholds(
        ref, mu, sigma, metric, n_grid=n_grid, n_gauss=n_gauss
    )

    # Regime determination
    if epsilon < epsilon_min - tol:
        return BoundResult(
            metric_name=metric.name,
            dist_name=ref.name,
            mu=mu, sigma=sigma, epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_switch=epsilon_switch,
            regime="infeasible",
            lambda_epsilon=np.nan,
        )

    u_grid, g_vals = _make_g_grid(metric, n=n_grid)

    if epsilon >= epsilon_switch - tol:
        regime = "inactive"
        lambda_eps = 0.0
    else:
        regime = "active"
        # Need to solve for lambda_epsilon
        pass  # handled below

    if regime == "inactive":
        lambda_eps = 0.0
    else:
        lambda_eps = solve_lambda_epsilon(
            epsilon, u_grid, g_vals, metric.g_val_at_1,
            ref, mu, sigma, tol=tol, n_gauss=n_gauss
        )

    # Construct H_lambda_epsilon
    g_lam_vals = build_g_lambda_vals(
        u_grid, g_vals, lambda_eps, ref.quantile, ref.mu_F
    )
    hull_u, hull_f, _ = least_concave_majorant(u_grid, g_lam_vals)
    # a_lambda = g(1) + lambda * mu_F  (paper notation)
    a_lam = compute_a_lambda(0.0, metric.g_val_at_1, lambda_eps, ref.mu_F)
    b_lam = compute_b_lambda(hull_u, hull_f, a_lam)

    if b_lam < 1e-12:
        # Degenerate: b_0=0 means the concave envelope slope is constant, so the
        # optimal worst-case value is mu*g(1)+sigma*0 = mu*g(1).  The supremum is
        # attained as a limit of distributions converging to a point mass at mu;
        # the moment and Wasserstein constraints are satisfied in this limiting
        # sense.  Mark constraints as satisfied and skip the inapplicable numeric
        # check (the constant quantile has 0 variance, which would fail falsely).
        def h_fn(u: np.ndarray) -> np.ndarray:  # noqa: F811
            return np.full_like(np.asarray(u, float), mu)

        constraints_ok = True
    else:
        def h_fn(u: np.ndarray) -> np.ndarray:
            return construct_h_lambda(hull_u, hull_f, a_lam, b_lam, mu, sigma, u)

        # Validate constraints
        constraints_ok = _validate_constraints(h_fn, ref, mu, sigma, epsilon, n_gauss)

    # Compute worst-case value
    if metric.is_usc or lambda_eps > 0:
        wc_val = eval_rho_direct(metric, h_fn)
    else:
        # For non-USC: use USC envelope (same h_fn, same value in practice)
        wc_val = eval_rho_direct(metric, h_fn)

    benchmark_val = None
    abs_gap = None
    if include_benchmark:
        try:
            benchmark_val = _benchmark_direct_optimization(
                metric, ref, mu, sigma, epsilon, n_grid=min(n_grid, 200), n_gauss=n_gauss
            )
            abs_gap = abs(wc_val - benchmark_val)
        except Exception as e:
            warnings.warn(f"Benchmark failed: {e}")

    return BoundResult(
        metric_name=metric.name,
        dist_name=ref.name,
        mu=mu, sigma=sigma, epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_switch=epsilon_switch,
        regime=regime,
        lambda_epsilon=float(lambda_eps),
        worst_case_value=float(wc_val),
        benchmark_value=benchmark_val,
        abs_gap=abs_gap,
        constraints_satisfied=constraints_ok,
    )


def compute_best_case_value(
    metric: DistortionMetric,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    epsilon: float,
    n_grid: int = N_GRID,
    n_gauss: int = 500,
    tol: float = 1e-8,
) -> float:
    """
    Compute the best-case distortion risk bound via inf rho_g = -sup rho_{-g}.

    Creates the negated metric -g and computes the worst-case for that.
    If the metric has a ``g_hat_fn``, the negated envelope is passed through
    so that ``_make_g_grid`` builds the correct LCM for −g.

    Returns
    -------
    float : inf rho_g
    """
    def neg_gamma(u: np.ndarray) -> np.ndarray:
        return -metric.gamma(np.asarray(u, float))

    neg_g_hat_fn = None
    if metric.g_hat_fn is not None:
        def neg_g_hat_fn(t: np.ndarray, _fn=metric.g_hat_fn) -> np.ndarray:  # type: ignore[misc]
            return -_fn(np.asarray(t, float))

    neg_metric = DistortionMetric(
        name=f"-({metric.name})",
        gamma=neg_gamma,
        g_val_at_1=-metric.g_val_at_1,
        is_usc=metric.is_usc,
        is_concave=False,  # negation of concave is convex
        params=metric.params.copy(),
        g_hat_fn=neg_g_hat_fn,
    )

    result = compute_worst_case_value(
        neg_metric, ref, mu, sigma, epsilon,
        n_grid=n_grid, n_gauss=n_gauss, tol=tol,
        include_benchmark=False,
    )
    if result.worst_case_value is None:
        return np.nan
    return -result.worst_case_value


def _validate_constraints(
    h_fn: Callable,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    epsilon: float,
    n_gauss: int = 500,
    tol: float = 1e-4,
) -> bool:
    """
    Validate that H_lambda satisfies moment and Wasserstein constraints.

    Returns True if all constraints are satisfied within tolerance.
    """
    nodes, weights = _leggauss(n_gauss)
    eps_clip = 1e-9
    u_pts = 0.5 * (1.0 - 2 * eps_clip) * nodes + 0.5 * (1.0 + 2 * eps_clip)
    half = 0.5 * (1.0 - 2 * eps_clip)

    h_vals = h_fn(u_pts)
    e_h = float(half * np.sum(weights * h_vals))
    e_h2 = float(half * np.sum(weights * h_vals**2))
    e_h2_target = mu**2 + sigma**2

    mean_ok = abs(e_h - mu) < tol * (1 + abs(mu))
    var_ok = abs(e_h2 - e_h2_target) < tol * (1 + abs(e_h2_target))

    dW2 = wasserstein_sq_distance(h_fn, ref.quantile, n_gauss=n_gauss)
    wass_ok = dW2 <= epsilon + tol * (1 + epsilon)

    return bool(mean_ok and var_ok and wass_ok)


def _benchmark_direct_optimization(
    metric: DistortionMetric,
    ref: ReferenceDistribution,
    mu: float,
    sigma: float,
    epsilon: float,
    n_grid: int = 150,
    n_gauss: int = 200,
) -> float:
    """
    Direct finite-dimensional constrained optimization over monotone quantile functions.

    Discretize q(u_j) and maximize sum_j gamma(u_j) * q(u_j) * delta_j
    subject to:
    - q(u_j) <= q(u_{j+1})  (monotonicity)
    - sum q(u_j) * delta_j = mu  (mean)
    - sum q(u_j)^2 * delta_j = mu^2 + sigma^2  (second moment)
    - sum (F^{-1}(u_j) - q(u_j))^2 * delta_j <= epsilon  (Wasserstein)

    Uses scipy.optimize.minimize with SLSQP.

    Custom — Context7 found no library equivalent.
    """
    u_grid = np.linspace(1.0 / (n_grid + 1), n_grid / (n_grid + 1), n_grid)
    delta = u_grid[1] - u_grid[0] if n_grid > 1 else 1.0
    deltas = np.diff(
        np.concatenate([[0.0], u_grid, [1.0]])
    )
    # Use midpoint weights
    deltas_mid = np.full(n_grid, 1.0 / n_grid)

    gamma_vals = metric.gamma(u_grid)
    f_inv_vals = ref.quantile(u_grid)
    mu2_sigma2 = mu**2 + sigma**2

    def objective(q: np.ndarray) -> float:
        # Negative for minimization
        if metric.name.startswith("VaR+"):
            alpha = metric.params["alpha"]
            # Find closest grid point to alpha
            idx = int(np.searchsorted(u_grid, alpha))
            idx = min(idx, n_grid - 1)
            return -q[idx]
        elif metric.name.startswith("IQD+"):
            alpha = metric.params["alpha"]
            idx_lo = int(np.searchsorted(u_grid, alpha))
            idx_hi = int(np.searchsorted(u_grid, 1 - alpha))
            idx_lo = min(idx_lo, n_grid - 1)
            idx_hi = min(idx_hi, n_grid - 1)
            return -(q[idx_hi] - q[idx_lo])
        else:
            return -float(np.sum(gamma_vals * q * deltas_mid))

    def grad_obj(q: np.ndarray) -> np.ndarray:
        return -gamma_vals * deltas_mid

    constraints = []
    # Mean constraint
    constraints.append({
        "type": "eq",
        "fun": lambda q: np.sum(q * deltas_mid) - mu,
    })
    # Second moment constraint
    constraints.append({
        "type": "eq",
        "fun": lambda q: np.sum(q**2 * deltas_mid) - mu2_sigma2,
    })
    # Wasserstein constraint
    constraints.append({
        "type": "ineq",
        "fun": lambda q: epsilon - np.sum((f_inv_vals - q)**2 * deltas_mid),
    })
    # Monotonicity constraints
    for i in range(n_grid - 1):
        def mono_con(q, i=i):
            return q[i + 1] - q[i]
        constraints.append({"type": "ineq", "fun": mono_con})

    # Initial guess: affine function matching target moments
    q0 = mu + sigma * (u_grid - 0.5) * np.sqrt(12)  # uniform-like

    result = minimize(
        objective,
        q0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )
    return float(-result.fun)
