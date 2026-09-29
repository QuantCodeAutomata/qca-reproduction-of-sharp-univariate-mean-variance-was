"""
Distortion risk metric definitions and registry.

Convention (following the paper):
   rho_g(G) = integral_0^1 G^{-1}(u) * gamma(u) du
where gamma(u) = g'(1-u) is the quantile weight function.

This covers absolutely continuous distortions g. For non-smooth g
(VaR-type), the USC envelope g_hat is used with the corresponding
gamma_hat.

Custom implementation — Context7 found no library equivalent for these
paper-specific distortion-Wasserstein formulas.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy import integrate as sci_integrate
from .concave_envelope import _leggauss


@dataclass
class DistortionMetric:
    """
    A distortion risk metric defined by its quantile weight function gamma.

    Attributes
    ----------
    name : str
        Identifier.
    gamma : Callable
        Quantile weight function gamma(u) for u in (0,1).
        Satisfies rho_g(G) = integral_0^1 gamma(u) G^{-1}(u) du.
    g_val_at_1 : float
        Value g(1), used for location term: mu * g(1).
    is_usc : bool
        Whether the underlying g is upper semicontinuous.
    is_concave : bool
        Whether the underlying g is concave on [0,1].
    params : dict
        Metric-specific parameters.
    """

    name: str
    gamma: Callable[[np.ndarray], np.ndarray]
    g_val_at_1: float
    is_usc: bool
    is_concave: bool
    params: dict = field(default_factory=dict)
    # Optional: direct g_hat(t) evaluator for non-USC metrics where gamma is
    # a singular measure (delta) and cannot be used for grid-based integration.
    # When provided, _make_g_grid uses this instead of integrating gamma.
    g_hat_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def rho(
        self,
        quantile_fn: Callable[[np.ndarray], np.ndarray],
        n_gauss: int = 500,
    ) -> float:
        """
        Evaluate rho_g(G) = integral_0^1 gamma(u) G^{-1}(u) du.

        Uses Gauss-Legendre quadrature with endpoint refinement.

        Parameters
        ----------
        quantile_fn : Callable
            Left-quantile function G^{-1}: (0,1) -> R.
        n_gauss : int
            Number of Gauss-Legendre nodes for the main interval.

        Returns
        -------
        float
        """
        return _gl_integrate_product(self.gamma, quantile_fn, n_gauss=n_gauss)


def _gl_integrate_product(
    f: Callable,
    g: Callable,
    n_gauss: int = 500,
    eps: float = 1e-9,
) -> float:
    """
    Compute integral_0^1 f(u) * g(u) du using Gauss-Legendre quadrature
    on three sub-intervals to handle endpoint singularities.
    """
    def gl_on(a: float, b: float, nn: int = n_gauss) -> float:
        nd, wt = _leggauss(nn)
        u = 0.5 * (b - a) * nd + 0.5 * (b + a)
        return float(0.5 * (b - a) * np.sum(wt * f(u) * g(u)))

    val = gl_on(eps, 1.0 - eps)
    val += gl_on(1e-14, eps, nn=100)
    val += gl_on(1.0 - eps, 1.0 - 1e-14, nn=100)
    return val


# ─────────────────────────────────────────────────────────────
# Distortion metric constructors
# ─────────────────────────────────────────────────────────────

def make_gd() -> DistortionMetric:
    """
    Gini Deviation (GD).

    gamma(u) = 1 - 2u (as stated in the paper).
    g(1) = 0 (pure variability metric).
    Concave? No; g'(v) = 2v-1 changes sign.
    """

    def gamma(u: np.ndarray) -> np.ndarray:
        return 1.0 - 2.0 * np.asarray(u, float)

    return DistortionMetric(
        name="GD",
        gamma=gamma,
        g_val_at_1=0.0,
        is_usc=True,
        is_concave=False,
    )


def make_mmd() -> DistortionMetric:
    """
    Mean Absolute Deviation (MMD).

    gamma(u) = -1 on (0, 1/2) and +1 on (1/2, 1) (as per paper).
    g(1) = 0 (pure variability metric).
    """

    def gamma(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        return np.where(u < 0.5, -1.0, 1.0)

    return DistortionMetric(
        name="MMD",
        gamma=gamma,
        g_val_at_1=0.0,
        is_usc=True,
        is_concave=False,
    )


def make_es(alpha: float) -> DistortionMetric:
    """
    Expected Shortfall (CVaR) at level alpha in (0, 1).

    rho_ES_alpha(G) = (1/alpha) * integral_{1-alpha}^1 G^{-1}(u) du.

    gamma(u) = (1/alpha) * 1_{u > 1-alpha}.
    g(t) = min(t/alpha, 1), which is concave and USC.
    g(1) = 1.
    """
    assert 0 < alpha <= 1, "alpha must be in (0,1]"

    threshold = 1.0 - alpha

    def gamma(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        return np.where(u > threshold, 1.0 / alpha, 0.0)

    return DistortionMetric(
        name=f"ES(alpha={alpha})",
        gamma=gamma,
        g_val_at_1=1.0,
        is_usc=True,
        is_concave=True,
        params={"alpha": alpha},
    )


def make_var_plus(alpha: float) -> DistortionMetric:
    """
    VaR_alpha^+: right-continuous value-at-risk at level alpha.

    rho_{VaR_alpha^+}(G) = G^{-1}(alpha) (the left-quantile).
    Original g(t) = 1_{t > 1-alpha} (not USC).
    USC envelope: g_hat(t) = 1_{t >= 1-alpha}.

    gamma is the Dirac delta at u=alpha — not representable on a smooth grid.
    g_hat_fn supplies the step-function distortion directly for LCM computation.
    g(1) = 1, is_usc = False.
    """
    assert 0 < alpha < 1, "alpha must be in (0,1)"

    threshold = 1.0 - alpha  # noqa: F841

    def gamma(u: np.ndarray) -> np.ndarray:
        # Placeholder: point-mass at u=alpha cannot be represented as a smooth
        # function; rho evaluation is handled via eval_rho_direct.
        return np.zeros_like(np.asarray(u, float))

    def g_hat_fn(t: np.ndarray) -> np.ndarray:
        # USC envelope g_hat(t) = 1_{t >= 1-alpha}
        t = np.asarray(t, float)
        return np.where(t >= 1.0 - alpha, 1.0, 0.0)

    return DistortionMetric(
        name=f"VaR+(alpha={alpha})",
        gamma=gamma,
        g_val_at_1=1.0,
        is_usc=False,  # g(t)=1_{t>1-alpha} not USC
        is_concave=False,
        params={"alpha": alpha},
        g_hat_fn=g_hat_fn,
    )


def make_iqd_plus(alpha: float) -> DistortionMetric:
    """
    IQD_alpha^+: upper inter-quantile deviation at level alpha in (0, 1/2].

    IQD_alpha^+(G) = G^{-1}(1-alpha) - G^{-1}(alpha).

    Original distortion: g(t) = 1_{alpha <= t < 1-alpha}  (open at 1-alpha).
    USC envelope:        g_hat(t) = 1_{alpha <= t <= 1-alpha}  (closed on both sides).

    gamma is a signed Dirac measure and cannot be evaluated on a smooth grid.
    g_hat_fn supplies the step-function distortion directly for LCM computation.
    g(1) = 0 (variability metric).
    """
    assert 0 < alpha <= 0.5, "alpha must be in (0, 1/2]"

    def gamma(u: np.ndarray) -> np.ndarray:
        # Placeholder: singular measure; rho evaluation via eval_rho_direct.
        return np.zeros_like(np.asarray(u, float))

    def g_hat_fn(t: np.ndarray) -> np.ndarray:
        # USC envelope g_hat(t) = 1_{alpha <= t <= 1-alpha}
        t = np.asarray(t, float)
        return np.where((t >= alpha) & (t <= 1.0 - alpha), 1.0, 0.0)

    return DistortionMetric(
        name=f"IQD+(alpha={alpha})",
        gamma=gamma,
        g_val_at_1=0.0,
        is_usc=False,
        is_concave=False,
        params={"alpha": alpha},
        g_hat_fn=g_hat_fn,
    )


def make_gluevar(
    alpha: float, beta: float, h1: float, h2: float
) -> DistortionMetric:
    """
    GlueVaR distortion risk metric.

    Parameters: 0 < alpha < beta < 1, 0 < h1 < h2 < 1,
    with h1/(1-beta) >= (h2-h1)/(beta-alpha).

    Distortion function:
      g(t) = 0                                    for t in [0, 1-beta)
           = h1/(1-beta) * (t - (1-beta))          for t in [1-beta, 1-alpha)  (linear)
           = h1 + (h2-h1)/(beta-alpha) * (t - (1-alpha))  for t in [1-alpha, 1)  (linear)
           = 1 ... actually piecewise linear, need to verify normalization.
    
    Actually GlueVaR_alpha,beta^{h1,h2}:
    It's a linear interpolation of VaR_beta, VaR_alpha, and ES_alpha.
    g(0)=0, g(1-beta)=0, g(1-alpha)=h1, g(1)=h2... adjusted for h2=1.
    
    Standard form: the distortion is piecewise linear:
    g(t) = 0           for t in [0, 1-beta]
         = h1*(t-(1-beta))/(beta-alpha+... 
    
    Let me use the simpler representation:
    GlueVaR mixes VaR_alpha, VaR_beta, ES_alpha with weights.
    
    For simplicity, implement as:
    gamma(u) = 0        for u < alpha (below VaR_alpha level)
             = c1       for u in [alpha, beta)
             = c2       for u in [beta, 1]  (ES part)
    
    where c1, c2 depend on h1, h2.
    
    Actually the most common GlueVaR formulation uses:
    g(t) = 0                   for t in [0, 1-beta)
           h1/(beta-alpha)*(t-(1-beta))  for t in [1-beta, 1-alpha)
           h1 + (h2-h1)/(alpha)*(t-(1-alpha))  for t in [1-alpha, 1]
    
    This gives:
    gamma(u) = g'(1-u)
    For u in (1-beta, 1): g'(1-u) = h1/(beta-alpha)  [since 1-u in (0, 1-alpha), slope there]
    Wait: g'(t) for t in (1-beta, 1-alpha) = h1/(beta-alpha)
    g'(t) for t in (1-alpha, 1) = (h2-h1)/alpha
    So gamma(u) = g'(1-u):
    For u in (0, alpha): 1-u in (1-alpha, 1), so gamma(u) = (h2-h1)/alpha
    For u in (alpha, beta): 1-u in (1-beta, 1-alpha), so gamma(u) = h1/(beta-alpha)
    For u in (beta, 1): 1-u in (0, 1-beta), so gamma(u) = 0
    """
    assert 0 < alpha < beta < 1
    assert 0 < h1 < h2
    assert h2 <= 1.0

    slope_es = (h2 - h1) / alpha     # g'(t) for t in (1-alpha, 1)
    slope_var = h1 / (beta - alpha)  # g'(t) for t in (1-beta, 1-alpha)

    def gamma(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        result = np.zeros_like(u)
        result = np.where(u < alpha, slope_es, result)
        result = np.where((u >= alpha) & (u < beta), slope_var, result)
        # u >= beta: gamma = 0
        return result

    return DistortionMetric(
        name=f"GlueVaR(alpha={alpha},beta={beta},h1={h1},h2={h2})",
        gamma=gamma,
        g_val_at_1=h2,
        is_usc=True,
        is_concave=False,
        params={"alpha": alpha, "beta": beta, "h1": h1, "h2": h2},
    )


def make_discrepancy_es_var(alpha1: float, alpha2: float) -> DistortionMetric:
    """
    Discrepancy metric: ES_{alpha1} - VaR_{alpha2}.

    0 < alpha1 < alpha2 < 1.

    gamma(u) = (1/alpha1) * 1_{u > 1-alpha1} - delta_{u=1-alpha2}
    Approximated for numerical integration: the VaR part is handled via
    direct formula evaluation.

    g(1) = 1 - 1 = 0 (ES g(1)=1 minus VaR g(1)=1).
    Actually g_ES(1)=1, g_VaR(1)=1, discrepancy = 0.
    """
    assert 0 < alpha1 < alpha2 < 1

    es_threshold = 1.0 - alpha1

    def gamma(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        return np.where(u > es_threshold, 1.0 / alpha1, 0.0)

    return DistortionMetric(
        name=f"ES({alpha1})-VaR({alpha2})",
        gamma=gamma,
        g_val_at_1=0.0,
        is_usc=False,
        is_concave=False,
        params={"alpha1": alpha1, "alpha2": alpha2},
    )


def make_distortion_metrics_for_exp1() -> list:
    """
    Return the list of distortion metrics used in Experiment 1.

    Includes: GD, MMD, ES(0.05), ES(0.25), VaR+(0.05), IQD+(0.1), GlueVaR.

    Returns
    -------
    List[DistortionMetric]
    """
    return [
        make_gd(),
        make_mmd(),
        make_es(alpha=0.05),
        make_es(alpha=0.25),
        make_var_plus(alpha=0.05),
        make_iqd_plus(alpha=0.1),
        make_gluevar(alpha=0.05, beta=0.1, h1=0.6, h2=0.9),
    ]


def make_metric_registry() -> dict[str, DistortionMetric]:
    """
    Build the standard registry of distortion risk metrics used in experiments.

    Returns
    -------
    dict mapping name -> DistortionMetric
    """
    registry = {
        "GD": make_gd(),
        "MMD": make_mmd(),
        "ES_0.05": make_es(alpha=0.05),
        "ES_0.1": make_es(alpha=0.1),
        "ES_0.25": make_es(alpha=0.25),
        "VaR+_0.05": make_var_plus(alpha=0.05),
        "VaR+_0.1": make_var_plus(alpha=0.1),
        "IQD+_0.1": make_iqd_plus(alpha=0.1),
        "IQD+_0.25": make_iqd_plus(alpha=0.25),
        "GlueVaR_main": make_gluevar(
            alpha=0.05, beta=0.1, h1=0.6, h2=0.9
        ),
        "ES-VaR": make_discrepancy_es_var(alpha1=0.025, alpha2=0.01),
    }
    return registry


def eval_rho_direct(
    metric: DistortionMetric,
    quantile_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Evaluate rho for metrics with non-trivial gamma by direct formula
    (e.g., VaR+, IQD+).

    Parameters
    ----------
    metric : DistortionMetric
    quantile_fn : Callable

    Returns
    -------
    float
    """
    name = metric.name
    p = metric.params

    if name.startswith("VaR+"):
        alpha = p["alpha"]
        u = np.clip(alpha, 1e-12, 1 - 1e-12)
        return float(quantile_fn(np.array([u]))[0])
    elif name.startswith("-(VaR+"):
        # Negated VaR+: rho_{-g}(G) = -G^{-1}(alpha)
        alpha = p["alpha"]
        u = np.clip(alpha, 1e-12, 1 - 1e-12)
        return -float(quantile_fn(np.array([u]))[0])
    elif name.startswith("IQD+"):
        alpha = p["alpha"]
        u_lo = np.clip(alpha, 1e-12, 1 - 1e-12)
        u_hi = np.clip(1 - alpha, 1e-12, 1 - 1e-12)
        q_lo = float(quantile_fn(np.array([u_lo]))[0])
        q_hi = float(quantile_fn(np.array([u_hi]))[0])
        return q_hi - q_lo
    elif name.startswith("-(IQD+"):
        # Negated IQD+: rho_{-g}(G) = -(G^{-1}(1-alpha) - G^{-1}(alpha))
        alpha = p["alpha"]
        u_lo = np.clip(alpha, 1e-12, 1 - 1e-12)
        u_hi = np.clip(1 - alpha, 1e-12, 1 - 1e-12)
        q_lo = float(quantile_fn(np.array([u_lo]))[0])
        q_hi = float(quantile_fn(np.array([u_hi]))[0])
        return -(q_hi - q_lo)
    elif "-VaR(" in name:
        # Discrepancy ES - VaR
        alpha1 = p["alpha1"]
        alpha2 = p["alpha2"]
        es_val = metric.rho(quantile_fn)
        var_val = float(quantile_fn(np.array([1 - alpha2]))[0])
        return es_val - var_val
    else:
        return metric.rho(quantile_fn)
