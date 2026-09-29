"""
Reference distributions for Wasserstein ambiguity sets.

Provides standardized Gaussian and Student-t reference distributions with
explicit quantile functions F^{-1}(u), mean mu_F, and standard deviation sigma_F.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy import stats


@dataclass
class ReferenceDistribution:
    """
    Container for a reference distribution with quantile function and moments.

    Attributes
    ----------
    name : str
        Human-readable name.
    quantile_fn : Callable
        Left-quantile function F^{-1}(u) for u in (0, 1).
    mu_F : float
        Mean of the reference distribution.
    sigma_F : float
        Standard deviation of the reference distribution (> 0).
    """

    name: str
    quantile_fn: Callable[[np.ndarray], np.ndarray]
    mu_F: float
    sigma_F: float

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the quantile function at probability levels u."""
        u = np.asarray(u, dtype=float)
        return self.quantile_fn(u)


def make_gaussian_reference(mu: float = 0.0, sigma: float = 1.0) -> ReferenceDistribution:
    """
    Construct a Gaussian reference distribution N(mu, sigma^2).

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (> 0).

    Returns
    -------
    ReferenceDistribution
    """
    assert sigma > 0, "sigma must be positive"

    def q(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-15, 1 - 1e-15)
        return mu + sigma * stats.norm.ppf(u)

    return ReferenceDistribution(
        name=f"Gaussian(mu={mu}, sigma={sigma})",
        quantile_fn=q,
        mu_F=mu,
        sigma_F=sigma,
    )


def make_student_t_reference(
    mu: float = 0.0, sigma: float = 1.0, nu: float = 5.0
) -> ReferenceDistribution:
    """
    Construct a Student-t reference distribution with mean mu, standard deviation sigma,
    and degrees of freedom nu > 2.

    The raw t_nu has variance nu/(nu-2). We rescale so that the resulting distribution
    has variance exactly sigma^2: X = mu + sigma * sqrt((nu-2)/nu) * T_nu.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (> 0).
    nu : float
        Degrees of freedom (> 2 so that variance exists).

    Returns
    -------
    ReferenceDistribution
    """
    assert sigma > 0, "sigma must be positive"
    assert nu > 2, "nu must be > 2 for finite variance"

    # Scale factor so that Var(X) = sigma^2
    scale = sigma * np.sqrt((nu - 2.0) / nu)

    def q(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-15, 1 - 1e-15)
        return mu + scale * stats.t.ppf(u, df=nu)

    return ReferenceDistribution(
        name=f"Student-t(mu={mu}, sigma={sigma}, nu={nu})",
        quantile_fn=q,
        mu_F=mu,
        sigma_F=sigma,
    )


# Default standardized references used throughout experiments
STANDARD_GAUSSIAN = make_gaussian_reference(mu=0.0, sigma=1.0)
STANDARD_STUDENT_T5 = make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0)
