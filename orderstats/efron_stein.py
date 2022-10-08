"""Concentration  Inequalities For Order Statistics.

References:
    [1] Boucheron, S., & Thomas, M. (2012). Concentration inequalities for order statistics.
        Electronic Communications in Probability, 17, 1-12.
"""  # pylint: disable=line-too-long

from typing import Callable

import numpy as np


def jackknife_sample(X, col):
    bootstrap_col = np.random.choice(X[:, col], len(X), replace=False)
    new_x = X.copy()
    new_x[:, 2] = bootstrap_col
    return new_x


def efron_stein_inequality(X, function):
    """Efron-Stein Inequality for variance.

    Args:
        X: dataset
        function: $R^n -> R$
    Returns:
        Efron-Stein estimation of the variance of Z = function(X).
    """
    it = (np.mean(np.square(function(X) - function(jackknife_sample(X, col))))
          for col in range(X.shape[1]))
    return np.sum(np.fromiter(it, np.float64))


def entropy(X):
    """Entropy of a positive random variable X."""
    return np.mean(X * np.log(X)) - np.mean(X) * np.log(np.mean(X))


def logarithmic_sobolev(X, function, t):
    """Modified logarithmic Sobolev inequality.

    Upper bound for Ent[e^{t * Z}].
    """
    tau = lambda x: np.exp(x) - x - 1
    z = function(X)
    moment = np.exp(t * z)
    it = (np.mean(moment * tau(-t * (z - function(jackknife_sample(X, col)))))
          for col in range(X.shape[1]))
    return np.sum(np.fromiter(it, np.float64))


def order_statistic_spacing(X, k):
    """Variance estimate for the k-th order statistic."""
    n = len(X)
    if 1 <= k <= n // 2:
        return k * np.mean(np.square(X[:, k - 1] - X[:, k]))
    elif n // 2 <= k <= n:
        return (n - k + 1) * np.mean(np.square(X[:, k - 2] - X[:, k - 1]))


def order_statistic_entropy(X, k, t):
    """Estimate of $Ent[e^{tX_{(k)}}]$ for the k-th order statistic."""
    tau = lambda x: np.exp(x) - x - 1
    psi = lambda x: 1 + (x - 1) * np.exp(x)
    n = len(X)
    if 1 <= k <= n // 2:
        moement = np.exp(t * X[:, k])
        return k * np.mean(moement * psi(t * (X[:, k - 1] - X[:, k])))
    elif n // 2 <= k <= n:
        moment = np.exp(t * X[:, k - 1])
        return (n - k + 1) * np.mean(moment * tau(t *
                                                  (X[:, k - 2] - X[:, k - 1])))


def u_transform(cdf: Callable):

    def u_function(x):
        return 1 / (1 - cdf(x))

    return u_function


def hazard_rate(pdf: Callable, cdf: Callable) -> Callable:
    """Returns the hazard rate of the given probability distribution.

    The hazard rate of an absolutely continuous probability distribution with
    distribution function F is: h = f/F where f and F = 1−F are respectively
    the density and the survival function associated with F.
    """

    def hazard_rate_function(x):
        return pdf(x) / 1 - cdf(x)

    return hazard_rate_function


def renyi_representation(n, alpha=1):
    """Renyi's Representation for n i.i.d exponentials with mean alpha."""
    x = np.random.exponential(alpha, n)
    x = np.concatenate((np.zeros(1), x))
    x = np.diff(np.sort(x))
    y = x * np.arange(1, n + 1, 1)[::-1]
    renyi_rep = np.cumsum(y / np.arange(1, n + 1, 1)[::-1])
    return x, y, renyi_rep


def generalized_renyi_statistic(Z):
    """Generalized Renyi Statistic for non-negative i.i.d ranndom variables."""
    return np.cumsum(Z / np.arange(1, len(Z) + 1, 1)[::-1])
