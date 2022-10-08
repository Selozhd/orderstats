"""Density estimation for Heavy-tailed distributions."""

from functools import partial

import jax.numpy as np
from jax import lax
from jax.scipy import linalg

from orderstats.tail_estimation import DoubleBootstrap, hills_estimator
from orderstats.utils import calculate_riemann_sum, estimate_cdf


def estimate_pdf(x_array, x, n_bins):
    """Pdf estimate for histogram density estimation.
    E[f(x)] = lim h->0 1/h * [P(X<x+h) - P(X<x)]."""
    _, bin_edges = np.histogram(x_array, bins=n_bins)
    len_bin = np.diff(bin_edges)[0]
    density = (1 / len_bin) * (estimate_cdf(x_array, x + len_bin) -
                               estimate_cdf(x_array, x))
    return density


def get_estimation_error(x, x_test, n_bins):
    r"""L2 estimation error for histogram denstiy estimate.
    L2 error measure is: -2/n \sum_{i=1}^n \hat{f}(x_i) + \int \hat{f}^2(x) dx.
    """
    _, bin_edges = np.histogram(x, bins=n_bins)
    first_term = -2 / len(x) * np.sum(
        [estimate_pdf(x, i, n_bins) for i in x_test])
    second_term = np.sum([
        estimate_pdf(x, i, n_bins)**2 * 0.001
        for i in np.arange(bin_edges[0], bin_edges[-1], 0.001)
    ])  ## Start, stop, step
    return first_term + second_term


def scotts_factor(n, d=1):
    """
    D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
    Visualization", John Wiley & Sons, New York, Chicester, 1992.
    """
    return np.power(n, -1. / (d + 4))


def silverman_factor(n, d=1):
    """
    B.W. Silverman, "Density Estimation for Statistics and Data
    Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
    Chapman and Hall, London, 1986.
    """
    return np.power(n * (d + 2.0) / 4.0, -1. / (d + 4))


def get_cdf_from_kernel_estimate(kernel_density, lower_bound):
    """Returns cdf_func for density estimate, necessary for plot_cdf().

    Args:
        kernel_density: `scipy.stats.gaussian_kde`-like density estimate.
        lower_bound: Infimum of the support for the density estimation.
    """

    def cdf(x):
        return kernel_density.integrate_box(lower_bound, x)

    return cdf


class ParetoDensityEstimation:
    """Combined parametric - nonparametric density estimation.

    The logic presented here does not fully follow the Chapter 3.2 of
    Markovich. Kernel density estimation in particular is taken from `scipy`
    stats.gaussian_kde() function.

    Parameters:
        boundary: The boundary point between the non-parametric, parametric
                estimations.
        kernel_scaling: The total probability for the kernel density estimation
                        until the boundary point.
    """

    def __init__(self, dataset, weights=None):
        self.original_dataset = np.atleast_2d(np.asarray(dataset))
        if not self.original_dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.boundary, self.gamma = self.set_boundary(np.asarray(dataset))
        _dataset = self.original_dataset[self.original_dataset < self.boundary]
        self.dataset = np.atleast_2d(_dataset)
        self.d, self.n = self.dataset.shape
        self.n = np.sum(self.dataset < self.boundary)

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= np.sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1 / np.sum(self._weights**2)

        self.set_bandwith()
        self.kernel_scaling = estimate_cdf(self.original_dataset, self.boundary)

    def is_in_boundary(self, x):
        return x < self.boundary

    def _is_kernel_integrate_to_1(self):
        """Obsolete. Recommended in Markovich."""
        return calculate_riemann_sum(self.kernel_estimate, 0,
                                     self.boundary) >= 1.

    def _set_scaling(self):
        """Obsolete. Recommended in Markovich."""
        if self._is_kernel_integrate_to_1():
            return 1 + self.boundary**(-1 / self.gamma) + self.boundary**(
                -2 / self.gamma)
        else:
            return calculate_riemann_sum(self.pdf, 0, 1000)

    def set_boundary(self, dataset):
        dataset = dataset.sort()[::-1]
        hills_bootstrap = DoubleBootstrap()
        k_opt = hills_bootstrap(500, dataset)
        boundary = dataset[k_opt]
        gamma = hills_estimator(dataset, k_opt)
        return boundary, gamma

    def parametric_estimate(self, x):
        return (1 / self.gamma) * x**(-1 / (self.gamma - 1)) + (
            2 / self.gamma) * x**(-2 / (self.gamma - 1))

    def kernel_estimate(self, points):
        points = np.atleast_2d(np.asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (
                    d, self.d)
                raise ValueError(msg)

        result = np.zeros((m,), dtype=float)

        whitening = linalg.cholesky(self.inv_cov)
        scaled_dataset = np.dot(whitening, self.dataset)
        scaled_points = np.dot(whitening, points)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = scaled_dataset[:, i, np.newaxis] - scaled_points
                energy = np.sum(diff * diff, axis=0) / 2.0
                result += self.weights[i] * np.exp(-energy)
        else:
            # loop over points
            kernel_func = partial(self.gaussian_kernel, scaled_dataset)
            result = lax.map(kernel_func, scaled_points.reshape(-1))
        return result * self.kernel_scaling

    def pdf(self, x):
        if self.is_in_boundary(x):
            return self.kernel_estimate(x)
        else:
            return self.parametric_estimate(x)

    def set_bandwith(self):
        self.covariance_factor = scotts_factor
        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor(self.neff, self.d)
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, "_data_inv_cov"):
            self._data_covariance = np.atleast_2d(
                np.cov(self.dataset,
                       rowvar=1,
                       bias=False,
                       aweights=self.weights))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(linalg.det(2 * np.pi * self.covariance))

    def gaussian_kernel(self, scaled_dataset, scaled_point):
        diff = scaled_dataset - scaled_point
        energy = diff**2 / 2.0
        result = np.sum(np.exp(-energy) * self.weights)
        result = result / self._norm_factor
        return result

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.n) / self.n
            return self._weights

    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1 / sum(self.weights**2)
            return self._neff
