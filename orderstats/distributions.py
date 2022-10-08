r"""Simulations of order statistics.

Distribution examples from `scipy.stats`:
    >>> stats.norm(0, 1)  # Normal distribution with mean = 0, standart deviation = 1.
    >>> stats.cauchy(0, 1)  # Cauchy distribution with x_0 = 0, \gamma = 1.
    >>> stats.uniform(0, 1)  # Uniform distribution on [0, 1].
    >>> stats.expon(0, 1/2)  # Exponential distribution with lambda = 1/2.
    >>> stats.gamma(
    >>>    a=1, scale=1/2
    >>> )  # Gamma distribution with \frac{\beta^\alpha * x^{\alpha -1} e^{-\beta x}}{\Gamma(\alpha)}.
    >>> stats.chi2(df=1)  # Chi-squared distribution with degrees of freedom 1.
"""  # pylint: disable=line-too-long

import numpy as np
from scipy import stats

from orderstats.utils import sort_arrays, mix_arrays


def get_selection_vec(n, m):
    return np.concatenate((np.ones(m), np.zeros(n - m)))


def calculate_s_m(obs, m):
    return np.sum(np.take(obs, range(m)))


def calculate_t_nm(obs, m):
    n = len(obs)
    return np.sum(np.take(obs, range(m, n)))


def calculate_kappa(obs, m):
    n = len(obs)
    obs = np.asarray(obs)
    selection_vec = get_selection_vec(n, m)
    s_m = 1 / m * obs @ selection_vec
    t_n_m = 1 / (n - m) * obs @ (1 - selection_vec)
    return s_m / t_n_m


def calculate_unscaled_kappa(obs, m):
    n = len(obs)
    obs = np.asarray(obs)
    selection_vec = get_selection_vec(n, m)
    s_m = obs @ selection_vec
    t_n_m = obs @ (1 - selection_vec)
    return s_m / t_n_m


def calculate_asymptotic_stat(obs, k):
    """$S_{n,k}/X_{(n-k+1)}$ ratio of k-trimmed sum to next order statistic."""
    n = len(obs)
    obs = np.asarray(obs)
    selection_vec = get_selection_vec(n, n - k)
    s_nk = obs @ selection_vec
    x_k1 = np.take(obs, n - k)
    asymptotic_stat = s_nk / x_k1
    return asymptotic_stat


def moving_average_kappa(obs):
    n = len(obs)
    cum_sum = np.cumsum(obs)
    lens = np.arange(n)
    ma_kappa = (cum_sum / lens) / ((cum_sum[-1] - cum_sum) / (lens[-1] - lens))
    return ma_kappa


def moving_average_unscaled_kappa(obs):
    cum_sum = np.cumsum(obs)
    ma_kappa = cum_sum / (cum_sum[-1] - cum_sum)
    return ma_kappa


def moving_average_asymptotic_stat(obs):
    n = len(obs)
    cum_sum = np.cumsum(obs)
    obs_inc = obs[::-1]
    x_k1 = np.concatenate((np.take(obs_inc,
                                   range(1)), np.take(obs_inc, range(n - 1))))
    ma_asymptotic_stat = cum_sum[::-1] / x_k1
    return ma_asymptotic_stat


def calculate_min_kappa_index(obs, kappa_threshold):
    n = len(obs)
    for m in range(1, n):
        if calculate_unscaled_kappa(obs, m) < kappa_threshold:
            continue
        else:
            return m
    return m


def calculate_min_unscaled_index(obs, kappa_threshold):
    values = moving_average_unscaled_kappa(obs)
    m = np.sum(values < kappa_threshold)
    return m


def calculate_min_asymptotic_index(obs, kappa_threshold):
    n = len(obs)
    for m in range(1, n):
        if calculate_asymptotic_stat(obs, m) < kappa_threshold:
            continue
        else:
            return m
    return m


def identified_outliers_model(n, lambda_, k, b):
    """Outlier generator for identified outliers model."""
    sample = stats.expon(0, lambda_).rvs(n - k)
    anomalous_sample = stats.expon(0, lambda_ * b).rvs(k)
    outlier_sample = np.sort(np.concatenate((sample, anomalous_sample)))
    return outlier_sample


class IdentifiedOutliersModel:
    """Identified Outliers Model ease of use with `scipy.stats`."""

    def __init__(self, lambda_, k, b):
        self.name = "identified_outliers_model"
        self.lambda_ = lambda_
        self.k = k
        self.b = b
        self.dist = self  # A trick to do self.dist.name similiary to scipy

    def rvs(self, n):
        return identified_outliers_model(n, self.lambda_, self.k, self.b)


class OrderSimulation:
    """Simulation of order statistics given distribution and calculation
    function.

    Example:
        Here is an example for calculating kappa value for m = 85 in a sequence of
        100 i.i.d standart normal rvs.
        >>> simulate_normal_dist = OrderSimulation(stats.norm(0, 1), calculate_kappa)
        >>> simulation = simulate_normal_dist(10000, 100, 85)

    Attributes:
        random_variable: scipy.stats random_variable
        value_calculator: A function that takes an observation
            and a constant and calculates a value. e.g. one of calculate_kappa,
            calculate_S_m, or calculate_T_nm.
    """

    def __init__(self, random_variable, value_calculator):
        self.value_calculator = value_calculator
        self.random_variable = random_variable

    def sort(self, array):
        return np.sort(array)

    def simulate(self, n, const):
        obs = self.random_variable.rvs(n)
        obs = self.sort(obs)
        return self.value_calculator(obs, const)

    def __call__(self, N, n, const):
        sim = np.array([self.simulate(n, const) for i in range(N)])
        return sim


class MixSimulation:
    """Simulation study of a change point of distributions.

    dist1, dist2 are functions that takes the number of samples and
    returns a sample. It is assumed that dist2 has a heavier tail.
    """

    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2

    def get_mixed_array(self, n1, n2):
        """Mixed sample from `dist1` and `dist2`.

        Returns:
            Arrays of shape (n1+n2,) of mixed distributions and of 0, 1 indices
            corresponding to `dist1` and `dist2`.
        """
        x = self.dist1(n1)
        y = self.dist2(n2)
        x_, idx = mix_arrays([x, y])
        x_, idx = sort_arrays(x_, idx)
        return x_, idx

    def simulate_proportion(self, n1, n2, scoring_func, threshold):
        """Proportion of the heavier tail above the scoring threshold."""
        x, idx = self.mixed_array(n1, n2)
        scores = scoring_func(x)
        mask_threshold = scores > threshold
        idx_right = idx[mask_threshold]
        return np.mean(idx_right)
