import unittest

import numpy as np
from scipy import stats

from orderstats.distributions import OrderSimulation, calculate_kappa, calculate_S_m
from orderstats.efron_stein import entropy, jackknife_sample, logarithmic_sobolev
from orderstats.tail_estimation import (get_bootstrap_samples, group_estimator,
                                        k_ratio, split_into_m)
from orderstats.utils import calculate_riemann_sum, check_sort, prod


def f(x):
    return 1 / (1 + np.power(x, 2))


def math_equal(x, y, sensitivity):
    return np.abs(x - y) < sensitivity


def example_func(X):
    """Example function f: R^n -> R for Efron-Stein inequalities."""
    mu = np.mean(X, axis=0)
    X_ = np.random.normal(mu, 0.01, X.shape)
    return np.mean(np.abs(X - X_), axis=-1)


class Test_math_functions(unittest.TestCase):

    def setUp(self):
        pass

    def test_integration(self):
        calc = calculate_riemann_sum(f, 0, 100, 0.001)
        result = np.math.atan(100)
        error = np.abs(calc - result)
        assert error < 0.001

    def test_product(self):
        assert np.math.factorial(29) == prod(range(1, 30))

    def tearDown(self):
        pass


class Test_simulation(unittest.TestCase):

    def setUp(self):
        self.N = 1000
        self.n = 100
        self.const = 50
        self.sim_normal = OrderSimulation(stats.norm(1, 1), calculate_S_m)
        self.sim = self.sim_normal(self.N, self.n, self.const)

    def test_len(self):
        assert len(self.sim) == self.N

    def test_difference_in_results(self):
        """Different sims should return different results."""
        rand_const = np.random.random_integers(self.n)
        assert np.all(self.sim != self.sim_normal(self.N, self.n, rand_const))

    def test_calculations_functs(self):
        obs = np.arange(1, 11, 1)
        rand1 = np.random.random_integers(10)
        rand2 = np.random.random_integers(len(self.sim))
        assert calculate_kappa(
            obs, rand1) == np.mean(obs[:rand1]) / np.mean(obs[rand1:])
        assert math_equal(calculate_kappa(self.sim, rand2),
                          np.mean(self.sim[:rand2]) / np.mean(self.sim[rand2:]),
                          sensitivity=0.1)

    def test_mean(self):
        sim = self.sim_normal(10000, 10, 10)
        #assert np.sum(np.logical_and(9 < sim, sim < 11)) < 5


class Test_Bootsrapping(unittest.TestCase):
    """Draft"""

    def setUp(self):
        self.X = stats.invweibull(0.4).rvs(10000)
        self.X[::-1].sort()
        self.alpha = 2 / 3
        self.beta = 1 / 2
        N = 1000
        self.n = len(self.X)
        n1 = self.n1()
        samples = get_bootstrap_samples(N, self.X, n1)
        self.samples = samples
        V = split_into_m(samples, 3)
        self.mse(V, self.X, 3)

    def n1(self):
        return int(self.n**self.beta)

    def get_m(self, n, m1):
        n1 = self.n1()
        return int(m1 * (n / n1)**self.alpha)

    def mse(self, V, X, m):
        n = len(X)
        z = np.mean(np.apply_along_axis(k_ratio, 2, V), axis=1)
        variance = np.nanvar(z)
        mean = group_estimator(X, self.get_m(n, m))
        bias = np.nanmean(z - mean)
        return bias**2 + variance

    def test_shuffle(self):
        X = self.samples
        assert X.ndim == 2
        X_ = X
        np.random.shuffle(X)
        assert np.sum(X != X) == 0
        X_ = np.sort(X_)[:, ::-1]
        X = np.sort(X)[:, ::-1]
        assert np.all(X == X_)


class Test_EfronStein(unittest.TestCase):

    def setUp(self):
        self.X = np.random.exponential(0.5, (10000, 10))
        self.Z = example_func(self.X)
        self.t = 1

    def test_logartihmic_sobolev(self):
        tau = lambda x: np.exp(x) - x - 1
        value = entropy(np.exp(self.t * self.Z))
        moment = np.exp(self.t * self.Z)
        it = (np.mean(
            moment *
            tau(-self.t *
                (self.Z - example_func(jackknife_sample(self.X, col)))))
              for col in range(self.X.shape[1]))
        efron_stein_estimate = np.sum(np.fromiter(it, np.float64))
        efron_stein_function_value = logarithmic_sobolev(
            self.X, example_func, self.t)
        assert value < efron_stein_function_value
        assert math_equal(efron_stein_function_value, efron_stein_estimate,
                          0.01)


if __name__ == '__main__':
    unittest.main()
