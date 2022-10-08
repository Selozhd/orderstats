"""Utils."""

import math
from functools import reduce

import numpy as np


def _multiply(x, y):
    return x * y


def prod(ls):
    return reduce(_multiply, ls)


def estimate_cdf(X_array, x):
    """P(X < x)"""
    return np.mean(X_array < x)


def calculate_riemann_sum(func, start, stop, increment=0.001):
    partition = np.arange(start, stop, increment)
    return np.sum(func(partition) * increment)


def convolution(t, f, g):
    conv = lambda x: f(t - x) * g(x)
    return calculate_riemann_sum(conv, 0, 100)


def check_sort(X):
    n_increasing = np.sum(np.diff(X) > 0)
    if n_increasing == len(X) - 1:
        return 'increasing'
    elif n_increasing == 0:
        return 'decreasing'
    else:
        return 'neither'


def sort_arrays(x, y):
    """Sorts x and y with respect to x."""
    args = np.argsort(x)
    if isinstance(y, list):
        return [y_[args] for y_ in y]
    else:
        return x[args], y[args]


def mix_arrays(arrays, labels=None, axis=0):
    """Concatenates a list of arrays into 2 arrays of id-value pairs.

    Example:
        >>> mix_arrays([np.full(3, 3), np.full(3, 1)])
        (array([3, 3, 3, 1, 1, 1]), array([0, 0, 0, 1, 1, 1]))
    """
    iterator = zip(labels, arrays) if labels else enumerate(arrays)
    ls = [(array, np.full(len(array), idx)) for idx, array in iterator]
    new_arrays, idx = zip(*ls)
    return np.concatenate(new_arrays, axis=axis), np.concatenate(idx, axis=axis)


def scotts_rule(sim):
    """Scott's rule for histogram bin size."""
    n = len(sim)
    bin_size = 3.49 * np.std(sim) * n**(-1 / 3)
    n_bins = (np.max(sim) - np.min(sim)) / bin_size
    return int(n_bins)


# Curve fitting
def exp_fit(x, a, b, c):
    return a * np.exp(-b * x) + c


def logistic_fit(x, L, k, x_0):
    """f(x) = L / 1 + e^(-k(x-x_0))."""
    return L / (1 + math.e**(-k * (x - x_0)))


def polynomial_fit(x, a, b, c):
    return a * x + b * x**2 + c


def pareto_fit(x, a):
    return a * np.power(x, a) / np.power(x, a + 1)


def power_law(x, gamma):
    first_term = (1 / gamma) * x**(-1 / (gamma - 1))
    second_term = (2 / gamma) * x**(-2 / (gamma - 1))
    return first_term + second_term
