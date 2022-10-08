"""Examplary pdfs on exponential distribution given in overleaf."""

from functools import partial

import numpy as np

from orderstats.utils import calculate_riemann_sum, prod


def pdf_sum_distinct_exp(t, betas):
    """Pdf for the sum of independent non identical exponential variables.
  Theorem 1 in Overleaf pg 7"""
    return np.sum([
        prod(betas) /
        prod([beta - beta_i for beta in betas if beta != beta_i]) *
        np.exp(-t * beta_i) for beta_i in betas
    ])


def get_lambda(n, i):
    return n - i + 1


def get_psi(i, n, m):
    lambdas = [get_lambda(n, i) for i in range(1, m + 1)]
    _lambda_i = lambdas[i - 1]
    assert _lambda_i == get_lambda(n, i)
    ls = [l - _lambda_i for l in lambdas if l != _lambda_i]
    psi_1 = prod(ls)
    return 1 / psi_1


def pdf_exp_t_nm(t, n, m):
    """Pdf for the sum of top n-m variables out of n iid exponentials."""
    k = n - m - 1

    def integrand(i, x):
        return np.exp(
            (get_lambda(n, i) / (k + 1) - 1.0) * x) * np.power(x, k - 1)

    lambdas = [get_lambda(n, i) for i in range(1, m + 2)]
    first_expr = (1 / np.math.factorial(k - 1))
    second_expr = (prod(lambdas) / (k + 1))
    the_sum = np.sum([
        get_psi(i=i, n=n, m=m + 1) * np.exp((-get_lambda(n, i) / (k + 1)) * t) *
        calculate_riemann_sum(partial(integrand, i), 0, t)
        for i in range(1, m + 2)
    ])
    return first_expr * second_expr * the_sum


def f_l(x, n, m):
    lambdas = [get_lambda(n, i) for i in range(1, m + 2)]
    first_expr = prod(lambdas) / (n - m)
    the_sum = np.sum([
        get_psi(i, n, m + 1) * np.exp(-(get_lambda(n, i) / (n - m)) * x)
        for i in range(1, m + 2)
    ])
    return first_expr * the_sum
