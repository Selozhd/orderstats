"""Methods for estimating tail decay for regularly-varying distributions.

Hill's estimator, ratio estimator, group esimator as well as bootstrapping
  and plotting methods for deciding parameters.

Most of the code here is taken from the works of:
    [1] Ivan Voitalov, Pim van der Hoorn, Remco van der Hofstad, and Dmitri Krioukov
    Phys. Rev. Research 1, 033034 – Published 18 October 2019

    [2] Markovich, N. (2008). Nonparametric analysis of univariate heavy-tailed data:
    research and practice (Vol. 753). John Wiley & Sons.

    Example usage for the bootstrap methods:
    >>> from scipy import stats
    >>> cauchy_sample = stats.cauchy(0,1).rvs(10000)
    >>> cauchy_sample[::-1].sort()  # sort in decreasing order
    >>> single_bootstrap = SingleBootstrap()
    >>> double_bootstrap = DoubleBootstrap()
    >>> k_single =  single_bootstrap(1000, cauchy_sample)
    >>> k_double =  double_bootstrap(1000, cauchy_sample)
    >>> gamma = hills_estimator(cauchy_sample, k_double)
"""  # pylint: disable=line-too-long

import numpy as np

from orderstats.plot_utils import plt_plot
from orderstats.utils import check_sort


class NonDecreasingArrayError(Exception):
    """Raise when an array must be decreasing.

    Only call through `_check_if_decreasing()`.
    """
    pass


def _check_if_decreasing(array, msg=""):
    """Raises NonDecreasingArrayError if an array is not decreasing.

    Use this check before expensive calculations.
    """
    result = check_sort(np.asarray(array))
    if result == "decreasing":
        return
    else:
        raise NonDecreasingArrayError(msg)


def hills_estimator(X, k):
    """Calculates Hill's estimator for k.
    Args:
        X: A numpy array of decreasing order.
        k: k-th index for which the Hill's estimator will be calculated.
    Returns:
        Hill's estimator for k.
    """
    logs = np.log(X[:k])
    log_k = np.log(X[k])
    return (1. / k) * np.sum(logs - log_k)


def hills_estimator_full(X):
    """Calculates first moments array given an ordered data sequence.
    Decreasing ordering is required.
    Args:
        X: A numpy array of decreasing order.

    Returns:
        A numpy array of 1st moments (Hill estimator) corresponding to all
        possible order statistics of the dataset.
    """
    _check_if_decreasing(X, msg="`X` must be a decreasing array!")
    logs_1 = np.log(X)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(X))
    m1 = (1. / k_vector) * logs_1_cumsum - logs_1[1:]
    return m1


def ratio_estimator(X, x_n):
    """Calculates ratio estimator for x_n.
    Args:
        X: numpy array of decreasing order.
        x_n: value which the ratio estimator
            will be calculated.
    Returns:
      ratio estimator for x_n.
    """
    indicator = X > x_n
    return np.sum(np.log(X / x_n) * indicator) / np.sum(indicator)


def ratio_estimator_vector(X, x_n):
    """Calculates ratio estimator for x_n in samples.
    Args:
        X: 2-dim numpy array of (# samples, sample) in which each sample
            is of decreasing order.
        x_n: value which the ratio estimator
            will be calculated.
    Returns:
        ratio estimator for x_n.
    """
    _check_if_decreasing(X, msg="`X` must be a decreasing array!")
    indicator = X > x_n
    return np.sum(np.log(X / x_n) * indicator, axis=1) / np.sum(indicator,
                                                                axis=1)


def get_pickands_possible_indices(n):
    """Returns an array possible indices for Pickands estimator.
    The same indices are used pickands_estimator_vector()."""
    return np.arange(1, int(np.floor(n / 4.)) + 1)


def pickands_estimator(X, k):
    """Calculates Pickands estimator for the tail index.

    First proposed in Embrechts et al., 1997.
    Args:
        X : A numpy array of decreasing order.
        k: k-th index for which the Pickands estimator will be calculated.
    Raises:
        ValueError if value of k does not satisfy, k <= len(X)/4.
    Returns:
        Pickands estimator for k.
    """
    n = len(X)
    if k > n / 4:
        raise ValueError("The value of k must be k <= n/4.")
    else:
        return (1. / np.log(2)) * np.log(
            (X[k - 1] - X[2 * k - 1]) / (X[2 * k - 1] - X[2 * k - 1]))


def pickands_estimator_vector(X):
    """Calculates Pickands estimator for the tail index.
    Args:
        X : A numpy array of decreasing order.
    Returns:
        A numpy array of Pickands estimations and the corresponding indices
        in float.
    """
    _check_if_decreasing(X, msg="`X` must be a decreasing array!")
    indices = get_pickands_possible_indices(len(X))
    x_k = X[indices - 1]
    x_2k = X[2 * indices - 1]
    x_4k = X[4 * indices - 1]
    pickands_vector = (1. / np.log(2)) * np.log((x_k - x_2k) / (x_2k - x_4k))
    return pickands_vector


def empirical_mean_excess(X, u):
    """Empirical mean excess for a decreasing array X and a given u."""
    indicator = X > u
    excess_values = (X - u) * indicator
    return np.sum(excess_values) / np.sum(indicator)


def exceedance_plot(X):
    """Draw exceedance plot for a decreasing array X."""
    ls_u = np.arange(np.min(X), np.max(X), 0.01)
    e_u = [empirical_mean_excess(X, u) for u in ls_u]
    plt_plot(ls_u, e_u)


def get_bootstrap_sample(X, n1):
    """Draws with replacement a decreasing samples from X of size n."""
    sample = np.random.choice(X, size=n1, replace=True)
    sample[::-1].sort()
    return sample


def get_bootstrap_samples(N, X, n):
    """Draws with replacement N decreasing samples from X of size n."""
    return np.array([
        np.sort(np.random.choice(X, size=n, replace=True))[::-1]
        for i in range(N)
    ])


def get_double_bootsrap_errors(X):
    """Calculates AMSE estimates for all bootstrap samples."""
    logs_1 = np.log(X)
    logs_2 = (logs_1)**2
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    k_vector = np.arange(1, len(X))
    m1 = (1. / k_vector) * logs_1_cumsum - logs_1[1:]
    m2 = (1. / k_vector) * logs_2_cumsum - (
        2. * logs_1[1:] / k_vector) * logs_1_cumsum + logs_2[1:]
    return (m2 - 2. * m1**2)**2


def get_indecies_not_nan(X):
    return np.where(np.logical_not(np.isnan(X)))


def danielsson_rho(k1, n1):
    rho = (np.log(k1) /
           (2. * np.log(n1) - np.log(k1)))**(2. * (np.log(n1) - np.log(k1)) /
                                             (np.log(n1)))
    return rho


def qi_rho(k1, n1):
    rho = (1. - (2 * (np.log(k1) - np.log(n1)) /
                 (np.log(k1))))**(np.log(k1) / np.log(n1) - 1.)
    return rho


def markovich_rho(k1, n1):
    rho = np.log(k1) / (2 * np.log(k1 / n1))
    return np.power(1 - 1 / rho, 2 / (2 * rho - 1))


class SingleBootstrap:
    """Single bootstrap method for the selection of `k` in Hill's Estimator.

    alpha and beta estimates for default to 2/3, and 1/2 respectively which are
    recommended by Hall(1990) for Pareto type distributions.

    Attributes:
        alpha: Parameter controlling the k values in bootstrap samples.
        beta: Parameter for the number of samples in each bootstrap sample.
        eps_sensitivity: Parameter controlling the range of MSE minimization,
                        defined as the fraction of order statistics to consider
                        during the MSE minimization.
        min_index: The start of indexes for the candidate k1 values.
                Minimum value is 1.
    """

    def __init__(self, alpha=2 / 3, beta=1 / 2, eps_sensitivity=1.):
        self.alpha = alpha
        self.beta = beta
        self.eps_sensitivity = eps_sensitivity
        self.min_index = 1

    def n1(self, n):
        return int(n**self.beta)

    def get_k(self, n, k1):
        return int(k1 * (n / self.n1(n))**self.alpha)

    def mse(self, samples, original_data):
        """Calculates MSE estimates for all bootsrap samples.
        Recommended method in Caers and Van Dyck(1999)
        """
        hill = np.array([hills_estimator_full(sample) for sample in samples])
        variance = np.nanvar(hill, axis=0)
        ls_k = np.array([
            self.get_k(len(original_data), i)
            for i in np.arange(1, samples.shape[1])
        ])
        mean = np.array([hills_estimator(original_data, k) for k in ls_k])
        bias = np.nanmean(hill, axis=0) - mean
        mse = bias**2 + variance
        return mse

    def get_argmin(self, n_, mse_array):
        """Finds the value of k1 which minimizes the MSE estimate."""
        max_index = (
            np.abs(np.linspace(1. / n_, 1.0, n_) -
                   self.eps_sensitivity)).argmin()
        k_ = np.nanargmin(mse_array[self.min_index:max_index]
                         ) + 1 + self.min_index  # take care of indexing
        return k_

    def bootstrap_step(self, N, X, n1):
        samples = get_bootstrap_samples(N, X, n1)
        mse_array = self.mse(samples, X)
        return mse_array

    def search_k_opt(self, N, X):
        """Find optimum k value using N samples from X.

        Args:
            N: Number of bootstrap samples to draw.
            X: A numpy array of decreasing order.

        Returns:
            Single bootstrap estimate for the optimal value of k.
        """
        _check_if_decreasing(X, msg="`X` must be a decreasing array!")
        n = len(X)
        n1 = self.n1(n)
        mse_array = self.bootstrap_step(N, X, n1)
        k1 = self.get_argmin(n1, mse_array)
        k_opt = self.get_k(n, k1)
        return k_opt

    __call__ = search_k_opt


class DoubleBootstrap:
    """Double bootstrap method for the selection of `k` in Hill's Estimator.
    Proposed in Danielsson et al. (1997)

    Attributes:
        t_bootstrap: Parameter controlling the size of the 2nd bootstrap,
                     defined from n2 = n*(t_bootstrap).
        eps_sensitivity: parameter controlling range of AMSE minimization.
                         Defined as the fraction of order statistics to consider
                         during the AMSE minimization step.
        self.rho_func: rho value used in the estimation of k_opt from k1.
        self.plotting: Flag for the generation of AMSE diagnostic plots.
        min_index: The start of indexes for the candidate k1 values.
                Minimum value is 1.
    """

    def __init__(self,
                 t_bootstrap=0.5,
                 eps_sensitivity=1.0,
                 rho_func=markovich_rho,
                 plotting=False):
        self.t_bootstrap = t_bootstrap
        self.eps_sensitivity = eps_sensitivity
        self.plotting = plotting
        self.rho_func = rho_func
        self.min_index = 1

    def refresh_min_index(self):
        self.min_index = 1

    def diagnostic_plots(self, n_, k_averages):
        amse = k_averages
        x_arr = np.linspace(1. / n_, 1.0, n_)
        plt_plot(x_arr, amse)

    def eps(self, n):
        return 0.5 * (1 + np.log(int(self.t_bootstrap * n)) / np.log(n))

    def n1(self, n):
        eps = self.eps(n)
        return int(n**eps)

    def n2(self, n, n1):
        return int(np.power(n1, 2) / float(n))

    def get_argmin(self, n_, k_averages):
        """Finds the value of k1 which minimizes the AMSE estimate."""
        max_index = (
            np.abs(np.linspace(1. / n_, 1.0, n_) -
                   self.eps_sensitivity)).argmin()
        k_ = np.nanargmin(k_averages[self.min_index:max_index]
                         ) + 1 + self.min_index  # take care of indexing
        return k_

    def bootstrap_step(self, N, X, n_):
        samples = np.zeros(n_ - 1)
        good_counts = np.zeros(n_ - 1)
        for _ in range(N):
            sample = get_bootstrap_sample(X, n_)
            amse = get_double_bootsrap_errors(sample)
            samples += amse
            good_counts[get_indecies_not_nan(amse)] += 1
        return samples / good_counts

    def get_k_opt(self, n1, k1, k2):
        k_opt = (np.power(k1, 2) / float(k2)) * self.rho_func(k1, n1)
        k_opt = int(np.round(k_opt))
        return k_opt

    def check_k_opt(self, k_opt, n):
        if k_opt == 0:
            k_opt = 2
        if int(k_opt) >= n:
            print(
                "WARNING: estimated threshold k is larger than the size of data"
            )
            k_opt = n - 1
        return k_opt

    def get_bootstrap_values(self, N, X, n1, n2):
        k1_averages = self.bootstrap_step(N, X, n1)
        k2_averages = self.bootstrap_step(N, X, n2)
        k1 = self.get_argmin(n1, k1_averages)
        k2 = self.get_argmin(n2, k2_averages)
        if self.plotting:
            self.diagnostic_plots(n1, k1_averages)
            self.diagnostic_plots(n2, k2_averages)
        return k1, k2

    def search_k_opt(self, N, X):
        """Iteratively searches for the optimal k for the Hill's estimator.
        Args:
            N: Number of samples to generate for bootstrapping.
            X: numpy array of decreasing order.
        Returns:
            Optimal k value.
        """
        _check_if_decreasing(X, msg="`X` must be a decreasing array!")
        n = len(X)
        n1 = self.n1(n)
        n2 = self.n2(n, n1)
        k1, k2 = self.get_bootstrap_values(N, X, n1, n2)
        if k2 > k1:
            print(
                "Warning (Hill): k2 > k1, AMSE false minimum suspected, resampling..."
            )
            self.min_index = self.min_index + int(0.005 * n)
            self.search_k_opt(N, X)
        else:
            k_opt = self.get_k_opt(n1, k1, k2)
            k_opt = self.check_k_opt(k_opt, n)
            self.refresh_min_index()
            return k_opt

    __call__ = search_k_opt


class RatioBootstrap:
    """Single bootstrapping method for ratio estimator."""

    def __init__(self, alpha=2 / 3, beta=1 / 2, eps_sensitivity=1.):
        self.alpha = alpha
        self.beta = beta
        self.eps_sensitivity = eps_sensitivity
        self.min_index = 1

    def n1(self, n):
        return int(n**self.beta)

    def get_corresponding_index(self, X, opt_value):
        return np.sum(X > opt_value) - 1

    def _mse(self, samples, original_data, k1):
        hill = ratio_estimator_vector(samples, k1)
        variance = np.nanvar(hill)
        mean = ratio_estimator(original_data, k1)
        bias = np.nanmean(hill) - mean
        mse = bias**2 + variance
        return mse

    def get_argmin(self, n_, k_averages):
        """Finds the value of k1 which minimizes the AMSE estimate."""
        max_index = (
            np.abs(np.linspace(1. / n_, 1.0, n_) -
                   self.eps_sensitivity)).argmin()
        k_ = np.nanargmin(k_averages[self.min_index:max_index]
                         ) + 1 + self.min_index  # take care of indexing
        return k_

    def __call__(self, N, X, max_search):
        _check_if_decreasing(X, msg="`X` must be a decreasing array!")
        n = len(X)
        n1 = self.n1(n)
        samples = get_bootstrap_samples(N, X, self.n1(n))
        search = np.linspace(0., max_search, 10000)
        mse = [self._mse(samples, X, k_) for k_ in search]
        opt_value = self.get_argmin(n1, mse)
        return opt_value


def split_into_m(X, m):
    np.random.shuffle(X)
    X = arrange_dims(X, m)
    if X.ndim == 1:
        v = X.reshape(-1, m)
        v = np.sort(v)[:, ::-1]  # sort decreasing in the second dim
    elif X.ndim == 2:
        n = X.shape[0]
        v = X.reshape(n, -1, m)
        v = np.sort(v)[:, :, ::-1]  # sort decreasing in the third dim
    return v


def arrange_dims(X, m):
    if X.ndim == 1:
        last_sample = len(X) % m
        zeros = np.zeros(m - last_sample)
        X = np.concatenate([X, zeros])
    elif X.ndim == 2:
        zero_dim = X.shape[0]
        n = X.shape[1]
        last_sample = n % m
        zeros = np.zeros((zero_dim, int(m - last_sample)))
        X = np.concatenate([X, zeros], axis=1)
    return X


def k_ratio(X):
    return X[1] / X[0]


def get_z(V):
    return np.mean(np.apply_along_axis(k_ratio, 1, V))


def get_gamma(z):
    return (1. / z) - 1


def get_possible_m(n):
    """Generates a numpy array according to the inequality 2 < m < n/2."""
    return np.arange(3, int(n / 2), 1)


def group_estimator(X, m):
    """Group estimator for a given m.
    Args:
        X: A numpy array of decreasing order.
    Returns:
        The estimated gamma value."""
    v = split_into_m(X, m)
    z = get_z(v)
    gamma = get_gamma(z)
    return gamma


class GroupBootstrap:
    """Bootstrap method for the selection of `m` in group estimator.

    Attributes:
        alpha: Parameter controlling the m values in bootstrap samples.
        beta: Parameter for the number of samples in each bootstrap sample.
        eps_sensitivity: parameter controlling range of MSE minimization.
                         Defined as the fraction of order statistics to consider
                         during the MSE minimization step.
        min_index: The start of indexes for the candidate m values.
                Minimum value is 1.
    """

    def __init__(self, alpha=2 / 3, beta=1 / 2, eps_sensitivity=1.):
        self.alpha = alpha
        self.beta = beta
        self.eps_sensitivity = eps_sensitivity
        self.min_index = 1

    def n1(self, n):
        return int(n**self.beta)

    def get_m(self, n, m1):
        return int(m1 * (n / self.n1(n))**self.alpha)

    def mse(self, V, X, m):
        n = len(X)
        z = np.mean(np.apply_along_axis(k_ratio, 2, V), axis=1)
        variance = np.nanvar(z)
        mean = group_estimator(X, self.get_m(n, m))
        bias = np.nanmean(z - mean)
        return bias**2 + variance

    def get_argmin(self, n_, mse_array):
        """Finds the value of m which minimizes the MSE estimate."""
        max_index = (
            np.abs(np.linspace(1. / n_, 1.0, n_) -
                   self.eps_sensitivity)).argmin()
        k_ = np.nanargmin(mse_array[self.min_index:max_index]
                         ) + 1 + self.min_index  # take care of indexing
        return k_

    def bootstrap_step(self, samples, X):
        mse_array = []
        for m in get_possible_m(samples.shape[-1]):
            v = split_into_m(samples, m)
            mse_array.append(self.mse(v, X, m))
        return np.array(mse_array)

    def __call__(self, N, X):
        _check_if_decreasing(X, msg="`X` must be a decreasing array!")
        n = len(X)
        n1 = self.n1(n)
        samples = get_bootstrap_samples(N, X, n1)
        mse_array = self.bootstrap_step(samples, X)
        m_opt = self.get_argmin(n1, mse_array)
        return m_opt
