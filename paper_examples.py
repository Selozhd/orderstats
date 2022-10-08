"""For the Experiments section in overleaf.

    - Plots analyzing the selected statistic in the beginning of the section:
        See analyse_distribution in plot_utils
    - Data generation and change point detection for mixture models:
        Our goal is to find the change point of a mixture model of RV distributions,
        with α values below and above 2.
"""

import numpy as np
from numpy import random
from scipy import stats

from orderstats import scoring
from orderstats.classification_utils import get_classification_report
from orderstats.distributions import (Identified_Outliers_Model,
                                      OrderSimulation,
                                      calculate_min_unscaled_index,
                                      calculate_unscaled_kappa,
                                      identified_outliers_model,
                                      moving_average_unscaled_kappa)
from orderstats.plot_utils import *
from orderstats.utils import check_sort, mix_arrays, sort_arrays

DISTRIBUTION_NAMES = {
    # Plot title names for `scipy.stats.distributions`
    "halfnorm": "Half-Normal",
    "halfcauchy": "Half-Cauchy",
    "chi2": "$\chi^2$",
    "pareto": "Pareto",
    "expon": "Exponential",
    "identified_outliers_model": "Identified Outliers Model"
}


def analyse_distribution(N, n, distribution):
    """Produces the plots at the beginning of the expirement section.

    The following plots are produced in order and saved in IMAGES_PATH:
        MA of the R statistic in the n sample
        MA increasing sample
        Histogram of m over N simulations given kappa value
        N simulations of kappa to choose same m

    Args:
        N: Number of times to iterate sampling.
        n: Size of the sample drawn from the distribution
        distribution: `scipy.stats` distribution

    Returns:
        threshold: Selected $\kappa$ value for threshold
        sample: Sampled n values from the distribution
        scores: Corresponding unscaled R statistic for each value in the sample
        predictions: `np.array` of 0's and 1's, in which 1 denotes anomalies
        sim: N list of n sample scores, for the concentration of $\kappa$
        sim2: N list of m cut-off indices, for the concentration of m
    """
    sample = distribution.rvs(n)
    sample_sorted = np.sort(sample)

    scores = moving_average_unscaled_kappa(sample_sorted)
    threshold = scoring.get_kappa_threshold(scores)
    predictions = scores > threshold
    m = n - np.sum(predictions)
    print(f"The value of threshold chosen by kneecap detection is: {threshold}")
    print(f"The corresponding value of m is: {m}")

    dist_name = distribution.dist.name
    kwargs1 = {
        'title': 'R statistic',
        'name': f'Rplot_{dist_name}',
        'kappa': np.round(threshold, 3),
        'm': m
    }
    kwargs2 = {
        'title': 'Order Statistic',
        'name': f'OrderStatsplot_{dist_name}',
        'kappa': np.round(sample_sorted[m - 1], 3),
        'm': m
    }
    plot_anomalies(scores, predictions, **kwargs1)
    plot_anomalies(sample_sorted, predictions, **kwargs2)

    simulator = OrderSimulation(distribution, calculate_unscaled_kappa)
    sim = simulator(N, n, m)
    sim2 = np.array([
        calculate_min_unscaled_index(distribution.rvs(n), threshold)
        for i in range(N)
    ])
    kwargs3 = {
        'title': f'Distribution of $\kappa$ conditioned on $m={m}$',
        'save': f'Hist_k_{dist_name}'
    }
    kwargs4 = {
        'title':
            f'Distribution of $m$ conditioned on $\kappa={np.round(threshold, 3)}$',
        'save':
            f'Hist_m_{dist_name}'
    }
    plot_simulation_histogram(sim, 100, **kwargs3)
    plot_simulation_histogram(sim2, 100, **kwargs4)
    return (threshold, sample, scores, predictions, sim, sim2)


def mixed_pareto_sample(N=1_000_000,
                        thin_tail=1.5,
                        heavy_tail=2.5,
                        threshold=3.685):
    """Proportion of the heavier tail in a simulated sample of two pareto.
    Args:
        N: Number of samples from each distribution.
        thin_tail: Pareto index for the thin tail.
        fat_tail: Pareto index for the heavy tail.
        threshold: Outlier score threshold value for the thin tail.
    Returns:
        Proportion of samples above the threshold from the heavy tail.
    """
    x = random.pareto(heavy_tail, N)
    y = random.pareto(thin_tail, N)
    X, idx = mix_arrays([x, y])
    X, idx = sort_arrays(X, idx)
    scores, _ = scoring.get_anomaly_scores(X)  # X is already sorted
    scores = np.nan_to_num(scores)
    mask_threshold = scores > threshold
    idx_right = idx[mask_threshold]
    return np.mean(idx_right)


def exploratory_plots(distribution, N=10_000, n=1000):
    """Plots for beginning of Experiments."""
    dist_name = distribution.dist.name
    sample = np.sort(distribution.rvs(n))
    stat = moving_average_unscaled_kappa(sample)
    _save_sample = lambda x: "images/" + 'exp_8/samples_' + x
    _save_stat = lambda x: "images/" + 'exp_8/stat_' + x
    title_sample = f"Samples of {DISTRIBUTION_NAMES[dist_name]} Distribution"
    title_stat = f"R Statistic of {DISTRIBUTION_NAMES[dist_name]} Distribution"
    plot_ma_statistic(sample, title_sample, _save_sample(dist_name))
    plot_ma_statistic(stat, title_stat, _save_stat(dist_name))

    simulator = OrderSimulation(distribution, calculate_unscaled_kappa)
    per5 = int(n * 0.05)
    med = int(n * 0.5)
    per95 = int(n * 0.95)
    sim_5 = simulator(N, n, per5)
    sim_med = simulator(N, n, med)
    sim_95 = simulator(N, n, per95)
    _title_sim = lambda x: f'Distribution of $\kappa$ conditioned on $m={x}$'
    _save_sim = lambda m, dist_name: "images/" + "exp_8/" + f'Hist_m{m}_{dist_name}'
    plot_simulation_histogram(sim_5, 100, _title_sim(per5),
                              _save_sim('5', dist_name))
    plot_simulation_histogram(sim_med, 100, _title_sim(med),
                              _save_sim('med', dist_name))
    plot_simulation_histogram(sim_95, 100, _title_sim(per95),
                              _save_sim('95', dist_name))


@plotting_style('default')
def elementary_plots(N, dist, savedir="R_moving_plots/"):
    """Saves plots of sample distribution, R-statistic and outliers."""
    sample = np.sort(dist.rvs(N))
    stat = moving_average_unscaled_kappa(sample)
    threshold = scoring.get_kappa_threshold(stat, sensitivity=5)
    index = scoring.get_cut_off_index(stat, threshold)
    savedir = savedir if savedir.endswith("/") else savedir + "/"
    _save_sample = lambda x: "images/" + savedir + x.dist.name
    _save_stat = lambda x: "images/" + savedir + x.dist.name + "_stat"
    _save_threshold = lambda x: savedir + x.dist.name + "_threshold"
    title_sample = f"Samples of {DISTRIBUTION_NAMES[dist.dist.name]} Distribution"
    title_stat = f"R Statistics of {DISTRIBUTION_NAMES[dist.dist.name]} Distribution"
    kwargs = {
        "title": "R statistic",
        "kappa": np.round(threshold, 3),
        "m": index,
        "name": _save_threshold(dist)
    }
    plot_ma_statistic(sample, title_sample, _save_sample(dist))
    plot_ma_statistic(stat, title_stat, _save_stat(dist))
    plot_anomalies(stat, stat > threshold, **kwargs)


if False and __name__ == '__main__':
    # Plots of the statistic for a few distributions - Not for overleaf
    N = 1_000_000
    SAVE_DIR = "R_moving_plots/"
    normal = stats.halfnorm()
    cauchy = stats.halfcauchy()
    chisq = stats.chi2(1)
    elementary_plots(N, normal, SAVE_DIR)
    elementary_plots(N, cauchy, SAVE_DIR)
    elementary_plots(N, chisq, SAVE_DIR)

if False and __name__ == '__main__':
    # For Experiments in overleaf - Simulation plots
    # 5, med, and 95 percentile plots
    N = 10_000
    n = 1_000
    normal = stats.halfnorm()
    expon = stats.expon()
    pareto = stats.pareto(0.4)
    dic = {"lambda_": 1, "k": 100, "b": 3}
    identified_outliers = Identified_Outliers_Model(**dic)
    exploratory_plots(normal, N=N, n=n)
    exploratory_plots(expon, N=N, n=n)
    exploratory_plots(identified_outliers, N=N, n=n)

if False and __name__ == '__main__':
    # For Experiments 8 in overleaf - Simulation plots
    N = 3000
    n = 1000
    normal = stats.halfnorm()
    expon = stats.expon()
    pareto = stats.pareto(0.4)
    dic = {"lambda_": 1, "k": 100, "b": 3}
    identified_outliers = Identified_Outliers_Model(**dic)

    style = ['seaborn-talk']
    with plt.style.context(style):
        analyse_distribution(N, n, normal)
        analyse_distribution(N, n, expon)
        analyse_distribution(N, n, pareto)
        analyse_distribution(N, n, identified_outliers)

if False and __name__ == '__main__':
    # For Experiments in overleaf - Distinguishing Two Pareto Tails:
    # Kappa threshold as a change point over which the > .90 of observations
    # belong to the heavier tail.
    N = 1_000_000
    n_trials = 1000
    thin_tail = 1.5
    heavy_tails = {2.5, 2.3, 2.1, 1.9, 1.7}
    pareto_ = random.pareto(thin_tail, N)
    _, scores_sorted = scoring.get_anomaly_scores(pareto_)
    threshold = scoring.get_kappa_threshold(scores_sorted)

    def _tail_trial(tail):
        return np.array([
            mixed_pareto_sample(N=N,
                                thin_tail=thin_tail,
                                heavy_tail=tail,
                                threshold=threshold) for _ in range(n_trials)
        ])

    results = {tail: _tail_trial(tail) for tail in heavy_tails}
