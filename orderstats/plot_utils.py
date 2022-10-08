"""Plotting functions."""

import functools

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay

from orderstats.utils import scotts_rule

PLOTTING_STYLE = 'default'


def _save_or_show(filepath):
    if filepath is not None:
        plt.savefig(filepath)
        plt.clf()
    else:
        plt.show()


def plotting_style(matplotlib_style):
    """Decorater to wrap a plotting function with a style."""

    def style_decorator(plot_func):

        @functools.wraps(plot_func)
        def plot_with_style(*args, **kwargs):
            with plt.style.context(matplotlib_style):
                plot_func(*args, **kwargs)

        return plot_with_style

    return style_decorator


@plotting_style(PLOTTING_STYLE)
def plt_plot(*args, scalex=True, scaley=True, data=None, **kwargs):
    """Wrapper for `plt.plot` using `plotting_style` and optional saving."""
    plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)
    save = kwargs.get('save', None)
    _save_or_show(save)


@plotting_style(PLOTTING_STYLE)
def plot_roc_curve(fpr, tpr, roc_auc, save=None, **kwargs):
    name = kwargs.get('name')
    _, ax = plt.subplots(1)
    viz = RocCurveDisplay(fpr=fpr,
                          tpr=tpr,
                          roc_auc=roc_auc,
                          estimator_name=name,
                          pos_label=None)
    viz.plot(ax=ax, name=name)
    plt.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    plt.title('Receiver Operating Characteristic')
    _save_or_show(save)


def plot_curve_fit(xdata, ydata, func, popt):
    """Plots results of `scipy.optimize` curve_fit().

    Example:
    >>> sim = simulate_exp(10000, 100, 80)
    >>> probabilities, ls = tail_decay(sim, 0.15, 0.3, 0.0001)
    >>> popt, pcov = curve_fit(logistic_fit, ls, probabilities)
    >>> plot_curve_fit(ls, probabilities, logistic_fit, popt)
    """
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata,
             func(xdata, *popt),
             'r-',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.show()


def plot_ma_statistic(X, title='R Statistic', save=None):
    plt.plot(X)
    plt.title(title, usetex=True)
    plt.xlabel('Order Statistics')
    _save_or_show(save)


def plot_simulation_histogram(errors, n_bins, title=None, save=None):
    _ = plt.hist(np.asarray(errors), n_bins, density=True, facecolor='#f4810c')
    plt.plot()
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.grid(True)
    plt.title(title)
    _save_or_show(save)


def scatter_plot_anomalies(X, threshold):
    X = np.asarray(X)
    plt.scatter(np.arange(len(X)), X)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.show()


def plot_anomalies(df, predictions, **kwargs):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    predictions = np.asarray(predictions)
    df_subset = df[(predictions == 1)]
    _, ax = plt.subplots()
    df.plot(legend=False, ax=ax, color='deepskyblue')
    df_subset.plot(legend=False, ax=ax, color='#E3242B')
    labels = ['Normal', 'Anomalous']
    plt.legend(labels, ncol=1)
    plt.xlabel('i-th order statistics of the sample')
    plt.ylabel('Value of the statistic')
    if kwargs:
        plt.title(f"Moving Average of {kwargs['title']} across sample")
        empty_patch = patches.Patch(color='none')
        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(empty_patch)
        handles.append(empty_patch)
        labels.append(f'$\kappa$ threshold value is: {kwargs["kappa"]}')  # pylint: disable=anomalous-backslash-in-string
        labels.append(f'Cut-off point: {kwargs["m"]}')
        plt.legend(handles, labels)
    save_path = kwargs['name'] if kwargs.get('name') else None
    _save_or_show(save_path)


def plot_scores_vs_sample(scores, sample, predictions=None):
    df = pd.DataFrame({'Scores': scores}, index=sample)
    _, ax = plt.subplots()
    df.plot(legend=False, ax=ax)
    if predictions is not None:
        df_subset = df[(predictions == 1)]
        df_subset.plot(legend=False, ax=ax, color='r')
    plt.xlabel('i-th order statistics of the distribution')
    plt.ylabel('Value of the statistic')
    plt.title('Anomaly score vs Order Statistics')
    plt.grid(True)
    plt.show()


def _get_simulation_stats(sim):
    return np.min(sim), np.max(sim), len(sim)


def plot_pdf(sim, pdf_func, n_bins='auto'):
    """Plots the pdf of the estimated distribution from density estimation.

    Args:
        sim: The sample from which the density was estimated.
        pdf_func: Function providing point estimate of the pdf.
        n_bins: Number of bins for the histogram. Uses scott's rule if 'auto'.
    """
    if n_bins == 'auto':
        n_bins = scotts_rule(sim)
    _, ax = plt.subplots(figsize=(8, 4))
    # plot the cumulative histogram
    _, bins, _ = ax.hist(sim,
                         n_bins,
                         density=True,
                         cumulative=False,
                         label='Simulation Histogram')
    # Add a line showing the expected distribution.
    sim_min, sim_max, _ = _get_simulation_stats(sim)
    x_d = np.linspace(sim_min, sim_max, n_bins + 1)
    y = [pdf_func(i) for i in x_d]
    ax.plot(bins, y, 'k--', linewidth=1.5, label='Estimated pdf')
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Probability density function')
    plt.show()


def plot_cdf(sim, cdf_func, n_bins='auto'):
    """Plots the cdf of the estimated distribution from density estimation.

    Args:
        sim: The sample from which the density was estimated.
        cdf_func: Function providing point estimate of the cdf.
            get_cdf_from_kernel_estimate() in density_estimation can be used.
        n_bins: Number of bins for the histogram. Uses scott's rule if 'auto'.
    """
    if n_bins == 'auto':
        n_bins = scotts_rule(sim)
    _, ax = plt.subplots(figsize=(8, 4))
    # plot the cumulative histogram
    _, bins, _ = ax.hist(sim,
                         n_bins,
                         density=True,
                         cumulative=True,
                         label='Simulation Histogram')
    # Add a line showing the expected distribution.
    sim_min, sim_max, _ = _get_simulation_stats(sim)
    x_d = np.linspace(sim_min, sim_max, n_bins + 1)
    y = [cdf_func(i) for i in x_d]
    ax.plot(bins, y, 'k--', linewidth=1.5, label='Estimated cdf')
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative distribution function')
    plt.show()
