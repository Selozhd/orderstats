"""Anomaly scoring."""

import numpy as np
from kneed import KneeLocator

from orderstats.distributions import moving_average_unscaled_kappa


def get_anomaly_scores(X, scoring_func=moving_average_unscaled_kappa):
    """Calculates the unscaled anomaly statistic for each element in X."""
    X = np.asarray(X)
    sorting_idx = np.argsort(X)
    sample_sorted = X[sorting_idx]
    scores_sorted = scoring_func(sample_sorted)
    scores = scores_sorted[sorting_idx.argsort()]
    return scores, scores_sorted


def get_kappa_threshold(scores_sorted, sensitivity=5.0):
    knee = KneeLocator(np.where(scores_sorted[:-1])[0],
                       scores_sorted[:-1],
                       S=sensitivity,
                       curve="convex",
                       direction="increasing")
    threshold = knee.all_knees_y[0]
    return threshold


def get_cut_off_index(scores_sorted, threshold):
    predictions = scores_sorted > threshold
    m = len(scores_sorted) - np.sum(predictions)
    return m
