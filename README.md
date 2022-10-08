# OrderStatistics
A repository focused on outlier detection and analysis of heavy-tailed distributions using order statistics. Here you can find:
- Simualations of order statistics for outlier detection and the statistic provided in our paper titled [On a Notion of Outliers Based on Ratios of Order Statistics](https://arxiv.org/abs/2207.13068).
- Single and double bootstrap methods (& more) for tail index estimation of heavy-tailed data.
- Special kernel density estimation methods for heavy-tailed data. 

Also:
- Useful plotting functions for outlier detection.
- Easy reporting tools for binary classification such as `get_classification_report()`, which reports all classification metrics from confusion matrix to AUC and can plot ROC curve.

## Installation:
You can install the repository with:
```
pip install orderstats
```

Alternatively, you can download a copy of the repository from this page.
After downloading, you can do 
  `pip install -r requirements.txt`
to install the requirements.

## General explanation about Random Variables:
  We use `scipy.stats` package for generating random variables.
  A random variable instance can be created just by giving the appropriate parameters.
  For example for X ~ N(0, 1), we can do: <br>
  `>>> X = stats.norm(0, 1)` <br>
  Once an instance is created, we can calculate pdf, or cdf using `.pdf()`, `.cdf()` methods, or,
  we can take a sample using `.rvs()`: <br>
  `>>> X.pdf(1.96)` for pdf of X ~ N(0, 1) at x = 1.96; <br>
  `>>> X.cdf(1.96)` for cdf of X ~ N(0, 1) at x = 1.96;  <br>
  `>>> X.rvs(1000)` for an i.i.d sample of 1000 from X ~ N(0, 1). <br>

#### Some problems:
  Not every random variable function in `scipy.stats` is intuitive. <br>
  For instance `expon` function creates an instance of an exponential distribution.
  However, if we wish to get an instance of exponential distribution with lambda = 2
  (i.e. with pdf f(x) = 2e^(-2x)), then we would need to use `expon(0, 1/2)`.
  In distributions.py, there are examples given for some popular distributions to clarify any ambiguities.

## Examples

### Outlier Detection:
Here is an example use of our method for a 1D dataset `X`:

```python
from orderstats import scoring
from orderstats.distributions import moving_average_unscaled_kappa
from orderstats.plot_utils import plot_anomalies
scores, scores_sorted = scoring.get_anomaly_scores(X, scoring_func=moving_average_unscaled_kappa)
threshold = scoring.get_kappa_threshold(scores_sorted)
predictions = scores > threshold
plot_anomalies(X, predictions=predictions)
```

### Order Statistics Simulation:
  In general, we will use `OrderSimulation` class in distributions for simulations. Any random variable from the `scipy.stats` package can be given to this class as an argument.
  For example, if you wish to simulate the sums of first `m` order statistics from a sample of exponential distribution of size `n`:

```python
from orderstats import OrderSimulation
simulate_normal_dist = OrderSimulation(stats.expon(0, 1), calculate_S_m)
simulation = simulate_normal_dist(10000, n, m)
```

  For studying change point detection, we provide the `MixSimulation` class. For getting a mixed sample with corresponding ids:
```python
from orderstats import MixSimulation
simulate_mixture = MixSimulation(dist1=expon(0, 1), dist2=stats.pareto(2.))
mixed_array, idx = simulate_mixture(n1, n2)
```

### Tail Index Estimation:
Most of the methods for tail index estimation mentioned in [1] is implemented in `tail_estimation`.
As an example for the double bootstrap method:

```python
import numpy as np
N = 1000
pareto_sample = np.random.pareto(2, 1000)
sample_to_estimate_index = np.sort(X,)[::-1] # Sort decreasing
double_bootsrap = DoubleBootstrap()
tail_index = double_bootstrap(N, sample_to_estimate_index)
```

# References:
  [1] Markovich, N. (2008). Nonparametric analysis of univariate heavy-tailed data: research and practice (Vol. 753). John Wiley & Sons.