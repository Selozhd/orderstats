# OrderStatistics
Here I will try to keep general explanations about how different parts of the code works.

## Installation:
You can download a copy of the repository from this page.
After downloading, you can do 
> `pip install -r requirements.txt`
within the downloaded repository to install the requirements.

For installing jax:
> `pip install --upgrade jax jaxlib`

## General explanation about RVs:
  We are going to be using `scipy.stats` package for generating random variables.
  A random variable instance can be created just by giving the appropriate parameters.
  For example for X ~ N(0, 1), we can do: <br>
  `>>> X = stats.norm(0, 1)` <br>
  Once an instance is created, we can calculate pdf, or cdf using `.pdf()`, `.cdf()` methods, or,
  we can take a sample using `.rvs()`: <br>
  `>>> X.pdf(1.96)` for pdf of X ~ N(0, 1) at x = 1.96; <br>
  `>>> X.cdf(1.96)` for cdf of X ~ N(0, 1) at x = 1.96;  <br>
  `>>> X.rvs(1000)` for an i.i.d sample of 1000 from X ~ N(0, 1). <br>

#### Some problems:
  Not every random variable function is intuitive. <br>
  For instance `expon` function creates an instance of an exponential distribution.
  However, if we wish to get an instance of exponential distribution with lambda = 2
  (i.e. with pdf f(x) = 2e^(-2x)), then we would need to use `expon(0, 1/2)`.
  In distributions.py, there are examples given for some popular distributions to clarify any ambiguities.

#### OrderSimulation class:
  In general, we will use OrderSimulation class in distributions.py for simulations.
  Any random variable from the `scipy.stats` package can be given to this class as an argument.
  See class docstring for details.

## Tail Index Estimation:
  Most of the methods for tail index estimation mentioned in [1] is implemented in `tail_estimation`.

# References:
  [1] Markovich, N. (2008). Nonparametric analysis of univariate heavy-tailed data: research and practice (Vol. 753). John Wiley & Sons.