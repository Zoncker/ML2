import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# % matplotlib inline
# % precision 4


# plt.style.use('ggplot')
# np.set_printoptions(formatter={'float': lambda x: '%.3f' % x})
# np.random.seed(123)
# # Probability weights can be given
# np.random.choice(4, 12, p=[.4, .1, .1, .4])
# # Sampling is done with replacement by default
# # np.random.choice(4, 12)
# x = np.random.randint(0, 10, (8, 12))
#
# For example, what is the 95% confidence interval for
# the mean of this data set if you didn't know how it was generated?

x = np.concatenate([np.random.exponential(size=200), np.random.normal(size=100)])
plt.hist(x, 25, histtype='step')
plt.show()
n = len(x)
reps = 10000
xb = np.random.choice(x, (n, reps))
mb = xb.mean(axis=0)
mb.sort()

np.percentile(mb, [2.5, 97.5])
# array([0.483, 0.740])
# Reprise of bootstrap example for Monte Carlo integration


def f(x):
    return x * np.cos(71*x) + np.sin(13*x)


# data sample for integration
n = 100
x = f(np.random.random(n))
# bootstrap MC integration
reps = 1000
xb = np.random.choice(x, (n, reps), replace=True)
yb = 1/np.arange(1, n+1)[:, None] * np.cumsum(xb, axis=0)
upper, lower = np.percentile(yb, [2.5, 97.5], axis=1)
plt.plot(np.arange(1, n+1)[:, None], yb, c='grey', alpha=0.02)
plt.plot(np.arange(1, n+1), yb[:, 0], c='red', linewidth=1)
plt.plot(np.arange(1, n+1), upper, 'b', np.arange(1, n+1), lower, 'b')
plt.show()
