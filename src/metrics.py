"""Generic distributional-distance metrics, independent of any toy
problem's geometry -- shared by every experiment's coverage evaluation
(main.typ's "preservation of the true intrinsic distribution" axis).
"""

import numpy as np


def wasserstein_1d(a, b):
    """Exact 1-D Wasserstein-1 distance between two equal-size empirical
    samples: mean absolute gap between sorted order statistics."""
    n = min(len(a), len(b))
    return float(np.mean(np.abs(np.sort(a)[:n] - np.sort(b)[:n])))


def binary_js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two Bernoulli distributions with
    success probabilities p and q."""
    p, q = np.clip([p, 1 - p], eps, 1), np.clip([q, 1 - q], eps, 1)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))
