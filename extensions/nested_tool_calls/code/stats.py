"""Fit that may not increase with depth (pool adjacent violators), budget d_hat*, and bootstrap CI."""
import numpy as np


def pava_dec(y, w):
    """Weighted least-squares non-increasing fit."""
    y = [float(v) for v in y]
    w = [float(v) for v in w]
    blocks = []  # [mean, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = []
    for m, _, c in blocks:
        out += [m] * c
    return np.array(out)


def dhat(depths, acc, n, tau):
    """Largest depth whose fitted accuracy (pava_dec) is >= tau (0 if none)."""
    fit = pava_dec(acc, n)
    ok = [d for d, f in zip(depths, fit) if f >= tau - 1e-12]
    return (max(ok) if ok else 0), fit


def dhat_boot(depths, correct_by_depth, tau, reps=2000, seed=0):
    """correct_by_depth: list of 0/1 arrays (one per depth). Percentile CI of d_hat*."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        acc, n = [], []
        for c in correct_by_depth:
            c = np.asarray(c)
            s = c[rng.integers(0, len(c), len(c))]
            acc.append(s.mean()); n.append(len(c))
        vals.append(dhat(depths, acc, n, tau)[0])
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals
