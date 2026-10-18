# W1 estimator of App. G (exact, unregularised OT between jointly z-scored structural vectors), as in
# src/w1_robust.py: pooled z-score statistics of the three measures of one data block (sd == 0 -> 1); seed s:
# rng = default_rng(s) per pair; within-measure = entries 0..N-1 and N..2N-1 of one permutation (two disjoint
# samples); between = two independent draws without replacement; Euclidean cost; ot.emd2 with uniform weights.
# For the nested projections, the subsample indices of a (seed, pair, N) are drawn once and reused for every d,
# and Phi_d uses the first d (z-scored) columns, so every d is evaluated on the same trajectories.
import time

import numpy as np
import ot
from scipy.spatial.distance import cdist

import common as C


def zstats(arrs):
    allv = np.vstack(arrs)
    mu, sd = allv.mean(0), allv.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def block_data(block):
    """z-scored 34-column arrays of the three measures of a data block and the z statistics"""
    src = {"pool": "pool", "nsweep": "pool", "fresh": "fresh", "eval": "eval"}[block]
    F = {m: C.load_feats(src, m)["X"] for m in C.MEASURES}
    mu, sd = zstats([F[m] for m in C.MEASURES])
    return {m: (F[m] - mu) / sd for m in C.MEASURES}, mu, sd


def draw(seed, na, nb, N, same):
    rng = np.random.default_rng(seed)
    if same:
        idx = rng.permutation(na)
        return idx[:N], idx[N:2 * N]
    return rng.choice(na, N, replace=False), rng.choice(nb, N, replace=False)


def emd(s1, s2, num_iter=C.NUM_ITER):
    t0 = time.time()
    M = cdist(s1, s2)
    w1, w2 = np.ones(len(s1)) / len(s1), np.ones(len(s2)) / len(s2)
    cost, log = ot.emd2(w1, w2, M, numItermax=num_iter, log=True)
    return float(cost), int(log["result_code"]), round(time.time() - t0, 2)


def w1_dims(Za, Zb, i1, i2, dims=C.DIMS, num_iter=C.NUM_ITER):
    """{d: (w1, result_code, seconds)} on the same subsamples for every Phi_d"""
    A, B = Za[i1], Zb[i2]
    return {d: emd(A[:, :d], B[:, :d], num_iter) for d in dims}
