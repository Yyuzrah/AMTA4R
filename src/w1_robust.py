# W1 estimates of App. G: exact W1 (ot.emd2) on pooled z-scored features, pool of 20k trajectories per measure,
# N=2000 per side, 10 seeds. Also the Sinkhorn cost (reg 0.1, cost divided by its max), the debiased Sinkhorn
# divergence, and within-measure baselines (two disjoint samples of one measure). Run from src/.
import json, numpy as np, ot
from scipy.spatial.distance import cdist
from distribution_metrics import load_and_featurize
D = "../data/experiment_500k"
F = {k: load_and_featurize(f"{D}/train_{k}.json", k.upper(), max_process=20000) for k in ["bfs", "dfs", "mixed"]}
allv = np.vstack(list(F.values())); mu, sd = allv.mean(0), allv.std(0); sd[sd == 0] = 1
Z = {k: (v - mu) / sd for k, v in F.items()}
def pair(a, b, rng, N=2000, same=False):
    if same:  # two disjoint samples of size N from one pool
        idx = rng.permutation(len(a)); ia, ib = idx[:N], idx[N:2 * N]; s1, s2 = a[ia], a[ib]
    else:
        s1 = a[rng.choice(len(a), N, replace=False)]; s2 = b[rng.choice(len(b), N, replace=False)]
    M = cdist(s1, s2); w = np.ones(N) / N
    exact = float(ot.emd2(w, w, M, numItermax=2_000_000))
    Mn = M / (M.max() + 1e-9)
    sk = float(ot.sinkhorn2(w, w, Mn, 0.1))
    s11 = float(ot.sinkhorn2(w, w, cdist(s1, s1) / (M.max() + 1e-9), 0.1)); s22 = float(ot.sinkhorn2(w, w, cdist(s2, s2) / (M.max() + 1e-9), 0.1))
    return exact, exact / (M.max() + 1e-9), sk, sk - 0.5 * (s11 + s22)
out = {}
for name, (a, b, same) in {"BFS-BFS": ("bfs", "bfs", True), "DFS-DFS": ("dfs", "dfs", True), "BFS-MIXED": ("bfs", "mixed", False),
                           "BFS-DFS": ("bfs", "dfs", False), "DFS-MIXED": ("dfs", "mixed", False)}.items():
    r = np.array([pair(Z[a], Z[b], np.random.default_rng(s), same=same) for s in range(10)])
    out[name] = {m: [float(r[:, i].mean()), float(r[:, i].std(ddof=1))] for i, m in
                 enumerate(["exact_W1_zscored", "exact_W1_maxnormalized", "sinkhorn_cost_reg0.1", "sinkhorn_divergence_debiased"])}
    print(name, json.dumps(out[name]))
json.dump(out, open(f"{D}/w1_robust.json", "w"), indent=2)
