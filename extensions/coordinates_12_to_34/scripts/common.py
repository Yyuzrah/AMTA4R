# Shared constants and helpers of the 12 -> 34 coordinate experiment (App. M.1.1).
# Inputs are read from the repository (data/experiment_500k, gbc/, results/w1_robust.json); every output goes under
# BASE ($COORD_BASE, default outputs/coordinates_12_to_34 at the repository root).
# COORD_QUICK=1 runs every step at reduced scale (settings marked "reduced-scale run" below) in a subdirectory of BASE.
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
E = f"{ROOT}/outputs/coordinates_12_to_34"
SMOKE = os.environ.get("COORD_QUICK", "0") == "1"
BASE = os.environ.get("COORD_BASE") or (f"{E}/smoke" if SMOKE else E)
GBC = f"{ROOT}/gbc"
DATA_500K = f"{ROOT}/data/experiment_500k"
PAPER_W1 = f"{ROOT}/results/w1_robust.json"          # App. G values (exact_W1_zscored, mean and sd over 10 seeds)
APPJ_RESULTS = {"f1": f"{GBC}/results_f1.json", "f2": f"{GBC}/results_f2.json"}
GBC_FEATURES = f"{GBC}/features.npz"
TOKENIZER = "EleutherAI/gpt-neo-1.3B"

MEASURES = ["bfs", "dfs", "mixed"]
N_FRESH = {"bfs": 16667, "dfs": 16667, "mixed": 16666}  # App. J sample (first 16,667/16,667/16,666)
SOURCES = ["fresh", "pool", "eval"]                     # featurize.py task index = 3 * source + measure
POOL_SIZE, POOL_TOTAL, POOL_SEED = 20000, 500000, 42    # App. G pool (random.seed(42); random.sample)
MODELS = {"f1": "f1_small_deep_mixed_rope", "f2": "f2_large_deep_bfs_ape"}
APPJ_ACC = {"f1": 0.50146, "f2": 0.496}
APPJ_N = {"f1": 50000, "f2": 3000}

DIMS = [12, 16, 24, 34]                                 # nested projections Phi_d
N_CV_REPS = 2 if SMOKE else 10                          # repeated 5-fold CV (split seed = classifier seed = s)
W1_SEEDS = 2 if SMOKE else 10                           # App. G: 10 subsamples
W1_N = 2000                                             # App. G: N per side
W1_BLOCKS = ["pool", "fresh", "eval", "nsweep"]         # w1_task.py task index = W1_SEEDS * block + seed
W1_N_BLOCK = {"pool": 2000, "fresh": 300 if SMOKE else 2000, "eval": 1000}
NSWEEP_N = [250, 500, 1000, 4000]                       # plus N = 2000 from block "pool"
NUM_ITER = 2_000_000                                    # as in src/w1_robust.py
NUM_ITER_RESCUE = 100_000_000                           # w1_rescue.py: second run of non-optimal OT problems
PAIRS = [("bfs", "bfs", True), ("dfs", "dfs", True), ("mixed", "mixed", True),
         ("bfs", "mixed", False), ("bfs", "dfs", False), ("dfs", "mixed", False)]
PAPER_PAIRS = {"BFS-BFS": ("bfs", "bfs"), "DFS-DFS": ("dfs", "dfs"), "BFS-MIXED": ("bfs", "mixed"),
               "BFS-DFS": ("bfs", "dfs"), "DFS-MIXED": ("dfs", "mixed")}
SMOKE_FRESH_LIMIT = 1000                                # reduced-scale run: first 1,000 fresh trajectories per measure

# thresholds of the AUC check in analyze.py and of the checks in gate.py
AUC_DELTA = 0.01
G1_TOL = 1e-4
G3_TOL = 0.002
G4_TOL = 0.01


def pair_name(a, b):
    return f"{a.upper()}-{b.upper()}"


def n_auc_tasks():
    return len(MODELS) * N_CV_REPS


def n_w1_tasks():
    return len(W1_BLOCKS) * W1_SEEDS


def phi_order():
    """34 coordinate names in nesting order: chi_1..chi_12, then the 22 further coordinates in the class order of
    ontology_countdown.NAMES (classes TOP, CTRL, TERM, MEM, NEST, INTER, SURF). Phi_d = the first d names."""
    from ontology_countdown import NAMES, F12_NAMES
    new = [n for n in NAMES if n not in F12_NAMES]
    order = list(F12_NAMES) + new
    assert len(order) == 34 and len(set(order)) == 34
    return order


def phi_cols(d):
    """column indices (into the 34-column feature matrix stored in phi_order()) of Phi_d"""
    assert d in DIMS
    return list(range(d))


def problem_key(target, nums):
    return f"{int(target)}:{sorted(int(x) for x in nums)}"


def jload(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        if default is not None:
            return default
        raise


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def jdump(obj, p, indent=1):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_default)
    os.replace(tmp, p)


def savez(p, **kw):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, p)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def feat_path(source, m):
    return f"{BASE}/data/features/{source}_{m}.npz"


def load_feats(source, m):
    z = np.load(feat_path(source, m), allow_pickle=False)
    return {k: z[k] for k in z.files}


def labels(model_key):
    """per measure: (idx array, correct array) of the App. J labels; reduced-scale run: idx < SMOKE_FRESH_LIMIT only"""
    import glob
    out = {}
    for m in MEASURES:
        y = {}
        for p in sorted(glob.glob(f"{GBC}/labels/{MODELS[model_key]}_{m}*.jsonl")):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    y[r["idx"]] = int(r["correct"])
        idx = np.array(sorted(y), dtype=np.int64)
        if SMOKE:
            idx = idx[idx < SMOKE_FRESH_LIMIT]
        out[m] = (idx, np.array([y[i] for i in idx], dtype=np.int64))
    return out


def labelled_data(model_key):
    """X (n x 34, phi_order), y (= correct, the coding of gbc/analyze.py), operand count, problem key, measure;
    rows in App. J order (bfs, dfs, mixed; idx ascending)"""
    L = labels(model_key)
    X, y, ops, keys, meas = [], [], [], [], []
    for m in MEASURES:
        idx, ym = L[m]
        if len(idx) == 0:
            continue
        F = load_feats("fresh", m)
        pos = {int(s): r for r, s in enumerate(F["src_idx"])}
        rows = np.array([pos[int(i)] for i in idx])
        X.append(F["X"][rows]); y.append(ym); ops.append(F["ops"][rows]); keys.append(F["key"][rows])
        meas += [m] * len(idx)
    return np.concatenate(X), np.concatenate(y), np.concatenate(ops), np.concatenate(keys), np.array(meas)
