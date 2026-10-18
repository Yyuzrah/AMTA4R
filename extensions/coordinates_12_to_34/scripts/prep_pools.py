# Step 1 (CPU, one run per training measure; analysis environment, which has ijson).
# Streams data/experiment_500k/train_<m>.json (500,000 trajectories) once and writes data/raw/pool_<m>.jsonl under
# BASE: the pool of App. G, i.e. the 20,000 trajectories chosen by distribution_metrics.load_and_featurize
# (random.seed(42); random.sample(data, 20000)), in that order. random.sample draws positions from len(population)
# only, so sampling range(500000) selects the same trajectories in the same order; gate.py (G4) checks that chi on
# these pools reproduces the W1 values of App. G.
import argparse
import json
import os
import random
import sys
import time

import ijson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--measure", required=True, choices=C.MEASURES)
a = ap.parse_args()

t0 = time.time()
path = os.path.join(C.DATA_500K, f"train_{a.measure}.json")
random.seed(C.POOL_SEED)
pool_idx = random.sample(range(C.POOL_TOTAL), C.POOL_SIZE)
want = {j: r for r, j in enumerate(pool_idx)}
recs = [None] * len(pool_idx)
n = 0
with open(path, "rb") as f:
    for it in ijson.items(f, "item", use_float=True):
        r = want.get(n)
        if r is not None:
            recs[r] = {"pos": n, "target": int(it["target"]), "nums": [int(x) for x in it["nums"]],
                       "search_path": it["search_path"]}
        n += 1
assert n == C.POOL_TOTAL, (n, C.POOL_TOTAL)
assert all(r is not None for r in recs)
out = f"{C.BASE}/data/raw/pool_{a.measure}.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out + ".tmp", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
os.replace(out + ".tmp", out)
C.jdump({"measure": a.measure, "source": path, "n_items": n, "pool_size": len(recs), "seed": C.POOL_SEED,
         "first_positions": pool_idx[:5], "seconds": round(time.time() - t0, 1), "time": C.now()},
        f"{C.BASE}/data/raw/pool_{a.measure}.meta.json")
print(f"{a.measure}: {n} items streamed, pool {len(recs)} written to {out}, {time.time() - t0:.0f}s", flush=True)
