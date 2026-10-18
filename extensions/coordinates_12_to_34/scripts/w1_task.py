# Step 4 (CPU; analysis environment). Task t in 0..39 -> block W1_BLOCKS[t // W1_SEEDS], seed t % W1_SEEDS.
#   pool   : App. G pools (training measures, 20,000 each), N = 2,000, six pairs (3 within, 3 between)
#   fresh  : App. J fresh trajectories (16,667/16,667/16,666), N = 2,000, six pairs
#   eval   : evaluation sets (2,000 each), N = 1,000, six pairs; seed 0 also the full 2,000-vs-2,000 between pairs
#   nsweep : App. G pools, within pairs only, N in {250, 500, 1000, 4000} (N = 2,000 is in block pool)
# Writes runs/w1/<block>_s<seed>.json (values, OT result codes, seconds) and .npz (subsample indices) under BASE.
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import w1lib as W  # noqa: E402

t = int(sys.argv[1])
block, seed = C.W1_BLOCKS[t // C.W1_SEEDS], t % C.W1_SEEDS
out = f"{C.BASE}/runs/w1/{block}_s{seed}"
t0 = time.time()
Z, mu, sd = W.block_data(block)
n = {m: len(Z[m]) for m in C.MEASURES}
jobs = []  # (a, b, same, N, tag)
if block == "nsweep":
    for N in C.NSWEEP_N:
        for a, b, same in C.PAIRS:
            if same and 2 * N <= n[a]:
                jobs.append((a, b, same, N, "sub"))
else:
    N = C.W1_N_BLOCK[block]
    for a, b, same in C.PAIRS:
        jobs.append((a, b, same, N, "sub"))
    if block == "eval" and seed == 0:
        for a, b, same in C.PAIRS:
            if not same:
                jobs.append((a, b, same, min(n[a], n[b]), "full"))
entries, idx_store = [], {}
for a, b, same, N, tag in jobs:
    if tag == "full":
        i1, i2 = np.arange(N), np.arange(N)
    else:
        i1, i2 = W.draw(seed, n[a], n[b], N, same)
    key = f"{C.pair_name(a, b)}_N{N}_{tag}"
    idx_store[key + "_i1"], idx_store[key + "_i2"] = i1.astype(np.int32), i2.astype(np.int32)
    res = W.w1_dims(Z[a], Z[b], i1, i2)
    for d, (w, code, sec) in res.items():
        entries.append({"pair": C.pair_name(a, b), "a": a, "b": b, "same": same, "N": int(N), "tag": tag, "d": d,
                        "w1": w, "code": code, "seconds": sec})
    print(f"{block} s{seed} {key}: " + " ".join(f"d{d}={res[d][0]:.4f}(c{res[d][1]})" for d in C.DIMS)
          + f" [{time.time() - t0:.0f}s]", flush=True)
C.savez(out + ".npz", **idx_store)
C.jdump({"block": block, "seed": seed, "n_pool": n, "zstats": {"mean": mu, "sd": sd}, "entries": entries,
         "num_iter": C.NUM_ITER, "seconds": round(time.time() - t0, 1), "time": C.now(), "smoke": C.SMOKE},
        out + ".json")
bad = [e for e in entries if e["code"] != 1]
print(f"w1 task {t} ({block}, seed {seed}): {len(entries)} values, {len(bad)} not optimal, "
      f"{time.time() - t0:.0f}s", flush=True)
