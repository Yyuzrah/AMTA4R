# Step 5 (CPU; analysis environment). Every W1 value whose network-simplex run did not reach optimality (POT
# result_code != 1, normally the numItermax = 2,000,000 limit of src/w1_robust.py) is recomputed once on the same
# subsample with numItermax = 100,000,000. Writes runs/w1_rescue.json under BASE; analyze.py uses the recomputed
# value when it is optimal and drops the value otherwise.
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import w1lib as W  # noqa: E402

t0 = time.time()
todo = []
files = sorted(glob.glob(f"{C.BASE}/runs/w1/*.json"))
for p in files:
    R = C.jload(p)
    for e in R["entries"]:
        if e["code"] != 1:
            todo.append((p, R["block"], R["seed"], e))
print(f"{len(files)} w1 files, {len(todo)} values to rescue", flush=True)
cache, out = {}, []
for p, block, seed, e in todo:
    if block not in cache:
        cache[block] = W.block_data(block)[0]
    Z = cache[block]
    idx = np.load(p.replace(".json", ".npz"))
    key = f"{e['pair']}_N{e['N']}_{e['tag']}"
    i1, i2 = idx[key + "_i1"], idx[key + "_i2"]
    w, code, sec = W.emd(Z[e["a"]][i1][:, :e["d"]], Z[e["b"]][i2][:, :e["d"]], C.NUM_ITER_RESCUE)
    out.append({"block": block, "seed": seed, "pair": e["pair"], "N": e["N"], "tag": e["tag"], "d": e["d"],
                "w1_before": e["w1"], "code_before": e["code"], "w1": w, "code": code, "seconds": sec})
    print(out[-1], flush=True)
C.jdump({"n_files": len(files), "n_rescued": len(out), "n_still_not_optimal": sum(o["code"] != 1 for o in out),
         "num_iter": C.NUM_ITER_RESCUE, "entries": out, "seconds": round(time.time() - t0, 1), "time": C.now()},
        f"{C.BASE}/runs/w1_rescue.json")
print("w1_rescue done", len(out), f"{time.time() - t0:.0f}s", flush=True)
