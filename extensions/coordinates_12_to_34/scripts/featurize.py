# Step 2 (CPU; main environment, for the GPT-Neo tokenizer). Task t in 0..8 -> source SOURCES[t // 3], measure
# MEASURES[t % 3]. Writes data/features/<source>_<m>.npz under BASE:
#   X         (n x 34, float64) the 34 coordinates in nesting order (common.phi_order()); columns 0..11 are
#             chi_1..chi_12 from src/distribution_metrics.extract_structural_features (float32 cast to float64, the
#             inputs of App. G and App. J); columns 12..33 are the 22 further coordinates of ontology_countdown.extract;
#   chi_recomputed (n x 12) chi_1..chi_12 recomputed by ontology_countdown.extract (check 1 of gate.py compares it
#             with columns 0..11);
#   ops, key (problem key target:sorted nums), src_idx (row in the source file / pool rank), pos (pool only:
#   position in train_<m>.json), in_any_train (fresh only).
# Sources: fresh = gbc/data/fresh_<m>.json (first 16,667/16,667/16,666; App. J); pool = data/raw/pool_<m>.jsonl under
# BASE (App. G pool); eval = data/experiment_500k/val_<m>.json (evaluation sets, 2,000 each).
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
sys.path.insert(0, f"{C.ROOT}/src")
from distribution_metrics import extract_structural_features as released  # noqa: E402
from ontology_countdown import extract, NAMES, F12_NAMES  # noqa: E402

t = int(sys.argv[1])
source, m = C.SOURCES[t // 3], C.MEASURES[t % 3]
t0 = time.time()
if source == "fresh":
    inp = f"{C.GBC}/data/fresh_{m}.json"
    data = json.load(open(inp))[:C.N_FRESH[m]]
    if C.SMOKE:
        data = data[:C.SMOKE_FRESH_LIMIT]
elif source == "pool":
    inp = f"{C.BASE}/data/raw/pool_{m}.jsonl"
    data = [json.loads(line) for line in open(inp)]
    assert len(data) == C.POOL_SIZE, len(data)
else:
    inp = f"{C.DATA_500K}/val_{m}.json"
    data = json.load(open(inp))
print(f"featurize {source}/{m}: {len(data)} trajectories from {inp} (load {time.time() - t0:.0f}s)", flush=True)

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(C.TOKENIZER)
order = C.phi_order()
new_names = order[12:]
X = np.zeros((len(data), 34), dtype=np.float64)
chi_recomputed = np.zeros((len(data), 12), dtype=np.float64)
for i, x in enumerate(data):
    sp = x["search_path"]
    X[i, :12] = np.asarray(released(sp), dtype=np.float32).astype(np.float64)
    f = extract(sp, tok)
    X[i, 12:] = [f[n] for n in new_names]
    chi_recomputed[i] = [f[n] for n in F12_NAMES]
    if (i + 1) % 5000 == 0:
        print(f"  {i + 1} done, {time.time() - t0:.0f}s", flush=True)
ops = np.array([len(x["nums"]) for x in data], dtype=np.int64)
key = np.array([C.problem_key(x["target"], x["nums"]) for x in data])
extra = {}
if source == "fresh":
    extra["in_any_train"] = np.array([bool(x["in_any_train"]) for x in data])
if source == "pool":
    extra["pos"] = np.array([int(x["pos"]) for x in data], dtype=np.int64)
diff = float(np.abs(X[:, :12] - chi_recomputed).max()) if len(data) else float("nan")
finite = bool(np.isfinite(X).all())
C.savez(C.feat_path(source, m), X=X, chi_recomputed=chi_recomputed, ops=ops, key=key,
        src_idx=np.arange(len(data), dtype=np.int64), names=np.array(order), **extra)
C.jdump({"source": source, "measure": m, "input": inp, "n": len(data), "max_abs_chi_diff": diff, "finite": finite,
         "n_names": len(order), "seconds": round(time.time() - t0, 1), "time": C.now(), "smoke": C.SMOKE},
        C.feat_path(source, m).replace(".npz", ".meta.json"))
print(f"featurize {source}/{m}: n={len(data)} max|chi_released - chi_recomputed|={diff:.2e} finite={finite} "
      f"{time.time() - t0:.0f}s", flush=True)
assert set(NAMES) == set(order)
