# Features of the reference trajectories: F12 = chi (src/distribution_metrics.py), LEN (GPT-Neo token count),
# PROB (target, operand count, sorted operands), plus the training-overlap flag. Uses the main environment.
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, f"{ROOT}/src")
from distribution_metrics import extract_structural_features

N = {"bfs": 16667, "dfs": 16667, "mixed": 16666}
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
out = {}
for mi, m in enumerate(["bfs", "dfs", "mixed"]):
    d = json.load(open(f"{ROOT}/gbc/data/fresh_{m}.json"))[:N[m]]
    out[f"{m}_F12"] = np.stack([extract_structural_features(x["search_path"]) for x in d]).astype(np.float64)
    out[f"{m}_LEN"] = np.array([[len(tok(x["search_path"]).input_ids)] for x in d], dtype=np.float64)
    out[f"{m}_PROB"] = np.array([[x["target"], len(x["nums"])] + (sorted(x["nums"]) + [0] * 4)[:4] for x in d], dtype=np.float64)
    out[f"{m}_in_any_train"] = np.array([x["in_any_train"] for x in d])
    out[f"{m}_search_type"] = np.array([x["search_type"] for x in d])
    print(m, out[f"{m}_F12"].shape, "chi10 mean %.3f" % out[f"{m}_F12"][:, 9].mean(), "in_any_train %.3f" % out[f"{m}_in_any_train"].mean(), flush=True)
# Exploratory: chi of the models' own generations, keyed by (model, measure) and aligned by idx
import glob
for pat in ["f1_small_deep_mixed_rope", "f2_large_deep_bfs_ape"]:
    for m in ["bfs", "dfs", "mixed"]:
        rows = {}
        for p in glob.glob(f"{ROOT}/gbc/labels/{pat}_{m}*.jsonl"):
            for line in open(p):
                r = json.loads(line); rows[r["idx"]] = r["gen"]
        if rows:
            idx = np.array(sorted(rows))
            out[f"{pat}_{m}_G_idx"] = idx
            out[f"{pat}_{m}_G12"] = np.stack([extract_structural_features(rows[i]) for i in idx]).astype(np.float64)
np.savez(f"{ROOT}/gbc/features.npz", **out)
