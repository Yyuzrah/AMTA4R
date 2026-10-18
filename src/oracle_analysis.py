# Statistics of the reference trajectories (the three evaluation sets and a sample of each training set):
#  - oracle accuracy: success of the reference trajectory itself under metric_fn, uncapped and when
#    truncated to what the model can see/emit (1024-token training context; 512 new tokens at eval)
#  - token-length distribution; share of trajectories > 1024 tokens
#  - chi_1 (max depth) counts and mean chi_6 (backtracking), from extract_structural_features
import argparse, json, os, random, collections, statistics as st
import numpy as np
from transformers import AutoTokenizer
from countdown_utils import metric_fn
from distribution_metrics import extract_structural_features

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/experiment_500k")
p.add_argument("--n_train", type=int, default=20000)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
out = {}

def analyse(name, data):
    lens, unc, cap512, ctx1024, chi1, chi6 = [], [], [], [], [], []
    for s in data:
        text = s["search_path"].strip()
        prompt = f"Current State: {s['target']}:{s['nums']}, Operations: []\n"
        ids = tok(text)["input_ids"]
        lens.append(len(ids) + 2)  # + BOS/EOS as in training
        unc.append(int(metric_fn(text)[0] > 0))
        # eval: prompt (~ first line) + at most 512 generated tokens
        p_len = len(tok(prompt)["input_ids"])
        cap512.append(int(metric_fn(tok.decode(ids[: p_len + 512]))[0] > 0))
        # training context: BOS + first 1023 tokens
        ctx1024.append(int(metric_fn(tok.decode(ids[:1023]))[0] > 0))
        f = extract_structural_features(text)
        chi1.append(int(f[0])); chi6.append(float(f[5]))
    L = np.array(lens)
    out[name] = {
        "n": len(data),
        "oracle_acc_uncapped": float(np.mean(unc)),
        "oracle_acc_prompt+512_new_tokens": float(np.mean(cap512)),
        "oracle_acc_within_1024_train_ctx": float(np.mean(ctx1024)),
        "frac_len_gt_1024": float(np.mean(L > 1024)),
        "len_p50_p90_p99_max": [int(np.percentile(L, q)) for q in (50, 90, 99)] + [int(L.max())],
        "chi1_counts": dict(sorted(collections.Counter(chi1).items())),
        "chi6_mean": float(np.mean(chi6)),
    }
    print(name, json.dumps(out[name]))

for split in ["bfs", "dfs", "mixed"]:
    analyse(f"val_{split}", json.load(open(os.path.join(a.data_dir, f"val_{split}.json"))))
random.seed(0)
for split in ["bfs", "dfs", "mixed"]:
    d = json.load(open(os.path.join(a.data_dir, f"train_{split}.json")))
    analyse(f"train_{split}_sample{a.n_train}", random.sample(d, a.n_train)); del d
json.dump(out, open(os.path.join(a.data_dir, "oracle_analysis.json"), "w"), indent=2)
