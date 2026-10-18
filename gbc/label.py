# Per-problem labels for App. J. Decoding as in src/eval_flexible.py (same prompt, batch size, left padding,
# greedy, 512 new tokens, metric_fn); each line stores idx, correct (1 - zero-one loss), status and the generation.
import argparse, json, os, sys
import torch
from tqdm import tqdm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, f"{ROOT}/src")
from eval_flexible import load_model_and_tokenizer
from countdown_utils import metric_fn

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--measure", required=True, choices=["bfs", "dfs", "mixed"])
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

data = json.load(open(f"{ROOT}/gbc/data/fresh_{a.measure}.json"))[a.start:a.end]
model, tokenizer, pe_type = load_model_and_tokenizer(a.ckpt)
batch_size = 1 if pe_type == "ape" else 32
print(f"{a.ckpt} pe={pe_type} batch={batch_size} measure={a.measure} [{a.start},{a.end})", flush=True)

done = set()
if os.path.exists(a.out):  # resume after preemption
    for line in open(a.out):
        done.add(json.loads(line)["idx"])
prompts = [f"{tokenizer.bos_token}Current State: {s['target']}:{s['nums']}, Operations: []\n" for s in data]
with open(a.out, "a") as fo:
    for i in tqdm(range(0, len(prompts), batch_size)):
        idx = list(range(a.start + i, a.start + min(i + batch_size, len(prompts))))
        if all(j in done for j in idx):
            continue
        inputs = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        for j, text in zip(idx, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
            score, status = metric_fn(text)
            fo.write(json.dumps({"idx": j, "correct": 1 if score > 0 else 0, "status": status, "gen": text}) + "\n")
        fo.flush()
