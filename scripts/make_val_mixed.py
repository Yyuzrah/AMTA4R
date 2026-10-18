# Builds val_mixed.json from the first 1,000 trajectories of val_bfs.json and of val_dfs.json, shuffled (seed 42).
import argparse, json, os, random
p = argparse.ArgumentParser(); p.add_argument("--data_dir", default="data/experiment_500k"); p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
bfs = json.load(open(os.path.join(a.data_dir, "val_bfs.json")))
dfs = json.load(open(os.path.join(a.data_dir, "val_dfs.json")))
mixed = bfs[:1000] + dfs[:1000]
random.seed(a.seed); random.shuffle(mixed)
json.dump(mixed, open(os.path.join(a.data_dir, "val_mixed.json"), "w"), indent=4)
print(f"Created val_mixed.json with {len(mixed)} samples (seed={a.seed}).")
