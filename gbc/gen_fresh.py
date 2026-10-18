# New reference trajectories for App. J: the training-data generator with seed 2026, 17,000 per measure
# (the first 16,667/16,667/16,666 are used). overlap.py then marks problems that occur in a training set.
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, f"{ROOT}/src")
from generate_dataset_SOS import generate_data

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=17000)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default=f"{ROOT}/gbc/data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
args = argparse.Namespace(num_samples=a.n, seed=a.seed, max_target=100, min_range=3, start_range=4)
for m in ["bfs", "dfs", "mixed"]:
    generate_data(args, m, f"{a.out}/fresh_{m}.json")

