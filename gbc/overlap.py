# Records, for every new problem, whether (target, sorted nums) occurs in each training set.
import argparse, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/gbc/data")
a = ap.parse_args()


def problems(path):
    # stream the multi-GB training files: only "target" and "nums" are needed
    import ijson
    with open(path, "rb") as f:
        for x in ijson.items(f, "item"):
            yield (int(x["target"]), tuple(sorted(int(v) for v in x["nums"])))


train = {}
for m in ["bfs", "dfs", "mixed"]:
    train[m] = set(problems(f"{ROOT}/data/experiment_500k/train_{m}.json"))
    print(f"train_{m}: {len(train[m])} unique problems", flush=True)
union = set().union(*train.values())
stats = {f"train_{m}_unique": len(train[m]) for m in train}
stats["train_union_unique"] = len(union)
for m in ["bfs", "dfs", "mixed"]:
    d = json.load(open(f"{a.out}/fresh_{m}.json"))
    for x in d:
        p = (int(x["target"]), tuple(sorted(int(v) for v in x["nums"])))
        x["in_train"] = {k: (p in train[k]) for k in train}
        x["in_any_train"] = p in union
    json.dump(d, open(f"{a.out}/fresh_{m}.json", "w"))
    stats[f"fresh_{m}_in_any_train"] = sum(x["in_any_train"] for x in d) / len(d)
    stats[f"fresh_{m}_in_train_mixed"] = sum(x["in_train"]["mixed"] for x in d) / len(d)
json.dump(stats, open(f"{a.out}/overlap_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
