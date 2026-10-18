"""Instance lists of the harness-switch experiment.
Population: the SWE-bench Verified instances (500) that have a prebuilt OpenHands agent-server image on GHCR
under the given tag prefix (all 500 for the images of OpenHands software-agent-sdk 1.16.1).
Four setup-test instances are kept out of the sample: three fixed ones and the first pytest instance by id
(written to data/instances_setup_test.txt).
Sample: N=300 drawn without replacement, stratified by repo (proportional allocation, largest remainder),
order within repo by np.random.default_rng(0).permutation. Output: data/instances_setup_test.txt, data/instances_main.txt,
data/image_map.json (instance -> GHCR image). Refuses to overwrite an existing data/instances_main.txt.
Usage: python select_instances.py <tags_json> <tag_prefix>
  tags_json: JSON list of all tags of ghcr.io/openhands/eval-agent-server (registry API /v2/.../tags/list)
  tag_prefix: the part of the tag before "-sweb.eval.x86_64." shared by the images of the SDK version used"""
import json, os, sys, collections
import numpy as np
from datasets import load_dataset
HARNESS_DIR = os.environ["HARNESS_DIR"]
out_main, out_test = f"{HARNESS_DIR}/data/instances_main.txt", f"{HARNESS_DIR}/data/instances_setup_test.txt"
if os.path.exists(out_main):
    sys.exit("exists: " + out_main)
tags = json.load(open(sys.argv[1]))
pref = sys.argv[2] + "-sweb.eval.x86_64."
img = {}
for t in tags:
    if t.startswith(pref) and t.endswith("-source-minimal"):
        n = t[len(pref):-len("-source-minimal")]
        img[n.replace("_1776_", "__")] = "ghcr.io/openhands/eval-agent-server:" + t
ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
ids = [x["instance_id"] for x in ds]; repo = {x["instance_id"]: x["repo"] for x in ds}
have = [i for i in ids if i in img]
print("verified", len(ids), "with image", len(have), "missing", [i for i in ids if i not in img])
setup_test = ["django__django-11099", "astropy__astropy-12907", "sympy__sympy-20590"]
setup_test.append(sorted(i for i in have if repo[i] == "pytest-dev/pytest")[0])
pool = [i for i in have if i not in setup_test]
by = collections.defaultdict(list)
for i in sorted(pool): by[repo[i]].append(i)
N = 300; tot = len(pool)
quota = {r: N * len(v) / tot for r, v in by.items()}
alloc = {r: int(np.floor(q)) for r, q in quota.items()}
rest = N - sum(alloc.values())
for r in sorted(by, key=lambda r: (-(quota[r] - alloc[r]), r))[:rest]: alloc[r] += 1
rng = np.random.default_rng(0)
main = []
for r in sorted(by):
    perm = rng.permutation(len(by[r]))
    main += [by[r][k] for k in perm[:alloc[r]]]
main = sorted(main)
assert len(main) == N and not set(main) & set(setup_test)
open(out_test, "w").write("\n".join(setup_test) + "\n"); open(out_main, "w").write("\n".join(main) + "\n")
json.dump({i: img[i] for i in setup_test + main}, open(f"{HARNESS_DIR}/data/image_map.json", "w"), indent=0)
print("alloc", dict(sorted(alloc.items())), "pool", tot)
