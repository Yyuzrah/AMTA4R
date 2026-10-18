"""Exact W1 between the mini-SWE-agent and OpenHands pools and the within-harness baselines, with the estimator of
App. G (also used in extensions/structural_audit/scripts/w1.py): joint z-scoring over the union of the two harness pools,
Euclidean cost, uniform weights, exact ot.emd2 (numItermax 1e7), 10 seeds np.random.default_rng(s); between = N
episodes w/o replacement from each pool; within(h) = first N vs next N of one permutation of h's pool; N = floor(n/2),
n = number of instances finished in both harnesses. Restart-safe (tasks appended to <out_dir>/w1_<variant>.jsonl).
Also the per-coordinate attribution (1-D W1 per coordinate, share, signed standardized mean difference, 1-D within
baseline and ratio). Variants: main (16 coordinates), noLEN (without T4_tool_calls), family_c_file_rule (a label only; run it
on the features written by phi_features.py with focus_ext=0). Output: <out_dir>/summary_<variant>.json.
Usage: python w1.py <features_csv> <variant: main|noLEN|family_c_file_rule> <out_dir>"""
import json, os, sys, time
import numpy as np, pandas as pd, ot
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

COLS = ["T1_max_depth", "T2_mean_depth", "T3_sd_depth", "T4_tool_calls", "T5_distinct_files", "T6_file_reopen",
        "T6a_repeat_action", "C1_push_rate", "C2_pop_rate", "C7_failed_edits", "C9_failed_edit_cascade",
        "C14_run_calls", "C15_first_edit_pos", "I1_read_calls", "I1_write_calls", "I3_error_rate"]
H = ["mini", "oh"]; SEEDS = list(range(10))


def emd(A, B):
    w = np.ones(len(A)) / len(A)
    c, log = ot.emd2(w, w, cdist(A, B), numItermax=10_000_000, log=True)
    return float(c), (log.get("warning") or "")


def main(fcsv, variant, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(fcsv)
    cols = [c for c in COLS if not (variant == "noLEN" and c == "T4_tool_calls")]
    common = sorted(set(df[df.harness == "mini"].instance_id) & set(df[df.harness == "oh"].instance_id))
    df = df[df.instance_id.isin(common)].sort_values(["harness", "instance_id"])
    X = {h: df[df.harness == h][cols].to_numpy(float) for h in H}
    assert all(np.isfinite(X[h]).all() for h in H)
    allv = np.vstack([X[h] for h in H]); mu, sd = allv.mean(0), allv.std(0); sd[sd == 0] = 1
    Z = {h: (X[h] - mu) / sd for h in H}
    n = len(common); N = n // 2
    path = os.path.join(out, f"w1_{variant}.jsonl"); done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[(r["kind"], r["h"], r["seed"])] = r
    with open(path, "a") as fh:
        for s in SEEDS:
            for kind, h in [("between", "mini|oh"), ("within", "mini"), ("within", "oh")]:
                if (kind, h, s) in done:
                    continue
                rng = np.random.default_rng(s)
                if kind == "between":
                    A = Z["mini"][rng.choice(n, N, replace=False)]; B = Z["oh"][rng.choice(n, N, replace=False)]
                else:
                    p = rng.permutation(n); A, B = Z[h][p[:N]], Z[h][p[N:2 * N]]
                t0 = time.time(); c, w = emd(A, B)
                r = dict(kind=kind, h=h, seed=s, N=N, n=n, w1=c, warning=w, sec=round(time.time() - t0, 3))
                fh.write(json.dumps(r) + "\n"); fh.flush(); done[(kind, h, s)] = r
    bw = np.array([done[("between", "mini|oh", s)]["w1"] for s in SEEDS])
    wm = np.array([done[("within", "mini", s)]["w1"] for s in SEEDS]); wo = np.array([done[("within", "oh", s)]["w1"] for s in SEEDS])
    W, SD = bw.mean(), bw.std(ddof=1); B = max(wm.mean(), wo.mean())
    res = dict(variant=variant, n=n, N=N, d=len(cols), W=W, W_sd=SD, within_mini=wm.mean(), within_mini_sd=wm.std(ddof=1),
               within_oh=wo.mean(), within_oh_sd=wo.std(ddof=1), B=B, R=W / B, CV=SD / W,
               warnings=sorted({r["warning"] for r in done.values() if r["warning"]}))
    # per-coordinate attribution on the full pools (z-scored as above); 1-D within baseline per coordinate
    attr = []
    for j, c in enumerate(cols):
        a, b = Z["mini"][:, j], Z["oh"][:, j]
        base = []
        for s in SEEDS:
            rng = np.random.default_rng(s); per = []
            for h in H:
                p = rng.permutation(n); per.append(wasserstein_distance(Z[h][p[:N], j], Z[h][p[N:2 * N], j]))
            base.append(per)
        base = np.array(base).mean(0)
        pooled = np.sqrt((X["mini"][:, j].var(ddof=1) + X["oh"][:, j].var(ddof=1)) / 2)
        smd = (X["oh"][:, j].mean() - X["mini"][:, j].mean()) / pooled if pooled > 0 else 0.0
        attr.append(dict(coord=c, w1_1d=wasserstein_distance(a, b), within_1d_max=float(base.max()),
                         ratio_1d=wasserstein_distance(a, b) / base.max() if base.max() > 0 else float("inf"),
                         smd_oh_minus_mini=float(smd), mean_mini=float(X["mini"][:, j].mean()), mean_oh=float(X["oh"][:, j].mean())))
    tot = sum(a["w1_1d"] for a in attr)
    for a in attr:
        a["share"] = a["w1_1d"] / tot if tot > 0 else 0.0
    attr.sort(key=lambda a: -a["w1_1d"])
    res["attribution"] = attr; res["top3"] = [a["coord"] for a in attr[:3]]
    json.dump(res, open(os.path.join(out, f"summary_{variant}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "attribution"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
