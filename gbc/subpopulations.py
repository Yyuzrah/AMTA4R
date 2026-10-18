# Sub-populations (App. J): (A) pseudo-R2 family of the chi predictions, out of fold and in sample; (B) mean label
# vs. mean prediction in cells of search measure x operand count x target band (chi, LEN, PROB, constant, transfer).
import json, glob, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
M = ["bfs", "dfs", "mixed"]
F = np.load(f"{ROOT}/gbc/features.npz")


def labels(pattern, m):
    y = {}
    for p in glob.glob(f"{ROOT}/gbc/labels/{pattern}_{m}*.jsonl"):
        for line in open(p):
            r = json.loads(line); y[r["idx"]] = r["correct"]
    idx = np.array(sorted(y)); return idx, np.array([y[i] for i in idx])


def clf(): return HistGradientBoostingClassifier(random_state=0)


def ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def oof(X, y):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def pseudo_family(y, p):
    n = len(y); L1 = ll(y, p); L0 = ll(y, np.full(n, y.mean()))
    cs = 1 - np.exp(2 * (L0 - L1) / n)
    return {"mcfadden": 1 - L1 / L0, "cox_snell": float(cs), "nagelkerke": float(cs / (1 - np.exp(2 * L0 / n))),
            "efron": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "tjur": float(p[y == 1].mean() - p[y == 0].mean()), "auc": float(roc_auc_score(y, p))}


def cells_of(meas, nops, target, edges):
    band = np.digitize(target, edges[1:-1])  # band index from the inner edges
    return np.array([f"{a}|{b}|{c}" for a, b, c in zip(meas, nops, band)])


def meas_r2(y, p, cell, keep):
    ks = [k for k in keep]
    R = np.array([y[cell == k].mean() for k in ks]); Rh = np.array([p[cell == k].mean() for k in ks]); w = np.array([(cell == k).sum() for k in ks])
    d = R - Rh
    r2 = 1 - np.sum(d ** 2) / np.sum((R - R.mean()) ** 2)
    r2w = 1 - np.sum(w * d ** 2) / np.sum(w * (R - np.average(R, weights=w)) ** 2)
    return {"r2_meas": float(r2), "r2_meas_weighted": float(r2w), "mean_abs_dev": float(np.mean(np.abs(d))), "eps_hat_max": float(np.max(np.abs(d))),
            "risk_range": [float(R.min()), float(R.max())]}


def boot(y, p, cell, keep, B=1000):
    rng = np.random.default_rng(0); r, e = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); yy, pp, cc = y[i], p[i], cell[i]
        ok = [k for k in keep if (cc == k).sum() > 0]
        m = meas_r2(yy, pp, cc, ok); r.append(m["r2_meas"]); e.append(m["eps_hat_max"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"r2_meas_ci95": q(r), "eps_hat_max_ci95": q(e)}


def run(pattern, edges):
    X12, XL, XP, y, meas, nops, tgt = [], [], [], [], [], [], []
    for m in M:
        idx, ym = labels(pattern, m)
        X12.append(F[f"{m}_F12"][idx]); XL.append(F[f"{m}_LEN"][idx]); XP.append(F[f"{m}_PROB"][idx]); y.append(ym)
        meas += [m] * len(ym); nops.append(F[f"{m}_PROB"][idx][:, 1]); tgt.append(F[f"{m}_PROB"][idx][:, 0])
    X12, XL, XP, y = map(np.concatenate, (X12, XL, XP, y)); meas = np.array(meas); nops = np.concatenate(nops).astype(int); tgt = np.concatenate(tgt)
    R = {"n": int(len(y))}
    p12 = oof(X12, y)
    R["A_pseudo_family_oof"] = pseudo_family(y, p12)
    R["A_pseudo_family_in_sample"] = pseudo_family(y, clf().fit(X12, y).predict_proba(X12)[:, 1])
    cell = cells_of(meas, nops, tgt, edges)
    u, c = np.unique(cell, return_counts=True); keep = [k for k, n in zip(u, c) if n >= 100]
    R["cells_total"] = int(len(u)); R["cells_kept"] = len(keep); R["cells_dropped_lt100"] = int(len(u) - len(keep))
    R["B_F12"] = {**meas_r2(y, p12, cell, keep), **boot(y, p12, cell, keep)}
    R["B_LEN"] = meas_r2(y, oof(XL, y), cell, keep)
    R["B_PROB"] = meas_r2(y, oof(XP, y), cell, keep)
    pc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X12, y): pc[te] = y[tr].mean()
    R["B_constant"] = meas_r2(y, pc, cell, keep)
    for s in M:
        src = meas == s; model = clf().fit(X12[src], y[src]); tgt_mask = ~src
        pt = model.predict_proba(X12[tgt_mask])[:, 1]; kt = [k for k in keep if k.split("|")[0] != s]
        R[f"B_transfer_from_{s}"] = meas_r2(y[tgt_mask], pt, cell[tgt_mask], kt)
    return R


out = {"f1": run("f1_small_deep_mixed_rope", np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 101])),
       "f2": run("f2_large_deep_bfs_ape", np.array([10, 40, 70, 101]))}
json.dump(out, open(f"{ROOT}/gbc/results_subpopulations.json", "w"), indent=2)
print(json.dumps(out, indent=1))
