# Predicts per-problem success from chi = Phi(z) and from the baselines (App. J: Table 2 except the operand-count
# row, transfer across measures). Uses the analysis environment (requirements-analysis.txt, scikit-learn 1.5.2).
import json, glob, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
M = ["bfs", "dfs", "mixed"]
F = np.load(f"{ROOT}/gbc/features.npz")


def feats(m, name):
    x = F[f"{m}_F12"]
    return {"F12": x, "F11": np.delete(x, 9, axis=1), "LEN": F[f"{m}_LEN"], "PROB": F[f"{m}_PROB"]}[name]


def labels(pattern, m):
    y = {}
    for p in glob.glob(f"{ROOT}/gbc/labels/{pattern}_{m}*.jsonl"):
        for line in open(p):
            r = json.loads(line); y[r["idx"]] = r["correct"]
    idx = np.array(sorted(y)); return idx, np.array([y[i] for i in idx])


def ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def clf(): return HistGradientBoostingClassifier(random_state=0)


def cv(X, y):
    oof = np.zeros(len(y)); fold = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; oof[te] = p
        p0 = np.full(len(te), y[tr].mean())
        fold.append({"auc": roc_auc_score(y[te], p), "pseudo_r2": 1 - ll(y[te], p) / ll(y[te], p0),
                     "brier": float(np.mean((p - y[te]) ** 2)), "acc": float(np.mean((p > .5) == y[te])), "mean_abs": float(np.mean(np.abs(y[te] - p)))})
    s = {k: [float(np.mean([f[k] for f in fold])), float(np.std([f[k] for f in fold]))] for k in fold[0]}
    return s, oof


def boot(y, oof, base, B=1000):
    rng = np.random.default_rng(0); a, r = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); a.append(roc_auc_score(y[i], oof[i])); r.append(1 - ll(y[i], oof[i]) / ll(y[i], np.full(len(i), base)))
    return {"auc_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], "pseudo_r2_ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def run(pattern, tag):
    R = {"model": pattern}; Xs, ys, ms, tr = {}, {}, [], []
    for m in M:
        idx, y = labels(pattern, m)
        if len(idx) == 0: continue
        Xs[m] = {n: feats(m, n)[idx] for n in ["F12", "F11", "LEN", "PROB"]}; ys[m] = y
        tr.append(F[f"{m}_in_any_train"][idx]); ms += [m] * len(y)
        R[f"n_{m}"] = int(len(y)); R[f"accuracy_{m}"] = float(y.mean())
    y = np.concatenate([ys[m] for m in Xs]); intr = np.concatenate(tr)
    R["n_total"] = int(len(y)); R["accuracy_total"] = float(y.mean())
    for n in ["F12", "F11", "LEN", "PROB"]:
        X = np.concatenate([Xs[m][n] for m in Xs]); s, oof = cv(X, y); R[f"pooled_{n}"] = s
        if n == "F12": R["pooled_F12_bootstrap"] = boot(y, oof, y.mean())
    X = np.concatenate([Xs[m]["F12"] for m in Xs]); keep = ~intr
    R["n_not_in_train"] = int(keep.sum())
    if keep.sum() > 100 and len(set(y[keep])) == 2: R["not_in_train_F12"] = cv(X[keep], y[keep])[0]
    for m in Xs: R[f"within_{m}_F12"] = cv(Xs[m]["F12"], ys[m])[0]
    for s in Xs:
        model = clf().fit(Xs[s]["F12"], ys[s])
        for t in Xs:
            if s == t: continue
            p = model.predict_proba(Xs[t]["F12"])[:, 1]
            R[f"transfer_{s}_to_{t}"] = {"auc_T": float(roc_auc_score(ys[t], p)), "risk_T_observed": float(1 - ys[t].mean()),
                                         "risk_T_predicted": float(1 - p.mean()), "eps_hat": float(abs(ys[t].mean() - p.mean()))}
    # Exploratory: the same features computed from the model's own generation
    G, yg = [], []
    for m in Xs:
        k = f"{pattern}_{m}_G12"
        if k in F.files:
            gi = dict(zip(F[f"{pattern}_{m}_G_idx"].tolist(), range(len(F[f"{pattern}_{m}_G_idx"]))))
            idx, y_m = labels(pattern, m)
            G.append(F[k][[gi[i] for i in idx]]); yg.append(y_m)
    if G:
        G, yg = np.concatenate(G), np.concatenate(yg)
        R["exploratory_own_generation_G12"] = cv(G, yg)[0]
        R["exploratory_own_generation_G11"] = cv(np.delete(G, 9, axis=1), yg)[0]
    json.dump(R, open(f"{ROOT}/gbc/results_{tag}.json", "w"), indent=2); return R


if __name__ == "__main__":
    out = {"f1": run("f1_small_deep_mixed_rope", "f1")}
    if glob.glob(f"{ROOT}/gbc/labels/f2_*.jsonl"): out["f2"] = run("f2_large_deep_bfs_ape", "f2")
    print(json.dumps(out, indent=1))
