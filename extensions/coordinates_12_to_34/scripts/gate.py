# Step 3 (CPU; analysis environment): checks that the inputs reproduce App. G and App. J. Writes analysis/gate.json
# under BASE with "pass"; analyze.py aggregates nothing unless it is true.
#  check 1, extractor agreement: in every feature file chi_1..chi_12 of ontology_countdown.extract match
#     src/distribution_metrics.py (max abs diff <= 1e-4), all values finite, expected row counts.
#  check 2, label/feature alignment: every App. J label idx has a feature row; chi columns equal gbc/features.npz;
#     label counts and accuracies equal App. J (reduced-scale run: f2 only, whose labels all lie in the subset).
#  check 3, App. J reproduction: Phi_12, repetition 0 -> fold-mean ROC-AUC within 0.002 of gbc/results_<f>.json
#     (reduced-scale run: f2 only).
#  check 4, App. G reproduction: Phi_12 on the pools, N = 2,000, seeds 0-9 (always all 10) -> means within 0.01 of
#     results/w1_robust.json for BFS-BFS, DFS-DFS, BFS-MIXED, BFS-DFS, DFS-MIXED.
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auclib as A  # noqa: E402
import common as C  # noqa: E402
import w1lib as W  # noqa: E402

t0 = time.time()
G = {"time": C.now(), "smoke": C.SMOKE, "checks": {}}
ok = {}

# ---- check 1
g1 = {}
for s in C.SOURCES:
    for m in C.MEASURES:
        meta = C.jload(C.feat_path(s, m).replace(".npz", ".meta.json"), {})
        F = C.load_feats(s, m)
        exp_n = {"fresh": C.SMOKE_FRESH_LIMIT if C.SMOKE else C.N_FRESH[m], "pool": C.POOL_SIZE, "eval": 2000}[s]
        diff = float(np.abs(F["X"][:, :12] - F["chi_recomputed"]).max())
        g1[f"{s}_{m}"] = {"n": int(len(F["X"])), "expected_n": exp_n, "max_abs_chi_diff": diff,
                          "finite": bool(np.isfinite(F["X"]).all()), "names_ok": F["names"].tolist() == C.phi_order(),
                          "meta_n": meta.get("n")}
ok["G1"] = all(v["n"] == v["expected_n"] and v["max_abs_chi_diff"] <= C.G1_TOL and v["finite"] and v["names_ok"]
               for v in g1.values())
G["checks"]["G1"] = {"pass": ok["G1"], "files": g1}
print("G1", ok["G1"], flush=True)

# ---- check 2
gbc = np.load(C.GBC_FEATURES)
g2 = {}
for f in C.MODELS:
    L = C.labels(f)
    n = int(sum(len(L[m][0]) for m in C.MEASURES))
    acc = float(np.concatenate([L[m][1] for m in C.MEASURES]).mean())
    maxdiff, missing = 0.0, 0
    for m in C.MEASURES:
        F = C.load_feats("fresh", m)
        pos = {int(s_): r for r, s_ in enumerate(F["src_idx"])}
        idx = L[m][0]
        missing += int(sum(int(i) not in pos for i in idx))
        rows = np.array([pos[int(i)] for i in idx if int(i) in pos], dtype=np.int64)
        if len(rows):
            maxdiff = max(maxdiff, float(np.abs(F["X"][rows, :12] - gbc[f"{m}_F12"][idx[:len(rows)]]).max()))
    required = (not C.SMOKE) or f == "f2"
    g2[f] = {"n": n, "accuracy": acc, "appj_n": C.APPJ_N[f], "appj_accuracy": C.APPJ_ACC[f], "missing_rows": missing,
             "max_abs_diff_vs_gbc_features": maxdiff, "required": required,
             "pass": missing == 0 and maxdiff <= 1e-6 and (not required or (n == C.APPJ_N[f] and abs(acc - C.APPJ_ACC[f]) < 1e-9))}
ok["G2"] = all(v["pass"] for v in g2.values())
G["checks"]["G2"] = {"pass": ok["G2"], "models": g2}
print("G2", ok["G2"], g2, flush=True)

# ---- check 3
g3 = {}
for f in C.MODELS:
    required = (not C.SMOKE) or f == "f2"
    if not required:
        g3[f] = {"required": False, "note": "smoke subset of f1; App. J value not comparable"}
        continue
    X, y, *_ = C.labelled_data(f)
    _, _, folds = A.cv_oof(X[:, C.phi_cols(12)], y, 0)
    auc = float(np.mean([x["auc"] for x in folds]))
    mcf = float(np.mean([x["mcfadden"] for x in folds]))
    ref = C.jload(C.APPJ_RESULTS[f])["pooled_F12"]
    g3[f] = {"required": True, "auc": auc, "mcfadden": mcf, "appj_auc": ref["auc"][0], "appj_mcfadden": ref["pseudo_r2"][0],
             "abs_diff_auc": abs(auc - ref["auc"][0]), "pass": abs(auc - ref["auc"][0]) <= C.G3_TOL}
    print("G3", f, g3[f], f"[{time.time() - t0:.0f}s]", flush=True)
ok["G3"] = all(v.get("pass", True) for v in g3.values())
G["checks"]["G3"] = {"pass": ok["G3"], "models": g3}

# ---- check 4
paper = C.jload(C.PAPER_W1)
Z, mu, sd = W.block_data("pool")
g4 = {}
for name, (a, b) in C.PAPER_PAIRS.items():
    vals, codes = [], []
    for s in range(10):
        i1, i2 = W.draw(s, len(Z[a]), len(Z[b]), C.W1_N, a == b)
        w, code, _ = W.emd(Z[a][i1][:, :12], Z[b][i2][:, :12])
        vals.append(w); codes.append(code)
    ref = paper[name]["exact_W1_zscored"]
    g4[name] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "values": vals, "codes": codes,
                "paper_mean": ref[0], "paper_sd": ref[1], "abs_diff": abs(float(np.mean(vals)) - ref[0])}
    print("G4", name, round(g4[name]["mean"], 5), "paper", round(ref[0], 5), f"[{time.time() - t0:.0f}s]", flush=True)
ok["G4"] = all(v["abs_diff"] <= C.G4_TOL and all(c == 1 for c in v["codes"]) for v in g4.values())
G["checks"]["G4"] = {"pass": ok["G4"], "pairs": g4}

# ---- feature sanity (reported, not gating): coordinates with zero variance in a block
zero = {}
for s in C.SOURCES:
    allv = np.vstack([C.load_feats(s, m)["X"] for m in C.MEASURES])
    zero[s] = [C.phi_order()[j] for j in np.where(allv.std(0) == 0)[0]]
G["zero_variance_coordinates"] = zero
G["pass"] = all(ok.values())
G["seconds"] = round(time.time() - t0, 1)
C.jdump(G, f"{C.BASE}/analysis/gate.json")
print("gate pass:", G["pass"], ok, f"{time.time() - t0:.0f}s", flush=True)
