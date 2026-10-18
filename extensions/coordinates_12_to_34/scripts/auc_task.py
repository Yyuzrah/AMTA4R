# Step 4 (CPU; analysis environment). Task t in 0..19 -> model ["f1", "f2"][t // N_CV_REPS], repetition
# s = t % N_CV_REPS. For every d in DIMS: out-of-fold ROC-AUC / McFadden of the App. J classifier on Phi_d (same folds
# for every d). Repetition 0 also runs three further analyses: label coded as P(loss = 1) (all d), folds grouped by
# problem (d = 12, 34), and within operand count 3 / 4 (all d).
# Writes runs/auc/<model>_s<s>.json (fold metrics) and .npz (y, fold ids, OOF predictions per d) under BASE.
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auclib as A  # noqa: E402
import common as C  # noqa: E402

t = int(sys.argv[1])
model, s = list(C.MODELS)[t // C.N_CV_REPS], t % C.N_CV_REPS
out = f"{C.BASE}/runs/auc/{model}_s{s}"
t0 = time.time()
X, y, ops, keys, meas = C.labelled_data(model)
R = {"model": model, "model_name": C.MODELS[model], "seed": s, "n": int(len(y)), "accuracy": float(y.mean()),
     "label_coding": "y = correct (the implemented App. J coding)", "dims": {}, "smoke": C.SMOKE}
store = {"y": y.astype(np.int8)}
fold_ref = None
for d in C.DIMS:
    oof, fid, folds = A.cv_oof(X[:, C.phi_cols(d)], y, s)
    if fold_ref is None:
        fold_ref = fid
    assert (fid == fold_ref).all(), "folds differ across d"
    R["dims"][str(d)] = {"folds": folds, **A.summarize(folds)}
    store[f"oof_d{d}"] = oof.astype(np.float64)
    print(f"{model} s{s} d{d}: AUC {R['dims'][str(d)]['auc'][0]:.4f} McF {R['dims'][str(d)]['mcfadden'][0]:.4f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
store["fold_id"] = fold_ref
if s == 0:
    ex = {"coding_loss": {}, "grouped": {}, "within_ops": {}}
    yl = 1 - y
    for d in C.DIMS:
        _, _, folds = A.cv_oof(X[:, C.phi_cols(d)], yl, 0)
        ex["coding_loss"][str(d)] = {"folds": folds, **A.summarize(folds)}
    for d in [12, 34]:
        oof, fid, folds = A.cv_oof(X[:, C.phi_cols(d)], y, 0, groups=keys)
        ex["grouped"][str(d)] = {"folds": folds, **A.summarize(folds)}
        store[f"grouped_oof_d{d}"], store[f"grouped_fold_id_d{d}"] = oof, fid
    # duplicate problems across the ungrouped folds (split-leakage diagnostic)
    test_dup = 0
    for k in range(5):
        tr_keys = set(keys[fold_ref != k].tolist())
        test_dup += int(sum(kk in tr_keys for kk in keys[fold_ref == k].tolist()))
    ex["ungrouped_test_rows_with_problem_in_train_folds"] = test_dup
    ex["n_distinct_problems"] = int(len(set(keys.tolist())))
    for k in [3, 4]:
        sel = ops == k
        ex["within_ops"][str(k)] = {"n": int(sel.sum()), "accuracy": float(y[sel].mean()) if sel.any() else None}
        if sel.sum() >= 100 and len(set(y[sel].tolist())) == 2 and min(np.bincount(y[sel])) >= 5:
            for d in C.DIMS:
                _, _, folds = A.cv_oof(X[sel][:, C.phi_cols(d)], y[sel], 0)
                ex["within_ops"][str(k)][str(d)] = {"folds": folds, **A.summarize(folds)}
    R["secondary"] = ex
    print(f"{model} s0 secondary analyses done [{time.time() - t0:.0f}s]", flush=True)
R["seconds"] = round(time.time() - t0, 1)
R["time"] = C.now()
C.savez(out + ".npz", **store)
C.jdump(R, out + ".json")
print(f"auc task {t} ({model}, s{s}) done in {time.time() - t0:.0f}s", flush=True)
