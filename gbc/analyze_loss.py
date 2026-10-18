"""App. J analysis with the label coded as the zero-one loss, plus label checks (prompt alignment, re-scoring).

Inputs: label .jsonl files and features.npz. Output: results_loss.json next to this script.
Label convention: y = l(f,z) = 1 - correct (zero-one loss); p_hat = predicted P(l = 1).
"""
import sys
sys.dont_write_bytecode = True  # do not write .pyc for imported project modules
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")  # limit CPU threads
import json, io, contextlib, random
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sklearn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
GBC = f"{ROOT}/gbc"
OUT = GBC
sys.path.insert(0, f"{ROOT}/src")
from countdown_utils import metric_fn  # noqa: E402

MEAS = ["bfs", "dfs", "mixed"]
N_F1 = {"bfs": 16667, "dfs": 16667, "mixed": 16666}
N_F2 = {"bfs": 1000, "dfs": 1000, "mixed": 1000}
MODELS = {
    "f1": dict(prefix="f1_small_deep_mixed_rope", n=N_F1,
               bands=[(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]),
    "f2": dict(prefix="f2_large_deep_bfs_ape", n=N_F2, bands=[(10, 40), (40, 70), (70, 100)]),
}
EPS = 1e-15
print("sklearn", sklearn.__version__, flush=True)

feat = np.load(f"{GBC}/features.npz")
res = {"sklearn": sklearn.__version__}


# ---------------------------------------------------------------- labels and checks
def read_labels(prefix, m):
    rows = {}
    files = sorted(p for p in os.listdir(f"{GBC}/labels") if p.startswith(f"{prefix}_{m}") and p.endswith(".jsonl"))
    for fn in files:
        with open(f"{GBC}/labels/{fn}") as fh:
            for line in fh:
                r = json.loads(line)
                assert r["idx"] not in rows, f"duplicate idx {r['idx']} in {fn}"
                rows[r["idx"]] = (r["correct"], r["status"], r["gen"])
    return files, rows


def parse_prompt(gen):
    first = gen.split("\n", 1)[0]
    assert first.startswith("Current State: ")
    t, rest = first[len("Current State: "):].split(":", 1)
    nums = [int(x) for x in rest[rest.index("[") + 1:rest.index("]")].split(", ")]
    return int(t), nums


def rescore(gen):
    with contextlib.redirect_stdout(io.StringIO()):
        score, status = metric_fn(gen)
    return (1 if score > 0 else 0), status


data = {}
sanity = {}
for name, cfg in MODELS.items():
    ys, Xf, Xl, Xp, meas_id, nops, tgt = [], [], [], [], [], [], []
    gens_all = {}
    s = {}
    for mi, m in enumerate(MEAS):
        files, rows = read_labels(cfg["prefix"], m)
        n = cfg["n"][m]
        idx = sorted(rows)
        assert idx == list(range(n)), f"{name} {m}: idx not 0..{n-1} (got {len(idx)} rows, min {idx[0]}, max {idx[-1]})"
        correct = np.array([rows[i][0] for i in idx], dtype=int)
        F12 = feat[f"{m}_F12"][:n]; LEN = feat[f"{m}_LEN"][:n]; PROB = feat[f"{m}_PROB"][:n]
        # alignment: prompt in the stored generation must equal (target, sorted nums) of features row idx
        mism = 0
        for i in idx:
            t, nums = parse_prompt(rows[i][2])
            ref = PROB[i]
            if not (t == ref[0] and len(nums) == ref[1] and sorted(nums) == [int(v) for v in ref[2:2 + len(nums)]]
                    and all(v == 0 for v in ref[2 + len(nums):])):
                mism += 1
        # status field consistency with correct
        status_valid = np.array([rows[i][1] == "Valid path." for i in idx])
        s[m] = dict(files=files, n_labels=len(idx), n_features_rows_total=int(feat[f"{m}_F12"].shape[0]),
                    accuracy=float(correct.mean()), base_rate_loss=float(1 - correct.mean()),
                    prompt_feature_mismatches=int(mism),
                    correct1_but_status_not_valid=int(((correct == 1) & ~status_valid).sum()),
                    correct0_with_status_valid=int(((correct == 0) & status_valid).sum()))
        print(name, m, s[m], flush=True)
        ys.append(1 - correct); Xf.append(F12); Xl.append(LEN); Xp.append(PROB)
        meas_id.append(np.full(n, mi)); nops.append(PROB[:, 1].astype(int)); tgt.append(PROB[:, 0])
        for i in idx:
            gens_all[(m, i)] = rows[i]
    y = np.concatenate(ys)
    s["total"] = dict(n=int(len(y)), accuracy=float(1 - y.mean()), base_rate_loss=float(y.mean()))
    # full re-score of every stored generation with metric_fn
    n_dis = 0; dis_examples = []
    for (m, i), (c, st, g) in gens_all.items():
        c2, st2 = rescore(g)
        if c2 != c or st2 != st:
            n_dis += 1
            if len(dis_examples) < 5:
                dis_examples.append([m, i, c, c2, st, st2])
    s["full_rescore_disagreements"] = n_dis
    s["full_rescore_examples"] = dis_examples
    sanity[name] = s
    data[name] = dict(y=y, F12=np.vstack(Xf), LEN=np.vstack(Xl), PROB=np.vstack(Xp),
                      meas=np.concatenate(meas_id), nops=np.concatenate(nops), tgt=np.concatenate(tgt),
                      gens=gens_all)

# re-score 5 random labels and store each result
rng = random.Random(12345)
picks = []
for _ in range(5):
    name = rng.choice(["f1", "f2"]); m = rng.choice(MEAS); i = rng.randrange(MODELS[name]["n"][m])
    c, st, g = data[name]["gens"][(m, i)]
    c2, st2 = rescore(g)
    picks.append(dict(model=name, measure=m, idx=i, stored_correct=c, rescored_correct=c2,
                      stored_status=st, rescored_status=st2, match=bool(c == c2 and st == st2)))
sanity["random5_rescore"] = picks
res["sanity"] = sanity
for name in data:
    data[name].pop("gens")
print(json.dumps(picks, indent=1), flush=True)


# ---------------------------------------------------------------- metrics
def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def new_clf():
    return HistGradientBoostingClassifier(random_state=0)


def cv_oof(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    p0_per_sample = np.full(len(y), np.nan)
    folds = []
    for tr, te in skf.split(X, y):
        clf = new_clf().fit(X[tr], y[tr])
        k1 = list(clf.classes_).index(1)
        p = clf.predict_proba(X[te])[:, k1]
        oof[te] = p
        p0 = y[tr].mean()
        p0_per_sample[te] = p0
        llm = ll(y[te], p); lln = ll(y[te], np.full(len(te), p0))
        folds.append(dict(auc=roc_auc_score(y[te], p), mcfadden=1 - llm / lln, n_iter=int(clf.n_iter_)))
    return oof, p0_per_sample, folds


def pseudo_family(y, p, p0_vec):
    n = len(y)
    llm = ll(y, p)
    out = {}
    for tag, nullp in [("null_pooled_base_rate", np.full(n, y.mean())), ("null_training_fold_base_rate", p0_vec)]:
        lln = ll(y, nullp)
        cs = 1 - np.exp((2.0 / n) * (lln - llm))
        nag = cs / (1 - np.exp((2.0 / n) * lln))
        out[tag] = dict(mcfadden=1 - llm / lln, cox_snell=float(cs), nagelkerke=float(nag))
    out["efron"] = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    out["tjur"] = float(p[y == 1].mean() - p[y == 0].mean())
    out["auc_pooled_oof"] = float(roc_auc_score(y, p))
    return out


def cells(d, bands, mask):
    """Return list of (key, index array) for measure x n_operands x target band, restricted to mask."""
    out = []
    for mi, m in enumerate(MEAS):
        for k in (3, 4):
            for bi, (lo, hi) in enumerate(bands):
                last = bi == len(bands) - 1
                inb = (d["tgt"] >= lo) & ((d["tgt"] <= hi) if last else (d["tgt"] < hi))
                sel = np.where(mask & (d["meas"] == mi) & (d["nops"] == k) & inb)[0]
                if len(sel):
                    out.append((f"{m}|{k}|[{lo},{hi}{']' if last else ')'}", sel))
    return out


def r2_meas(d, p, bands, mask):
    cl = cells(d, bands, mask)
    total = len(cl)
    cl = [(k, s) for k, s in cl if len(s) >= 100]
    R = np.array([d["y"][s].mean() for _, s in cl]); Rh = np.array([p[s].mean() for _, s in cl])
    w = np.array([len(s) for _, s in cl], dtype=float)
    dev = np.abs(R - Rh)
    r2 = 1 - np.sum((R - Rh) ** 2) / np.sum((R - R.mean()) ** 2)
    Rw = np.sum(w * R) / w.sum()
    r2w = 1 - np.sum(w * (R - Rh) ** 2) / np.sum(w * (R - Rw) ** 2)
    n_in = sum(len(s) for _, s in cl)
    return dict(r2_meas=float(r2), r2_meas_weighted=float(r2w), mean_abs_dev=float(dev.mean()),
                eps_hat_max=float(dev.max()), argmax_cell=cl[int(dev.argmax())][0],
                cells_total=total, cells_kept=len(cl), cells_dropped_lt100=total - len(cl),
                n_in_cells=int(n_in), n_mask=int(mask.sum()),
                risk_range=[float(R.min()), float(R.max())], min_cell_n=int(w.min()))


for name, cfg in MODELS.items():
    d = data[name]; y = d["y"]; allmask = np.ones(len(y), bool)
    r = {"n": int(len(y))}
    oofs = {}
    for fs in ["F12", "LEN", "PROB"]:
        oof, p0v, folds = cv_oof(d[fs], y)
        oofs[fs] = oof
        aucs = [f["auc"] for f in folds]; mcf = [f["mcfadden"] for f in folds]
        r[f"cv_{fs}"] = dict(auc_fold_mean=float(np.mean(aucs)), auc_fold_sd=float(np.std(aucs)),
                             auc_fold_sd_ddof1=float(np.std(aucs, ddof=1)),
                             mcfadden_fold_mean=float(np.mean(mcf)), mcfadden_fold_sd=float(np.std(mcf)),
                             mcfadden_fold_sd_ddof1=float(np.std(mcf, ddof=1)),
                             folds=folds, pooled=pseudo_family(y, oof, p0v))
        r[f"meas_{fs}"] = r2_meas(d, oof, cfg["bands"], allmask)
        print(name, fs, json.dumps({k: v for k, v in r[f"cv_{fs}"].items() if k != "folds"}), flush=True)
        print(name, fs, "R2_meas", json.dumps(r[f"meas_{fs}"]), flush=True)
    r["meas_constant"] = r2_meas(d, np.full(len(y), y.mean()), cfg["bands"], allmask)
    # transfer: fit on all of source measure S (F12), score other two measures
    for si, S in enumerate(MEAS):
        tr = d["meas"] == si
        clf = new_clf().fit(d["F12"][tr], y[tr])
        k1 = list(clf.classes_).index(1)
        p = np.full(len(y), np.nan)
        p[~tr] = clf.predict_proba(d["F12"][~tr])[:, k1]
        r[f"meas_transfer_from_{S}"] = r2_meas(d, p, cfg["bands"], ~tr)
        for ti, T in enumerate(MEAS):
            if ti == si:
                continue
            tm = d["meas"] == ti
            r[f"transfer_{S}_to_{T}"] = dict(auc_T=float(roc_auc_score(y[tm], p[tm])),
                                            risk_T_observed=float(y[tm].mean()), risk_T_predicted=float(p[tm].mean()),
                                            eps_hat=float(abs(y[tm].mean() - p[tm].mean())))
        print(name, "transfer from", S, json.dumps(r[f"meas_transfer_from_{S}"]), flush=True)
    res[name] = r

with open(f"{OUT}/results_loss.json", "w") as fh:
    json.dump(res, fh, indent=1)
print("wrote", f"{OUT}/results_loss.json")
