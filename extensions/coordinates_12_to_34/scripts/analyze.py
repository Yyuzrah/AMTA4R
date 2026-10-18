# Step 6 (CPU; analysis environment): aggregates the per-run files of auc_task.py, w1_task.py and w1_rescue.py and
# runs a self-check that recomputes values by separate code paths. Writes analysis/summary.json,
# analysis/verify.json, analysis/auc_table.csv and analysis/w1_table.csv under BASE. export_tables.py writes the
# tables of results/ with the aggregation functions of this file.
import glob
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import w1lib as W  # noqa: E402

T0 = time.time()
OUT = f"{C.BASE}/analysis"
DS = C.DIMS
DNEW = [d for d in DS if d != 12]
MIN_AUC_SEEDS = C.N_CV_REPS if C.SMOKE else 8
MIN_W1_SEEDS = C.W1_SEEDS if C.SMOKE else 8
WITHIN = ["BFS-BFS", "DFS-DFS", "MIXED-MIXED"]
BETWEEN = ["BFS-MIXED", "BFS-DFS", "DFS-MIXED"]


# ------------------------------------------------------------------ loading
def load_auc():
    R = {}
    for f in C.MODELS:
        for s in range(C.N_CV_REPS):
            p = f"{C.BASE}/runs/auc/{f}_s{s}.json"
            if os.path.exists(p):
                R[(f, s)] = C.jload(p)
    return R


def load_w1():
    resc = C.jload(f"{C.BASE}/runs/w1_rescue.json", {"entries": []})
    rmap = {(e["block"], e["seed"], e["pair"], e["N"], e["tag"], e["d"]): e for e in resc.get("entries", [])}
    rows, files = [], 0
    for block in C.W1_BLOCKS:
        for s in range(C.W1_SEEDS):
            p = f"{C.BASE}/runs/w1/{block}_s{s}.json"
            if not os.path.exists(p):
                continue
            files += 1
            for e in C.jload(p)["entries"]:
                k = (block, s, e["pair"], e["N"], e["tag"], e["d"])
                w, code, rescued = e["w1"], e["code"], False
                if code != 1 and k in rmap:
                    w, code, rescued = rmap[k]["w1"], rmap[k]["code"], True
                rows.append(dict(block=block, seed=s, pair=e["pair"], same=e["same"], N=e["N"], tag=e["tag"],
                                 d=e["d"], w1=w, code=code, rescued=rescued))
    return rows, files, resc


def fold_mean(R, f, s, d, key="auc", section=None):
    node = R[(f, s)]["dims"] if section is None else R[(f, s)]["secondary"][section]
    return float(np.mean([x[key] for x in node[str(d)]["folds"]]))


# ------------------------------------------------------------------ AUC part
def auc_part(R):
    res = {}
    for f in C.MODELS:
        seeds = sorted(s for (ff, s) in R if ff == f)
        r = {"seeds": seeds, "n_seeds": len(seeds), "evaluable": len(seeds) >= MIN_AUC_SEEDS}
        if not seeds:
            r["H_auc"] = None
            res[f] = r
            continue
        auc = {d: np.array([fold_mean(R, f, s, d) for s in seeds]) for d in DS}
        mcf = {d: np.array([fold_mean(R, f, s, d, "mcfadden") for s in seeds]) for d in DS}
        r["n"] = R[(f, seeds[0])]["n"]
        r["accuracy"] = R[(f, seeds[0])]["accuracy"]
        r["auc"] = {str(d): [float(auc[d].mean()), float(auc[d].std(ddof=1)) if len(seeds) > 1 else 0.0] for d in DS}
        r["mcfadden"] = {str(d): [float(mcf[d].mean()), float(mcf[d].std(ddof=1)) if len(seeds) > 1 else 0.0] for d in DS}
        r["delta_auc"] = {str(d): float((auc[d] - auc[12]).mean()) for d in DNEW}
        r["delta_auc_seed_min_max"] = {str(d): [float((auc[d] - auc[12]).min()), float((auc[d] - auc[12]).max())] for d in DNEW}
        r["share_seeds_abs_delta_below"] = {str(d): float(np.mean(np.abs(auc[d] - auc[12]) < C.AUC_DELTA)) for d in DNEW}
        r["delta_mcfadden"] = {str(d): float((mcf[d] - mcf[12]).mean()) for d in DNEW}
        if 0 in seeds:
            r["seed0_appj_protocol"] = {str(d): [fold_mean(R, f, 0, d), fold_mean(R, f, 0, d, "mcfadden")] for d in DS}
        r["max_abs_delta_auc"] = float(max(abs(v) for v in r["delta_auc"].values()))
        r["H_auc"] = bool(r["evaluable"] and r["max_abs_delta_auc"] < C.AUC_DELTA)
        res[f] = r
    return res


def bootstrap_delta(f, B=1000):
    p = f"{C.BASE}/runs/auc/{f}_s0.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    y = z["y"].astype(int)
    rng = np.random.default_rng(0)
    out = {}
    base = z["oof_d12"]
    draws = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    for d in DNEW:
        o = z[f"oof_d{d}"]
        point = roc_auc_score(y, o) - roc_auc_score(y, base)
        v = []
        for i in draws:
            if len(set(y[i].tolist())) < 2:
                continue
            v.append(roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], base[i]))
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[str(d)] = {"pooled_oof_delta_auc": float(point), "ci95": [float(lo), float(hi)],
                       "ci_within_pm_0.01": bool(lo > -C.AUC_DELTA and hi < C.AUC_DELTA)}
    return out


def secondary_auc(R):
    out = {}
    for f in C.MODELS:
        if (f, 0) not in R or "secondary" not in R[(f, 0)]:
            continue
        sec = R[(f, 0)]["secondary"]
        o = {"coding_loss": {str(d): sec["coding_loss"][str(d)]["auc"][0] for d in DS},
             "coding_loss_delta": {str(d): sec["coding_loss"][str(d)]["auc"][0] - sec["coding_loss"]["12"]["auc"][0] for d in DNEW},
             "grouped": {d: {"auc": sec["grouped"][d]["auc"][0], "mcfadden": sec["grouped"][d]["mcfadden"][0],
                             "ungrouped_auc": fold_mean(R, f, 0, int(d))} for d in sec["grouped"]},
             "n_distinct_problems": sec.get("n_distinct_problems"),
             "ungrouped_test_rows_with_problem_in_train_folds": sec.get("ungrouped_test_rows_with_problem_in_train_folds"),
             "within_ops": {}}
        for k, v in sec["within_ops"].items():
            row = {"n": v["n"], "accuracy": v["accuracy"]}
            if "12" in v:
                row["auc"] = {d: v[str(d)]["auc"][0] for d in DS if str(d) in v}
                row["mcfadden"] = {d: v[str(d)]["mcfadden"][0] for d in DS if str(d) in v}
                row["delta_auc"] = {d: v[str(d)]["auc"][0] - v["12"]["auc"][0] for d in DNEW if str(d) in v}
            o["within_ops"][k] = row
        o["bootstrap_delta_auc_seed0"] = bootstrap_delta(f)
        out[f] = o
    return out


# ------------------------------------------------------------------ W1 part
def w1_stats(rows, block, pair, d, N, tag="sub"):
    v = [r["w1"] for r in rows if r["block"] == block and r["pair"] == pair and r["d"] == d and r["N"] == N
         and r["tag"] == tag and r["code"] == 1]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def w1_part(rows):
    res = {"blocks": {}}
    for block in ["pool", "fresh", "eval"]:
        N = C.W1_N_BLOCK[block]
        tab = {p: {str(d): w1_stats(rows, block, p, d, N) for d in DS} for p in WITHIN + BETWEEN}
        b = {"N": N, "table": tab}
        ratio, snr, seven = {}, {}, {}
        for p in WITHIN + BETWEEN:
            if tab[p]["12"] and all(tab[p][str(d)] for d in DS):
                ratio[p] = {str(d): tab[p][str(d)]["mean"] / tab[p]["12"]["mean"] for d in DS}
        for d in DS:
            wb, wd = tab["BFS-BFS"][str(d)], tab["DFS-DFS"][str(d)]
            s = {}
            if wb:
                for p in ["BFS-MIXED", "BFS-DFS"]:
                    if tab[p][str(d)]:
                        s[f"{p}/BFS-BFS"] = tab[p][str(d)]["mean"] / wb["mean"]
            if wd and tab["DFS-MIXED"][str(d)]:
                s["DFS-MIXED/DFS-DFS"] = tab["DFS-MIXED"][str(d)]["mean"] / wd["mean"]
            snr[str(d)] = s
            if "BFS-MIXED/BFS-BFS" in s and "BFS-DFS/BFS-BFS" in s:
                seven[str(d)] = bool(min(s["BFS-MIXED/BFS-BFS"], s["BFS-DFS/BFS-BFS"]) >= 7)
        b["ratio_to_d12"], b["snr"], b["sevenfold_both_shifts"] = ratio, snr, seven
        rho = {}
        for d in DS:
            wmean = [tab[p][str(d)]["mean"] for p in WITHIN if tab[p][str(d)]]
            bmean = [tab[p][str(d)]["mean"] for p in BETWEEN if tab[p][str(d)]]
            if len(wmean) == 3 and len(bmean) == 3:
                rho[str(d)] = float(np.mean(wmean) / np.mean(bmean))
        b["relative_noise_floor_mean_within_over_mean_between"] = rho
        if block == "eval":
            b["full_sets_between"] = {p: {str(d): w1_stats(rows, "eval", p, d, 2000, "full") for d in DS} for p in BETWEEN}
        res["blocks"][block] = b
    # the within-measure W1 on the pools increases strictly with d
    tab = res["blocks"]["pool"]["table"]
    ev = all(tab[p][str(d)] and tab[p][str(d)]["n"] >= MIN_W1_SEEDS for p in WITHIN for d in DS)
    inc = {p: (all(tab[p][str(DS[i])]["mean"] < tab[p][str(DS[i + 1])]["mean"] for i in range(len(DS) - 1))
               if all(tab[p][str(d)] for d in DS) else None) for p in WITHIN}
    res["H_w1_evaluable"] = bool(ev)
    res["within_strictly_increasing"] = inc
    res["H_w1"] = bool(ev and all(inc.values()))
    # N sweep (pool data, within pairs): N in NSWEEP_N plus 2000 from block pool; alpha = -OLS slope of log W1 on log N
    sweep = {}
    for p in WITHIN:
        sp = {}
        for d in DS:
            pts = []
            for N in sorted(set(C.NSWEEP_N + [2000])):
                st = w1_stats(rows, "pool" if N == 2000 else "nsweep", p, d, N)
                if st:
                    pts.append((N, st["mean"], st["sd"], st["n"]))
            fit = None
            if len(pts) >= 3:
                x, yv = np.log([q[0] for q in pts]), np.log([q[1] for q in pts])
                slope = float(np.polyfit(x, yv, 1)[0])
                fit = {"alpha": -slope, "n_points": len(pts)}
            sp[str(d)] = {"points": [{"N": q[0], "mean": q[1], "sd": q[2], "n": q[3]} for q in pts], "fit": fit}
        sweep[p] = sp
    res["n_sweep"] = sweep
    res["n_not_optimal_after_rescue"] = int(sum(r["code"] != 1 for r in rows))
    res["n_rescued"] = int(sum(r["rescued"] for r in rows))
    res["n_values"] = len(rows)
    return res


# ------------------------------------------------------------------ self-check
def mw_auc(y, p):
    r = rankdata(p)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def selfcheck(R, rows, gate):
    V, issues = {}, []
    # V1/V2 recompute fold metrics from saved OOF predictions; fold structure
    worst_auc, worst_mcf, nchk, fold_ok = 0.0, 0.0, 0, True
    for (f, s), J in R.items():
        z = np.load(f"{C.BASE}/runs/auc/{f}_s{s}.npz")
        y, fid = z["y"].astype(int), z["fold_id"]
        if not (set(np.unique(fid).tolist()) == set(range(5)) and len(fid) == len(y)):
            fold_ok = False
        rate = y.mean()
        for k in range(5):
            if abs(y[fid == k].mean() - rate) > 0.01:
                fold_ok = False
        for d in DS:
            o = z[f"oof_d{d}"]
            if not np.isfinite(o).all():
                fold_ok = False
            for k, fo in enumerate(J["dims"][str(d)]["folds"]):
                te, tr = fid == k, fid != k
                a = mw_auc(y[te], o[te])
                mc = 1 - ll(y[te], o[te]) / ll(y[te], np.full(te.sum(), y[tr].mean()))
                worst_auc = max(worst_auc, abs(a - fo["auc"])); worst_mcf = max(worst_mcf, abs(mc - fo["mcfadden"]))
                nchk += 1
    V["V1_auc_recomputed_from_oof"] = {"n_folds_checked": nchk, "max_abs_diff_auc": worst_auc,
                                       "max_abs_diff_mcfadden": worst_mcf,
                                       "pass": nchk > 0 and worst_auc < 1e-9 and worst_mcf < 1e-7}
    V["V2_fold_structure"] = {"pass": fold_ok and nchk > 0}
    # V3 recompute W1 on the pools (all seeds, BFS-BFS and BFS-DFS, d = 12 and 34) with an independent solver
    # (Hungarian assignment, exact for two uniform samples of equal size)
    feats = {m: np.load(C.feat_path("pool", m))["X"] for m in C.MEASURES}
    allv = np.concatenate([feats[m] for m in C.MEASURES], axis=0)
    mu = allv.sum(0) / len(allv)
    sd = np.sqrt(((allv - mu) ** 2).sum(0) / len(allv)); sd[sd == 0] = 1
    worst, nw, recs = 0.0, 0, []
    for s in range(C.W1_SEEDS):
        p = f"{C.BASE}/runs/w1/pool_s{s}"
        if not os.path.exists(p + ".json"):
            continue
        J, I = C.jload(p + ".json"), np.load(p + ".npz")
        for e in J["entries"]:
            if e["pair"] in ("BFS-BFS", "BFS-DFS") and e["d"] in (12, 34) and e["tag"] == "sub":
                key = f"{e['pair']}_N{e['N']}_sub"
                A_ = (feats[e["a"]][I[key + "_i1"]][:, :e["d"]] - mu[:e["d"]]) / sd[:e["d"]]
                B_ = (feats[e["b"]][I[key + "_i2"]][:, :e["d"]] - mu[:e["d"]]) / sd[:e["d"]]
                sq = (A_ ** 2).sum(1)[:, None] + (B_ ** 2).sum(1)[None, :] - 2.0 * A_ @ B_.T  # squared Euclidean distances, computed without scipy's cdist
                M = np.sqrt(np.clip(sq, 0.0, None))
                r_, c_ = linear_sum_assignment(M)
                w = float(M[r_, c_].mean())
                ref = [r for r in rows if r["block"] == "pool" and r["seed"] == s and r["pair"] == e["pair"]
                       and r["d"] == e["d"] and r["N"] == e["N"] and r["tag"] == "sub"][0]["w1"]
                worst = max(worst, abs(w - ref)); nw += 1
                recs.append({"seed": s, "pair": e["pair"], "d": e["d"], "hungarian": w, "reported": ref})
    V["V3_w1_recomputed_hungarian"] = {"n_checked": nw, "max_abs_diff": worst, "pass": nw > 0 and worst < 1e-6,
                                       "values": recs}
    # V4 monotonicity in d (holds by construction for nested coordinates and identical subsamples)
    viol, groups = [], {}
    for r in rows:
        if r["code"] == 1:
            groups.setdefault((r["block"], r["seed"], r["pair"], r["N"], r["tag"]), {})[r["d"]] = r["w1"]
    for k, g in groups.items():
        ds = sorted(g)
        for i in range(len(ds) - 1):
            if g[ds[i + 1]] < g[ds[i]] - 1e-9:
                viol.append([*k, ds[i], ds[i + 1]])
    V["V4_w1_monotone_in_d_per_subsample"] = {"n_groups": len(groups), "violations": viol[:20],
                                              "pass": len(groups) > 0 and not viol}
    # V5 subsample index checks (within halves disjoint, no repeated trajectory)
    bad = []
    for p in sorted(glob.glob(f"{C.BASE}/runs/w1/*.npz")):
        I = np.load(p)
        for k in I.files:
            if k.endswith("_i1"):
                i1, i2 = I[k], I[k[:-3] + "_i2"]
                a_, b_ = k.split("_")[0].split("-")
                if len(set(i1.tolist())) != len(i1) or len(set(i2.tolist())) != len(i2):
                    bad.append([os.path.basename(p), k, "repeat"])
                if a_ == b_ and set(i1.tolist()) & set(i2.tolist()):
                    bad.append([os.path.basename(p), k, "within halves overlap"])
    V["V5_subsample_indices"] = {"issues": bad[:20], "pass": not bad}
    # V6 split leakage: duplicate problems across folds do not drive the AUC
    lk = {}
    for f in C.MODELS:
        if (f, 0) in R and "secondary" in R[(f, 0)]:
            for d, g in R[(f, 0)]["secondary"]["grouped"].items():
                lk[f"{f}_d{d}"] = abs(g["auc"][0] - fold_mean(R, f, 0, int(d)))
    thr = {"f1": 0.005, "f2": 0.01}  # f2 (n = 3,000): two 5-fold partitions differ by about 0.003 in mean AUC
    V["V6_split_leakage_grouped_vs_ungrouped"] = {"abs_diff_auc": lk, "threshold": thr,
                                                  "pass": len(lk) == 2 * len(C.MODELS)
                                                  and all(v < thr[k.split("_")[0]] for k, v in lk.items())}
    # V7 the d = 12 values of auc_task.py and w1_task.py equal those of gate.py (same computation, two scripts)
    cons, cok = {}, True
    for f, g in gate.get("checks", {}).get("G3", {}).get("models", {}).items():
        if g.get("required") and (f, 0) in R:
            cons[f"G3_{f}"] = abs(g["auc"] - fold_mean(R, f, 0, 12)); cok &= cons[f"G3_{f}"] < 1e-12
    for name, g in gate.get("checks", {}).get("G4", {}).get("pairs", {}).items():
        for s, v in enumerate(g["values"][:C.W1_SEEDS]):
            ref = [r["w1"] for r in rows if r["block"] == "pool" and r["seed"] == s and r["pair"] == name
                   and r["d"] == 12 and r["N"] == 2000 and r["tag"] == "sub"]
            if ref:
                cons[f"G4_{name}_s{s}"] = abs(v - ref[0]); cok &= cons[f"G4_{name}_s{s}"] < 1e-9
    V["V7_consistency_with_gate"] = {"max_abs_diff": max(cons.values()) if cons else None, "n": len(cons),
                                     "pass": bool(cok and len(cons) > 0)}
    # V8 sanity
    s_ok = True
    for (f, s), J in R.items():
        for d in DS:
            for fo in J["dims"][str(d)]["folds"]:
                s_ok &= 0.5 <= fo["auc"] <= 1.0
        if not C.SMOKE:
            s_ok &= abs(J["accuracy"] - C.APPJ_ACC[f]) < 1e-9 and J["n"] == C.APPJ_N[f]
    s_ok &= all(np.isfinite(r["w1"]) and r["w1"] >= 0 for r in rows)
    V["V8_sanity"] = {"pass": bool(s_ok)}
    passed = all(v["pass"] for v in V.values())
    return V, passed


def main():
    os.makedirs(OUT, exist_ok=True)
    gate = C.jload(f"{OUT}/gate.json", {})
    gate_ok = bool(gate.get("pass"))
    R = load_auc() if gate_ok else {}
    rows, nfiles, resc = load_w1() if gate_ok else ([], 0, {})
    a = auc_part(R) if gate_ok else {f: {"evaluable": False, "H_auc": None} for f in C.MODELS}
    w = w1_part(rows) if gate_ok else {"H_w1": None, "H_w1_evaluable": False, "blocks": {}}
    if gate_ok:
        V, vpass = selfcheck(R, rows, gate)
    else:
        V, vpass = {"gate": {"pass": False, "note": "gate failed or missing; nothing computed"}}, False
    secs = sum(C.jload(p).get("seconds", 0) for p in glob.glob(f"{C.BASE}/runs/auc/*.json") + glob.glob(f"{C.BASE}/runs/w1/*.json"))
    secs += sum(C.jload(p).get("seconds", 0) for p in glob.glob(f"{C.BASE}/data/features/*.meta.json") + glob.glob(f"{C.BASE}/data/raw/*.meta.json"))
    secs += gate.get("seconds", 0) + C.jload(f"{C.BASE}/runs/w1_rescue.json", {}).get("seconds", 0)
    S = {"verify_pass": vpass, "generated": C.now(), "smoke": C.SMOKE,
         "hypothesis": "AUC changes by < 0.01 (d = 16, 24, 34 vs 12) for f1 and f2 while the within-measure W1 noise floor "
                       "grows with d",
         "H": {"H_auc_f1": a["f1"].get("H_auc"), "H_auc_f2": a["f2"].get("H_auc"), "H_w1": w.get("H_w1")},
         "gate": {"pass": gate_ok, **{k: v.get("pass") for k, v in gate.get("checks", {}).items()}},
         "phi_order": C.phi_order(), "dims": DS, "auc": a, "auc_secondary": secondary_auc(R) if gate_ok else {},
         "w1": w, "verify": V, "n_w1_files": nfiles,
         "compute": {"cpu_task_seconds": secs, "gpu_hours": 0.0, "analysis_seconds": round(time.time() - T0, 1)}}
    C.jdump(S, f"{OUT}/summary.json.partial")
    C.jdump({"pass": vpass, "checks": V, "generated": C.now()}, f"{OUT}/verify.json")
    # tables
    if gate_ok:
        with open(f"{OUT}/auc_table.csv", "w") as fh:
            fh.write("model,d,auc_mean,auc_sd,delta_auc,mcfadden_mean,delta_mcfadden,n_reps\n")
            for f in C.MODELS:
                r = a[f]
                for d in DS:
                    if r.get("seeds"):
                        fh.write(f"{f},{d},{r['auc'][str(d)][0]:.6f},{r['auc'][str(d)][1]:.6f},"
                                 f"{0 if d == 12 else r['delta_auc'][str(d)]:.6f},{r['mcfadden'][str(d)][0]:.6f},"
                                 f"{0 if d == 12 else r['delta_mcfadden'][str(d)]:.6f},{r['n_seeds']}\n")
        with open(f"{OUT}/w1_table.csv", "w") as fh:
            fh.write("block,pair,N,d,mean,sd,n_seeds\n")
            for blk, b in w["blocks"].items():
                for p, tab in b["table"].items():
                    for d, st in tab.items():
                        if st:
                            fh.write(f"{blk},{p},{b['N']},{d},{st['mean']:.6f},{st['sd']:.6f},{st['n']}\n")
    os.replace(f"{OUT}/summary.json.partial", f"{OUT}/summary.json")
    print("verify pass:", vpass, f"| {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
