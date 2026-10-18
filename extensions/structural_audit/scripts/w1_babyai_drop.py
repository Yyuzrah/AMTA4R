"""Family E without the initial-state size T9 (n_objs), which is constant within each BabyAI instruction kind (9 for
goto/pickup, 10 otherwise), and additionally without chi11_manip (number of manipulation subgoals, equal to its most
frequent value within a kind on at least 99.3% of the levels). Same estimator as w1.py 'main' (family z-scoring, N,
seeds). Usage: python w1_babyai_drop.py [n_procs]. Output: results/babyai_drop.json"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1 as W

DROPS = {"E_noT9": ["n_objs"], "E_noT9_noManip": ["n_objs", "chi11_manip"]}


def init(drop):
    W.BAB = [c for c in W.BAB if c not in DROPS[drop]]
    fam = W.load_families(); fam = {"E": fam["E"]}
    P = W.prepare(fam, "main")
    mu, sd = W.zstats([P["E"]["X"][g] for g in P["E"]["groups"]])
    W.STATE.update(P=P, Z={"E": {g: (P["E"]["X"][g] - mu) / sd for g in P["E"]["groups"]}}, variant="main")


def main():
    out = {}
    for drop in DROPS:
        init(drop)
        T = W.tasks_for(W.STATE["P"])
        with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8, initializer=init, initargs=(drop,)) as pool:
            res = pool.map(W.run_task, T)
        P = W.STATE["P"]["E"]; rows = []
        for a, b in P["pairs"]:
            N = min(W.NMAX, P["nunits"][a] // 2, P["nunits"][b] // 2)
            bt = [r["w1"] for r in res if r["kind"] == "between" and r["a"] == a and r["b"] == b]
            wa = [r["w1"] for r in res if r["kind"] == "within" and r["a"] == a and r["N"] == N]
            wb = [r["w1"] for r in res if r["kind"] == "within" and r["a"] == b and r["N"] == N]
            Wm, sd = float(np.mean(bt)), float(np.std(bt, ddof=1)); B = max(np.mean(wa), np.mean(wb))
            rows.append(dict(a=a, b=b, N=N, n_seeds=len(bt), W=Wm, W_sd=sd, B_a=float(np.mean(wa)), B_b=float(np.mean(wb)),
                             R=float(Wm / B), separated=bool(Wm / B >= 3 and sd / Wm <= 0.10)))
        out[drop] = dict(cols=W.STATE["P"]["E"]["cols"], pairs=rows, n_separated=sum(r["separated"] for r in rows),
                         warnings=sum(bool(r["warning"]) for r in res))
        print(drop, out[drop]["n_separated"], "/", len(rows), [(r["a"], r["b"], round(r["R"], 2)) for r in rows], flush=True)
    json.dump(out, open(os.path.join(W.RES, "babyai_drop.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
