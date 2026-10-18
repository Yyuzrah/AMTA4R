# Writes the result tables (CSV and tables.md) from the per-run files under --base (runs/auc/*.json,
# runs/w1/*.json, runs/w1_rescue.json), with the aggregation functions of analyze.py. CSV files hold the values at
# full precision; tables.md rounds them. Default --base and --out: the results/ folder of this experiment.
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)  # extensions/coordinates_12_to_34
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=f"{EXT}/results", help="directory that contains runs/")
ap.add_argument("--out", default=f"{EXT}/results", help="directory for the tables")
args = ap.parse_args()
os.environ["COORD_BASE"] = os.path.abspath(args.base)  # read by common.py at import
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
import common as C  # noqa: E402

DS = C.DIMS
PAIRS = A.WITHIN + A.BETWEEN
BLOCKS = ["pool", "fresh", "eval"]
NAME = {"f1": "Small Deep MIXED RoPE", "f2": "Large Deep BFS APE"}


def write(name, header, rows):
    with open(os.path.join(args.out, name), "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(header)
        for r in rows:
            wr.writerow(["" if v is None else v for v in r])


R = A.load_auc()
rows, _, _ = A.load_w1()
auc = A.auc_part(R)
w1 = A.w1_part(rows)
os.makedirs(args.out, exist_ok=True)

# ---- AUC: per repetition (fold mean and fold s.d. as stored by auc_task.py), then over repetitions
out = []
for f, s in sorted(R):
    J = R[(f, s)]
    for d in DS:
        n = J["dims"][str(d)]
        out.append([f, J["model_name"], J["n"], s, d, n["auc"][0], n["auc"][1], n["mcfadden"][0], n["mcfadden"][1],
                    n["brier"][0], n["brier"][1]])
write("auc_by_repetition.csv", ["model", "model_name", "n", "repetition", "d", "auc", "auc_fold_sd", "mcfadden",
                                "mcfadden_fold_sd", "brier", "brier_fold_sd"], out)
out = []
for f in C.MODELS:
    r = auc[f]
    for d in DS:
        k, new = str(d), d != 12
        out.append([f, R[(f, r["seeds"][0])]["model_name"], r["n"], r["accuracy"], r["n_seeds"], d,
                    r["auc"][k][0], r["auc"][k][1], r["mcfadden"][k][0], r["mcfadden"][k][1],
                    r["delta_auc"][k] if new else None,
                    r["delta_auc_seed_min_max"][k][0] if new else None,
                    r["delta_auc_seed_min_max"][k][1] if new else None,
                    r["share_seeds_abs_delta_below"][k] if new else None,
                    r["delta_mcfadden"][k] if new else None])
write("auc_summary.csv", ["model", "model_name", "n", "accuracy", "n_repetitions", "d", "auc_mean", "auc_sd",
                          "mcfadden_mean", "mcfadden_sd", "delta_auc", "delta_auc_min", "delta_auc_max",
                          "share_abs_delta_auc_below_0.01", "delta_mcfadden"], out)

# ---- W1: every value, then mean and s.d. over seeds
order = {b: i for i, b in enumerate(C.W1_BLOCKS)}
out = [[r["block"], r["pair"], r["same"], r["tag"], r["N"], r["d"], r["seed"], r["w1"], r["code"], r["rescued"]]
       for r in sorted(rows, key=lambda r: (order[r["block"]], r["tag"] != "sub", PAIRS.index(r["pair"]), r["N"],
                                            r["d"], r["seed"]))]
write("w1_by_seed.csv", ["block", "pair", "within", "sample", "N", "d", "seed", "w1", "ot_result_code",
                         "recomputed"], out)
out = []
for b in BLOCKS:
    B = w1["blocks"][b]
    for p in PAIRS:
        for d in DS:
            st = B["table"][p][str(d)]
            if st:
                out.append([b, p, p in A.WITHIN, "sub", B["N"], d, st["mean"], st["sd"], st["n"],
                            B["ratio_to_d12"].get(p, {}).get(str(d))])
    for p, tab in B.get("full_sets_between", {}).items():
        for d in DS:
            st = tab[str(d)]
            if st:
                out.append([b, p, False, "full", 2000, d, st["mean"], None, st["n"], None])
write("w1_summary.csv", ["block", "pair", "within", "sample", "N", "d", "mean", "sd", "n_seeds", "ratio_to_d12"], out)
out = []
for b in BLOCKS:
    B = w1["blocks"][b]
    for d in DS:
        s = B["snr"].get(str(d), {})
        out.append([b, d, B["relative_noise_floor_mean_within_over_mean_between"].get(str(d)),
                    s.get("BFS-MIXED/BFS-BFS"), s.get("BFS-DFS/BFS-BFS"), s.get("DFS-MIXED/DFS-DFS"),
                    B["sevenfold_both_shifts"].get(str(d))])
write("w1_ratios.csv", ["block", "d", "mean_within_over_mean_between", "bfs_mixed_over_bfs_bfs",
                        "bfs_dfs_over_bfs_bfs", "dfs_mixed_over_dfs_dfs", "bfs_shifts_at_least_7x_within_bfs"], out)

# ---- W1 against N (within pairs on the pools)
out, fits = [], []
for p in A.WITHIN:
    for d in DS:
        sp = w1["n_sweep"][p][str(d)]
        for q in sp["points"]:
            out.append([p, d, q["N"], "pool" if q["N"] == C.W1_N else "nsweep", q["mean"], q["sd"], q["n"]])
        fit = sp["fit"] or {}
        fits.append([p, d, fit.get("alpha"), fit.get("n_points")])
write("w1_n_dependence.csv", ["pair", "d", "N", "block", "mean", "sd", "n_seeds"], out)
write("w1_n_dependence_alpha.csv", ["pair", "d", "alpha", "n_points"], fits)


# ---- tables.md
def f3(st):
    return "n/a" if not st else f"{st['mean']:.3f} ± {st['sd']:.3f}"


L = ["# From 12 to 34 coordinates on Countdown: result tables", "",
     "Rounded from the CSV files of this folder (written by `scripts/export_tables.py`). "
     + "; ".join(f"{f} = {NAME[f]}" for f in C.MODELS) + ".", "",
     f"## Out-of-fold ROC-AUC and McFadden pseudo-R² of the App. J classifier on Phi_d ({C.N_CV_REPS} repetitions)",
     "", "Mean ± s.d. over repetitions; Delta = value(Phi_d) - value(Phi_12), paired by repetition.", "",
     "| model | n | d | ROC-AUC | Delta AUC | Delta AUC, min to max over repetitions | McFadden | Delta McFadden |",
     "|---|---|---|---|---|---|---|---|"]
for f in C.MODELS:
    r = auc[f]
    for d in DS:
        k = str(d)
        if d == 12:
            L.append(f"| {f} | {r['n']} | {d} | {r['auc'][k][0]:.4f} ± {r['auc'][k][1]:.4f} | - | - | "
                     f"{r['mcfadden'][k][0]:.4f} ± {r['mcfadden'][k][1]:.4f} | - |")
        else:
            lo, hi = r["delta_auc_seed_min_max"][k]
            L.append(f"| {f} | {r['n']} | {d} | {r['auc'][k][0]:.4f} ± {r['auc'][k][1]:.4f} | "
                     f"{r['delta_auc'][k]:.4f} | {lo:.4f} to {hi:.4f} | "
                     f"{r['mcfadden'][k][0]:.4f} ± {r['mcfadden'][k][1]:.4f} | {r['delta_mcfadden'][k]:.4f} |")
TITLE = {"pool": "the pools of the three training sets (App. G)", "fresh": "the reference trajectories of App. J",
         "eval": "the evaluation sets"}
for b in BLOCKS:
    B = w1["blocks"][b]
    L += ["", f"## Exact W1 on {TITLE[b]}, N = {B['N']:,} per side ({C.W1_SEEDS} seeds)", "",
          "Mean ± s.d. over seeds; within = two disjoint samples of one measure.", "",
          "| pair | " + " | ".join(f"d = {d}" for d in DS) + " | ratio d34/d12 |", "|---|" + "---|" * (len(DS) + 1)]
    for p in PAIRS:
        ratio = B["ratio_to_d12"].get(p, {}).get("34")
        L.append(f"| {p}{' (within)' if p in A.WITHIN else ''} | "
                 + " | ".join(f3(B["table"][p][str(d)]) for d in DS)
                 + f" | {'n/a' if ratio is None else f'{ratio:.2f}'} |")
    fs = B.get("full_sets_between")
    if fs:
        L += ["", "All 2,000 vs all 2,000 trajectories (one value): " + "; ".join(
            f"{p}: " + ", ".join(f"d = {d} {fs[p][str(d)]['mean']:.3f}" for d in DS if fs[p][str(d)]) for p in fs) + "."]
L += ["", "## W1 ratios", "", "| block | d | mean within / mean between | BFS-MIXED / BFS-BFS | BFS-DFS / BFS-BFS | "
      "DFS-MIXED / DFS-DFS |", "|---|---|---|---|---|---|"]
for b in BLOCKS:
    B = w1["blocks"][b]
    for d in DS:
        s = B["snr"].get(str(d), {})
        rho = B["relative_noise_floor_mean_within_over_mean_between"].get(str(d))
        L.append(f"| {b} | {d} | {rho:.3f} | {s['BFS-MIXED/BFS-BFS']:.2f} | {s['BFS-DFS/BFS-BFS']:.2f} | "
                 f"{s['DFS-MIXED/DFS-DFS']:.2f} |")
NS = sorted(set(C.NSWEEP_N + [C.W1_N]))
L += ["", "## Within-measure W1 on the pools against N", "",
      "Mean over seeds; alpha = -(least-squares slope of log mean W1 on log N).", "",
      "| pair | d | " + " | ".join(f"N = {N:,}" for N in NS) + " | alpha |", "|---|---|" + "---|" * (len(NS) + 1)]
for p in A.WITHIN:
    for d in DS:
        sp = w1["n_sweep"][p][str(d)]
        pts = {q["N"]: q["mean"] for q in sp["points"]}
        L.append(f"| {p} | {d} | " + " | ".join("n/a" if N not in pts else f"{pts[N]:.3f}" for N in NS)
                 + f" | {'n/a' if not sp['fit'] else format(sp['fit']['alpha'], '.3f')} |")
with open(os.path.join(args.out, "tables.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("tables written to", args.out)
