"""Result tables of the harness-switch experiment (CPU, a few seconds).
Reads the W1 summaries and per-seed files that harness/w1.py writes for its three variants (one directory) and the
per-instance resolve table, and writes w1_summary.csv, w1_per_seed.csv, coordinate_attribution.csv and
resolve_rates.csv to --out. With --eval-dir it first rebuilds resolve_by_instance.csv from the result.json files
written by harness/eval_patches.py. Resolve rates are computed as in harness/report.py (rates() below is copied from there), on
the instances whose gold patch passes, in sorted instance order.
Usage: python make_tables.py --w1-dir <dir> --out <results_dir> [--eval-dir <eval/main> --instances <list>]"""
import argparse, csv, json, os
import numpy as np
from scipy.stats import binomtest

VARIANTS = [("main", "all_coordinates"), ("noLEN", "without_tool_calls"), ("family_c_file_rule", "family_c_file_rule")]
CONDS = [("gold", "gold"), ("mini_native", "mini_native"), ("mini_common", "mini_common"), ("oh_native", "openhands")]


def rates(a, b, ids, B=10000):
    x = np.array([a[i] for i in ids], float); y = np.array([b[i] for i in ids], float)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(ids), size=(B, len(ids)))
    bx, by = x[idx].mean(1), y[idx].mean(1); d = by - bx
    nb, nc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    p = binomtest(nb, nb + nc, 0.5).pvalue if nb + nc > 0 else 1.0
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(n=len(ids), mini=float(x.mean()), mini_ci=q(bx), oh=float(y.mean()), oh_ci=q(by), diff_oh_minus_mini=float(y.mean() - x.mean()),
                diff_ci=q(d), only_mini=nb, only_oh=nc, both=int(((x == 1) & (y == 1)).sum()), mcnemar_p=float(p))


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)
    print("wrote", path, len(rows), "rows")


def resolve_from_eval(eval_dir, instances, path):
    ids = [l.strip() for l in open(instances) if l.strip()]
    rows = []
    for iid in sorted(ids):
        res = {}
        for cond, _ in CONDS:
            f = os.path.join(eval_dir, cond, iid, "result.json")
            if os.path.exists(f):
                res[cond] = json.load(open(f))
        if len(res) < len(CONDS):  # not finished in both harnesses or not graded
            continue
        rows.append([iid] + [int(bool(res[c]["resolved"])) for c, _ in CONDS] + [res[c]["reason"] for c, _ in CONDS])
    write(path, ["instance_id", "gold_patch_valid", "resolved_mini_native", "resolved_mini_common", "resolved_openhands",
                 "reason_gold", "reason_mini_native", "reason_mini_common", "reason_openhands"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--eval-dir"); ap.add_argument("--instances")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ, seeds, attr = [], [], []
    for v, label in VARIANTS:
        s = json.load(open(os.path.join(a.w1_dir, f"summary_{v}.json")))
        summ.append([label, s["d"], s["n"], s["N"], s["W"], s["W_sd"], s["within_mini"], s["within_mini_sd"],
                     s["within_oh"], s["within_oh_sd"], s["B"], s["R"], s["CV"]])
        for r in attr_rows(s):
            attr.append([label] + r)
        for l in open(os.path.join(a.w1_dir, f"w1_{v}.jsonl")):
            r = json.loads(l)
            seeds.append([label, r["kind"], {"mini|oh": "mini_vs_openhands", "mini": "mini", "oh": "openhands"}[r["h"]],
                          r["seed"], r["n"], r["N"], r["w1"]])
    write(os.path.join(a.out, "w1_summary.csv"),
          ["variant", "coordinates", "n_instances", "N_per_sample", "w1_between", "w1_between_sd", "baseline_mini",
           "baseline_mini_sd", "baseline_openhands", "baseline_openhands_sd", "baseline_max", "R", "CV"], summ)
    write(os.path.join(a.out, "w1_per_seed.csv"), ["variant", "kind", "pools", "seed", "n_instances", "N_per_sample", "w1"],
          sorted(seeds, key=lambda r: ([l for _, l in VARIANTS].index(r[0]), r[3], ["between", "within"].index(r[1]), r[2])))
    write(os.path.join(a.out, "coordinate_attribution.csv"),
          ["variant", "coordinate", "w1_1d", "baseline_1d", "ratio_1d", "smd_openhands_minus_mini", "mean_mini",
           "mean_openhands", "share"], attr)
    res_path = os.path.join(a.out, "resolve_by_instance.csv")
    if a.eval_dir:
        resolve_from_eval(a.eval_dir, a.instances, res_path)
    with open(res_path) as f:
        rows = list(csv.DictReader(f))
    ids = sorted(r["instance_id"] for r in rows if r["gold_patch_valid"] == "1")
    col = lambda c: {r["instance_id"]: r[c] == "1" for r in rows}
    out = []
    for rule, cm in [("native", "resolved_mini_native"), ("common", "resolved_mini_common")]:
        m = rates(col(cm), col("resolved_openhands"), ids)
        out.append([rule, m["n"], m["mini"], *m["mini_ci"], m["oh"], *m["oh_ci"], m["diff_oh_minus_mini"], *m["diff_ci"],
                    m["only_mini"], m["only_oh"], m["both"], m["mcnemar_p"]])
    write(os.path.join(a.out, "resolve_rates.csv"),
          ["patch_rule", "n_instances", "rate_mini", "rate_mini_ci_low", "rate_mini_ci_high", "rate_openhands",
           "rate_openhands_ci_low", "rate_openhands_ci_high", "diff_openhands_minus_mini", "diff_ci_low", "diff_ci_high",
           "resolved_only_mini", "resolved_only_openhands", "resolved_both", "mcnemar_p"], out)


def attr_rows(s):
    return [[x["coord"], x["w1_1d"], x["within_1d_max"], x["ratio_1d"], x["smd_oh_minus_mini"], x["mean_mini"],
             x["mean_oh"], x["share"]] for x in s["attribution"]]


if __name__ == "__main__":
    main()
