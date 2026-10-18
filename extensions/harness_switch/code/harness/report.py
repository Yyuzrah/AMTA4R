"""Report of one run of both harnesses, written to <main_dir>/report.json. Keys:
  episodes: episodes finished in both harnesses;
  serving: checks of the proxy logs per harness (served model, sampling keys the harness set, non-200 responses,
    tokens, finish reasons);
  tool_call_parsing: share of tool calls that the feature extraction recognises;
  W1_<variant>: the summaries written by w1.py;
  gold_patch: instances whose gold patch passes;
  resolve_native, resolve_common: resolve rates on the instances whose gold patch passes, with each harness's own
    patch / with the final working tree for both (percentile bootstrap CIs, paired bootstrap CI of the difference,
    exact McNemar test);
  smd_signs: sign of the standardized mean difference of C7_failed_edits and T1_max_depth;
  run_stats: exit statuses, wall times, retries and call counts.
Proxy logs of the jobs listed in data/setup_test_jobs.json (if present) are left out.
Reads <main_dir>/features_main.csv, <main_dir>/w1/summary_<variant>.json, results/proxy/*.jsonl,
eval/<runs_tag>/ and runs/<runs_tag>/ (all under $HARNESS_DIR).
Usage: python report.py <runs_tag> <main_dir>"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

HARNESS_DIR = os.environ["HARNESS_DIR"]; TAG = sys.argv[1]; OUT = sys.argv[2]


def load_res(cond):
    r = {}
    for f in glob.glob(f"{HARNESS_DIR}/eval/{TAG}/{cond}/*/result.json"):
        j = json.load(open(f)); r[j["instance_id"]] = bool(j["resolved"])
    return r


def rates(a, b, ids, B=10000):
    x = np.array([a[i] for i in ids], float); y = np.array([b[i] for i in ids], float)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(ids), size=(B, len(ids)))
    bx, by = x[idx].mean(1), y[idx].mean(1); d = by - bx
    nb, nc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    p = binomtest(nb, nb + nc, 0.5).pvalue if nb + nc > 0 else 1.0
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(n=len(ids), mini=float(x.mean()), mini_ci=q(bx), oh=float(y.mean()), oh_ci=q(by), diff_oh_minus_mini=float(y.mean() - x.mean()),
                diff_ci=q(d), only_mini=nb, only_oh=nc, both=int(((x == 1) & (y == 1)).sum()), mcnemar_p=float(p))


def main():
    rep = {}
    feats = pd.read_csv(f"{OUT}/features_main.csv")
    main_ids = [l.strip() for l in open(f"{HARNESS_DIR}/data/instances_main.txt") if l.strip()]
    done = {h: set(feats[feats.harness == h].instance_id) for h in ("mini", "oh")}
    common = sorted(done["mini"] & done["oh"] & set(main_ids))
    gave_up = {h: sorted(i for i in main_ids if i not in done[h]) for h in ("mini", "oh")}
    rep["episodes"] = dict(n_main=len(main_ids), done_mini=len(done["mini"]), done_oh=len(done["oh"]), common=len(common),
                     coverage=len(common) / len(main_ids), not_done=gave_up, pass_min100=len(common) >= 100,
                     pass_90pct=len(common) >= 0.9 * len(main_ids))
    # serving checks from the proxy logs
    px = []
    for f in glob.glob(f"{HARNESS_DIR}/results/proxy/*.jsonl"):
        job = os.path.basename(f).split("_")[0]
        for l in open(f):
            try:
                r = json.loads(l); r["file"] = os.path.basename(f); px.append(r)
            except Exception:
                pass
    px = pd.DataFrame(px)
    test_jobs = set(json.load(open(f"{HARNESS_DIR}/data/setup_test_jobs.json"))) if os.path.exists(f"{HARNESS_DIR}/data/setup_test_jobs.json") else set()
    if len(px):
        px = px[~px.file.str.split("_").str[0].isin(test_jobs)]
        v2 = {}
        px["ctx_exceeded"] = px.get("error", pd.Series("", index=px.index)).fillna("").astype(str).str.contains("maximum context length")
        for h, g in px.groupby("harness"):
            u = g["usage"].dropna()
            serving_fail = (g.status != 200) & ~g.ctx_exceeded
            v2[h] = dict(requests=int(len(g)), non200=int((g.status != 200).sum()), ctx_exceeded_400=int(g.ctx_exceeded.sum()),
                         serving_failures=int(serving_fail.sum()), non200_share=float(serving_fail.mean()),
                         models=sorted(set(g.model.dropna())), streamed=int(g.get("stream", pd.Series(dtype=bool)).fillna(False).sum()),
                         prompt_tokens=int(sum(x.get("prompt_tokens", 0) for x in u)), completion_tokens=int(sum(x.get("completion_tokens", 0) for x in u)),
                         finish_reasons=g.finish_reason.value_counts().to_dict(),
                         harness_set_keys=sorted({k for d in g.harness_set.dropna() for k in d}))
        rep["serving"] = v2; rep["serving"]["pass"] = all(v["non200_share"] <= 0.05 for v in v2.values())
    cov = feats[feats.instance_id.isin(common)].groupby("harness")[["n_tool_calls", "n_known"]].sum()
    rep["tool_call_parsing"] = dict(parse_coverage=(cov.n_known / cov.n_tool_calls.clip(lower=1)).to_dict())
    rep["tool_call_parsing"]["pass"] = all(v >= 0.95 for v in rep["tool_call_parsing"]["parse_coverage"].values())
    # W1 summaries written by w1.py
    for v in ("main", "noLEN", "family_c_file_rule"):
        p = f"{OUT}/w1/summary_{v}.json"
        if os.path.exists(p):
            rep["W1_" + v] = json.load(open(p))
    m = rep.get("W1_main")
    if m:
        smd = {a["coord"]: a["smd_oh_minus_mini"] for a in m["attribution"]}
        rep["smd_signs"] = {"C7_failed_edits_higher_OH": dict(smd=smd["C7_failed_edits"], hit=smd["C7_failed_edits"] > 0),
                     "T1_max_depth_higher_OH": dict(smd=smd["T1_max_depth"], hit=smd["T1_max_depth"] > 0)}
    # gold-patch check; resolve rates
    gold = load_res("gold")
    valid = [i for i in common if gold.get(i) is True]
    rep["gold_patch"] = dict(gold_graded=sum(i in gold for i in common), gold_valid=len(valid), gold_invalid=[i for i in common if gold.get(i) is False])
    mn, mc, on = load_res("mini_native"), load_res("mini_common"), load_res("oh_native")
    ids3 = [i for i in valid if i in mn and i in on]; ids4 = [i for i in valid if i in mc and i in on]
    if ids3:
        rep["resolve_native"] = rates(mn, on, ids3)
    if ids4:
        rep["resolve_common"] = rates(mc, on, ids4)
    # run statistics
    meta = []
    for h in ("mini", "oh"):
        for f in glob.glob(f"{HARNESS_DIR}/runs/{TAG}/{h}/*/meta.json"):
            j = json.load(open(f)); j["h"] = h
            d = os.path.dirname(f)
            j["infra_errors"] = len(glob.glob(f"{d}/infra_error_*.json")); j["killed"] = len(glob.glob(f"{d}/killed_*.json"))
            meta.append(j)
    meta = pd.DataFrame(meta)
    meta = meta[meta.instance_id.isin(common)]
    fc = feats[feats.instance_id.isin(common)]
    rep["run_stats"] = {h: dict(exit_status=g.exit_status.value_counts().to_dict(), wall_s_mean=float(g.wall_s.mean()), wall_s_median=float(g.wall_s.median()),
                         infra_retries=int(g.infra_errors.sum()), hard_kills=int(g.killed.sum()),
                         llm_calls_mean=float(fc[fc.harness == h].n_llm_calls.mean()), tool_calls_mean=float(fc[fc.harness == h].n_tool_calls.mean()))
                 for h, g in meta.groupby("h")}
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("W1_")}, indent=1, default=str)[:6000])
    for v in ("main", "noLEN", "family_c_file_rule"):
        if "W1_" + v in rep:
            r = rep["W1_" + v]; print(v, "R=%.3f CV=%.3f W=%.3f B=%.3f n=%d N=%d top3=%s" % (r["R"], r["CV"], r["W"], r["B"], r["n"], r["N"], r["top3"]))


if __name__ == "__main__":
    main()
