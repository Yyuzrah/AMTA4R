"""Aggregates results/tasks_<variant>.jsonl (written by w1.py) and applies the separation rule of App. M.2: a pair
is separated if R = W1 between / max of the two within-group baselines >= 3 and the seed CV of the between-group W1
is <= 0.10. Also counts the separated pairs overall and per family, and computes the 1-D W1 attribution per
coordinate (main variant).
Writes results/results.json, results/table_pairs.md, results/table_families.tex and fig_ratios.pdf (+ .png)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "scripts"))
import w1 as W

FAM_ORDER = ["A", "B", "B2", "C_H1", "C_H2", "D", "E"]
FAM_LABEL = {"A": "ALFWorld\ntypes", "B": "ETO\nenvs", "B2": "WebShop\nhum./GPT",
             "C_H1": "Harness\nshared", "C_H2": "Harness\nall", "D": "Bridge\nskills", "E": "BabyAI\nkinds"}
FAM_LONG = {"A": "ALFWorld task types (ETO)", "B": "ETO environments", "B2": "WebShop human vs GPT-4",
            "C_H1": "Harness, shared issues", "C_H2": "Harness, all issues", "D": "Bridge/ECoT skills (VLA)",
            "E": "BabyAI oracle kinds"}
SHORT = {"pick_and_place_simple": "simple", "look_at_obj_in_light": "look", "pick_clean_then_place_in_recep": "clean",
         "pick_heat_then_place_in_recep": "heat", "pick_cool_then_place_in_recep": "cool",
         "pick_two_obj_and_place": "two", "SWEagent_llama70b": "SWE-agent", "OpenHands_qwen3coder480b": "OpenHands",
         "pick up seq go to": "seq", "alfworld": "ALFWorld", "sciworld": "SciWorld", "webshop": "WebShop",
         "human": "human", "gpt": "GPT-4"}


def load(variant):
    p = os.path.join(RES, "tasks_%s.jsonl" % variant)
    if not os.path.exists(p):
        return None, None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    design = json.load(open(os.path.join(RES, "design_%s.json" % variant)))
    return pd.DataFrame(rows), design


def summarize(df, design, variant):
    out = []
    for fam in FAM_ORDER:
        d = design[fam]
        for a, b in d["pairs"]:
            N = min(W.NMAX, d["nunits"][a] // 2, d["nunits"][b] // 2)
            bt = df[(df.kind == "between") & (df.family == fam) & (df.a == a) & (df.b == b) & (df.N == N)]
            if variant == "pairz":
                wa = df[(df.kind == "within") & (df.family == fam) & (df.a == a) & (df.b == b) & (df.N == N)]
                wb = df[(df.kind == "within") & (df.family == fam) & (df.a == b) & (df.b == a) & (df.N == N)]
            else:
                wa = df[(df.kind == "within") & (df.family == fam) & (df.a == a) & (df.N == N)]
                wb = df[(df.kind == "within") & (df.family == fam) & (df.a == b) & (df.N == N)]
            complete = len(bt) == 10 and len(wa) == 10 and len(wb) == 10
            Wm, Wsd = bt.w1.mean(), bt.w1.std(ddof=1)
            Ba, Bb = wa.w1.mean(), wb.w1.mean()
            R = Wm / max(Ba, Bb); Rmean = Wm / (0.5 * (Ba + Bb))
            out.append(dict(family=fam, a=a, b=b, N=int(N), complete=bool(complete), W=Wm, W_sd=Wsd, cv=Wsd / Wm,
                            B_a=Ba, B_a_sd=wa.w1.std(ddof=1), B_b=Bb, B_b_sd=wb.w1.std(ddof=1), R=R, R_meanbase=Rmean,
                            separated=bool(R >= 3 and Wsd / Wm <= 0.10),
                            warnings=int((pd.concat([bt, wa, wb]).warning.fillna("") != "").sum())))
    return pd.DataFrame(out)


def attribution(design):
    """1-D W1 per coordinate between the two full pools, family-standardized (main variant)."""
    W.init("main"); P = W.STATE["P"]; Z = W.STATE["Z"]; res = {}
    for fam in FAM_ORDER:
        cols = P[fam]["cols"]
        for a, b in P[fam]["pairs"]:
            v = {c: float(wasserstein_distance(Z[fam][a][:, j], Z[fam][b][:, j])) for j, c in enumerate(cols)}
            res["%s|%s|%s" % (fam, a, b)] = sorted(v.items(), key=lambda t: -t[1])
    return res


def separation_counts(tab):
    n, k = len(tab), int(tab.separated.sum())
    fam = {f: (int(g.separated.sum()), len(g)) for f, g in tab.groupby("family", sort=False)}
    fam_major = {f: s > n_ / 2 for f, (s, n_) in fam.items()}
    return dict(n_pairs=n, n_separated=k, majority_separated=k > n / 2, per_family={f: dict(separated=s, pairs=n_, majority=fam_major[f]) for f, (s, n_) in fam.items()})


def fig(tabs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    COL = {"main": "#2a78d6", "noS": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, (variant, title) in zip(axes, [("main", "(a) full $\\Phi_D$"), ("noS", "(b) $\\Phi_D$ without surface (S) coordinates")]):
        t = tabs.get(variant)
        if t is None: continue
        for i, fam in enumerate(FAM_ORDER):
            g = t[t.family == fam].reset_index(drop=True)
            k = len(g)
            xs = i + (np.linspace(-0.28, 0.28, k) if k > 1 else np.zeros(1))
            sep = g.R.values >= 3
            # error bar: seed s.d. of the between-group W1 divided by the baseline
            yerr = g.W_sd.values / np.maximum(g.B_a.values, g.B_b.values)
            ax.errorbar(xs, g.R.values, yerr=yerr, fmt="none", ecolor=COL[variant], elinewidth=0.6, alpha=0.7, zorder=2)
            ax.scatter(xs[sep], g.R.values[sep], s=14, color=COL[variant], edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter(xs[~sep], g.R.values[~sep], s=14, facecolors="white", edgecolors=COL[variant], linewidths=0.9, zorder=3)
            ax.text(i, 0.62, "%d/%d" % (int(g.separated.sum()), k), ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.axhline(3, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
        ax.set_yscale("log"); ax.set_ylim(0.55, max(60, float(np.nanmax(t.R)) * 1.3))
        ax.set_xticks(range(len(FAM_ORDER))); ax.set_xticklabels([FAM_LABEL[f] for f in FAM_ORDER], fontsize=5.3)
        ax.set_xlim(-0.5, len(FAM_ORDER) - 0.5)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=7.5, loc="left", color=INK)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("$\\widehat W_1$ between / within-group baseline", fontsize=7)
    axes[0].text(len(FAM_ORDER) - 0.45, 3.15, "3$\\times$", ha="right", va="bottom", fontsize=6.5, color=INK)
    # layout check: no two text artists of one axes overlap
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); clashes = []
    for ax in axes:
        arts = [t for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.texts + [ax.title] if t.get_text()]
        bbs = [(t.get_text(), t.get_window_extent(rend)) for t in arts]
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                if bbs[i][1].overlaps(bbs[j][1]): clashes.append((bbs[i][0], bbs[j][0]))
    print("text overlaps:", clashes)
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


def fmt(x, d=2):
    return ("%." + str(d) + "f") % x


def main():
    tabs, out = {}, {}
    for variant in ["main", "noS", "pairz"]:
        df, design = load(variant)
        if df is None: continue
        t = summarize(df, design, variant); tabs[variant] = t
        out[variant] = dict(separation=separation_counts(t), pairs=t.to_dict(orient="records"), design=design,
                            n_tasks=len(df), n_warnings=int((df.warning.fillna("") != "").sum()))
    out["attribution_main"] = attribution(out["main"]["design"])
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=1, default=float)
    t = tabs["main"]; tn = tabs.get("noS"); tp = tabs.get("pairz")
    lines = ["| Family | Pair | N | W1 between (sd) | baseline a / b | R = W1/max | R (noS) | R (pair z) | top coordinates (1-D W1) | separated |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in t.iterrows():
        top = out["attribution_main"]["%s|%s|%s" % (r.family, r.a, r.b)][:3]
        rn = tn.iloc[i].R if tn is not None else float("nan"); rp = tp.iloc[i].R if tp is not None else float("nan")
        lines.append("| %s | %s vs %s | %d | %s (%s) | %s / %s | %s | %s | %s | %s | %s |" % (
            r.family, SHORT.get(r.a, r.a), SHORT.get(r.b, r.b), r.N, fmt(r.W), fmt(r.W_sd, 3), fmt(r.B_a), fmt(r.B_b),
            fmt(r.R), fmt(rn), fmt(rp), ", ".join("%s %.2f" % (c, v) for c, v in top), "yes" if r.separated else "no"))
    dec = out["main"]["separation"]
    lines += ["", "Main: %d of %d pairs separated. Per family: %s" % (
        dec["n_separated"], dec["n_pairs"],
        "; ".join("%s %d/%d" % (f, v["separated"], v["pairs"]) for f, v in dec["per_family"].items()))]
    for v in ["noS", "pairz"]:
        if v in out:
            dv = out[v]["separation"]
            name = {"noS": "Without surface (S) coordinates (noS)", "pairz": "Per-pair z-scoring (pairz)"}[v]
            lines.append("%s: %d of %d pairs separated." % (name, dv["n_separated"], dv["n_pairs"]))
    open(os.path.join(RES, "table_pairs.md"), "w").write("\n".join(lines) + "\n")
    # compact LaTeX table: one row per family
    tex = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
           "Family & Groups & $d$ & $N$ & pairs sep. & median $R$ & median $R$ w/o S \\\\", "\\midrule"]
    for fam in FAM_ORDER:
        g = t[t.family == fam]; gn = tn[tn.family == fam] if tn is not None else g
        d = out["main"]["design"][fam]
        Ns = sorted(set(g.N)); Nstr = str(Ns[0]) if len(Ns) == 1 else "%d--%d" % (Ns[0], Ns[-1])
        tex.append("%s & %d & %d & %s & %d/%d & %.1f & %.1f \\\\" % (FAM_LONG[fam], len(d["groups"]),
                   len(d["cols"]), Nstr, int(g.separated.sum()), len(g), g.R.median(), gn.R.median()))
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(RES, "table_families.tex"), "w").write("\n".join(tex) + "\n")
    fig(tabs, os.path.join(HERE, "fig_ratios.pdf"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
