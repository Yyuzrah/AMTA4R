# Draws Figures 1-3 of the paper from results/final_results_54.json (TrueType fonts embedded).
#   python figures/make_figures.py --results results/final_results_54.json --outdir figures/out
import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "fonts", "liberation-fonts-ttf-2.1.5")
for f in os.listdir(FONT_DIR):
    if f.endswith(".ttf"):
        fm.fontManager.addfont(os.path.join(FONT_DIR, f))
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, never Type 3
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"],   # metric-compatible with Arial
    "mathtext.fontset": "stix",
})
SERIF = "Liberation Serif"                         # metric-compatible with Times New Roman
RED, BLUE = (0.839, 0.153, 0.157), (0.122, 0.467, 0.706)   # tab:red / tab:blue

ARCH = {  # name: (L, d)
    "small_deep": (12, 192), "small_balanced": (6, 256), "small_wide": (3, 384),
    "medium_deep": (24, 512), "medium_balanced": (12, 768), "medium_wide": (6, 1024),
    "large_deep": (48, 768), "large_balanced": (24, 1024), "large_wide": (12, 1536)}
CORE = {k: 12 * L * d * d / 1e6 for k, (L, d) in ARCH.items()}
SCALES, SHAPES, DATA, PES = ["small", "medium", "large"], ["deep", "balanced", "wide"], ["bfs", "dfs", "mixed"], ["ape", "rope"]


def load(results, extra):
    R = json.load(open(results))
    if extra and os.path.exists(extra):
        R.update(json.load(open(extra)))
    return R


def fig_h2(R, out):
    sns.set_style("whitegrid", {"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"]})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    titles = [r"A) Small Scale (core $\approx$5M; total $\approx$30M)", r"B) Medium Scale (core $\approx$80M; total $\approx$150M)", r"C) Large Scale (core $\approx$300M; total $\approx$450M)"]
    order = ["bfs", "mixed", "dfs"]
    for ax, s, t in zip(axes, SCALES, titles):
        rows = [(m.upper(), {"ape": "APE", "rope": "RoPE"}[pe], R[f"{s}_{sh}_{m}_{pe}"][m]) for m in order for pe in PES for sh in SHAPES
                if f"{s}_{sh}_{m}_{pe}" in R]
        import pandas as pd
        df = pd.DataFrame(rows, columns=["measure", "pe", "acc"])
        sns.barplot(data=df, x="measure", y="acc", hue="pe", order=[m.upper() for m in order], hue_order=["APE", "RoPE"],
                    palette={"APE": RED, "RoPE": BLUE}, errorbar="sd", capsize=0.05, err_kws={"linewidth": 2.5, "color": (0.26, 0.26, 0.26)},
                    edgecolor="black", linewidth=2, ax=ax)
        ax.set_ylim(0.40, 0.56); ax.set_title(t, fontsize=14, fontweight="bold")
        ax.set_xlabel(r"Evaluation set (in distribution, $\nu_T = \nu_S$)", fontsize=13); ax.set_ylabel("Empirical In-Distribution Accuracy" if s == "small" else "", fontsize=13)
        if s == "large":
            ax.legend(title="Positional Encoding", loc="upper left", prop={"family": SERIF, "size": 11}, title_fontproperties={"family": SERIF, "size": 11})
        else:
            ax.get_legend().remove()
    fig.tight_layout(); fig.savefig(os.path.join(out, "app_baseline_variance_1x3.pdf")); plt.close(fig)


def fig1(R, out, beta, A=0.538, alpha=0.15):
    sns.set_style("whitegrid", {"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=(14.86, 10.85))
    # ---- (A) accuracy on mu_DFS vs physical depth ----
    ax = axes[0, 0]
    mk = {"bfs": "o", "dfs": "X", "mixed": "s"}
    for s in SCALES:
        for sh in SHAPES:
            for dname in DATA:
                for pe in PES:
                    k = f"{s}_{sh}_{dname}_{pe}"
                    if k in R:
                        ax.scatter(ARCH[f"{s}_{sh}"][0], R[k]["dfs"], marker=mk[dname], s=80, color=RED if pe == "ape" else BLUE,
                                   alpha=0.75, edgecolors="white", linewidths=0.75, zorder=3)
    Ls = np.linspace(2, 50, 100)
    ax.plot(Ls, A - beta * np.exp(-alpha * Ls), "k--", lw=2.5, zorder=2)
    ax.set_xlabel(r"Physical Layer Depth ($L$)", fontsize=13); ax.set_ylabel("Empirical Accuracy", fontsize=13)
    ax.set_title(r"A) The Supremum Capacity Limit on $\mu_{DFS}$", fontsize=14)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", label="Positional Encoding"),
         Line2D([], [], marker="o", ls="", color=RED, alpha=0.75, label="APE", ms=9), Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.75, label="RoPE", ms=9),
         Line2D([], [], ls="", label="Training Data Distribution"),
         Line2D([], [], marker="o", ls="", color=(0.2, 0.2, 0.2), label="lateral state expansions (BFS)"),
         Line2D([], [], marker="X", ls="", color=(0.2, 0.2, 0.2), label="frequent backtracking (DFS)"),
         Line2D([], [], marker="s", ls="", color=(0.2, 0.2, 0.2), label="compositional mixtures (MIXED)"),
         Line2D([], [], color="k", ls="--", lw=2.5, label=r"Theoretical Ceiling (calibrated, App. G.3)")]
    # place the legend where it covers no data point (the L=48 point sits in the lower right):
    # try one or two columns and, if needed, extend the y-range downward to make room below the points
    pts = np.array([[c.get_offsets()[0][0], c.get_offsets()[0][1]] for c in ax.collections if len(c.get_offsets())])
    y0, y1 = ax.get_ylim(); placed = None
    for bottom in [y0, 0.38, 0.36, 0.34, 0.32, 0.30]:
        ax.set_ylim(bottom, y1)
        for ncol in (1, 2):
            for loc in ["lower right", "lower center", "lower left", "center right", "upper left"]:
                leg = ax.legend(handles=h, loc=loc, ncol=ncol, fontsize=9.5, frameon=True, edgecolor="black", framealpha=1.0)
                fig.canvas.draw()
                bb = leg.get_window_extent(); disp = ax.transData.transform(pts)
                n_cov = sum(bb.x0 - 6 <= x <= bb.x1 + 6 and bb.y0 - 6 <= y <= bb.y1 + 6 for x, y in disp)
                if n_cov == 0:
                    placed = (bottom, ncol, loc); break
            if placed: break
        if placed: break
    print("Fig.1A legend:", placed, "| y-range", ax.get_ylim(), "| points covered:", n_cov, "of", len(pts))
    # ---- (B) DFS-trained on mu_DFS vs core params ----
    ax = axes[0, 1]
    sty = {"balanced": ("-", "o"), "wide": ("--", "s"), "deep": (":", "D")}
    for pe, col in [("rope", BLUE), ("ape", RED)]:
        for sh in ["balanced", "wide", "deep"]:
            xs = [CORE[f"{s}_{sh}"] for s in SCALES]; ys = [R[f"{s}_{sh}_dfs_{pe}"]["dfs"] for s in SCALES]
            ax.plot(xs, ys, ls=sty[sh][0], marker=sty[sh][1], color=col, lw=3, ms=10, markeredgecolor="white")
    ax.set_xscale("log"); ax.set_xticks([5, 80, 300]); ax.set_xticklabels(["5", "80", "300"]); ax.minorticks_off()
    ax.set_xlabel("Core Parameters (Millions, log scale)", fontsize=13); ax.set_ylabel(r"Accuracy on $\mu_{DFS}$", fontsize=13)
    ax.set_title("B) Scaling Limits: Representation Collapse vs. Stability", fontsize=14)
    h = [Line2D([], [], ls="", label="Positional Encoding"), Line2D([], [], color=BLUE, lw=3, label="RoPE"), Line2D([], [], color=RED, lw=3, label="APE"),
         Line2D([], [], ls="", label="Shape")] + [Line2D([], [], color=(0.2, 0.2, 0.2), ls=sty[k][0], marker=sty[k][1], lw=2, label=k.capitalize()) for k in ["balanced", "wide", "deep"]]
    ax.legend(handles=h, loc="lower left", fontsize=10.5, frameon=True, edgecolor="black")
    # ---- (C) wide models, mixed-trained on mu_MIXED ----
    ax = axes[1, 0]
    xs = [CORE[f"{s}_wide"] for s in SCALES]
    for pe, col, lab in [("ape", RED, "APE"), ("rope", BLUE, "RoPE")]:
        ax.plot(xs, [R[f"{s}_wide_mixed_{pe}"]["mixed"] for s in SCALES], marker="X", ms=12, lw=3, color=col, label=lab, markeredgecolor="white")
    floor = R["large_wide_mixed_ape"]["mixed"]
    ax.axhline(floor, color="red", ls=":", lw=3)
    ax.text(xs[-1] * 1.12, floor - 0.006, r"$\Omega(1)$ Representation Collapse", color="red", ha="right", va="top", fontsize=12)  # below the floor line, clear of the APE curve
    lo, hi = ax.get_ylim(); ax.set_ylim(min(lo, floor - 0.02), hi)  # room for the annotation
    ax.set_xscale("log"); ax.set_xticks([5, 80, 300]); ax.set_xticklabels(["5", "80", "300"]); ax.minorticks_off()
    ax.set_xlabel("Core Parameters (Millions, log scale)", fontsize=13); ax.set_ylabel(r"Accuracy on $\mu_{MIXED}$", fontsize=13)
    ax.set_title(r"C) Sequence Equivariance on $\mu_{MIXED}$ (Wide Models)", fontsize=14)
    ax.legend(loc="center right", fontsize=11, frameon=True, edgecolor="black")
    # ---- (D) all BFS-trained models across target measures ----
    ax = axes[1, 1]
    order = ["bfs", "mixed", "dfs"]
    for pe, col, lab in [("ape", RED, "APE"), ("rope", BLUE, "RoPE")]:
        first = True
        for s in SCALES:
            for sh in SHAPES:
                k = f"{s}_{sh}_bfs_{pe}"
                if k in R:
                    ax.plot(range(3), [R[k][m] for m in order], marker="o", ms=5, lw=2.5, alpha=0.5, color=col, label=lab if first else None); first = False
    ax.set_xticks(range(3)); ax.set_xticklabels(["BFS", "MIXED", "DFS"])
    ax.set_xlabel(r"Evaluation set (ordered by $\widehat{W}_1$ from $\nu_{BFS}$)", fontsize=13); ax.set_ylabel("Empirical Accuracy", fontsize=13)
    ax.set_title("D) Accuracy by Evaluation Set (All BFS-Trained)", fontsize=14)
    ax.legend(loc="upper right", fontsize=11, frameon=True, edgecolor="black")
    fig.tight_layout(); fig.savefig(os.path.join(out, "dense_analytical_verification_2x2.pdf")); plt.close(fig)
    # report envelope violations
    viol = [(k, ARCH["_".join(k.split("_")[:2])][0], v["dfs"], A - beta * math.exp(-alpha * ARCH["_".join(k.split("_")[:2])][0]))
            for k, v in R.items() if v["dfs"] > A - beta * math.exp(-alpha * ARCH["_".join(k.split("_")[:2])][0])]
    return viol


def fig_h1(R, out, ape_ls="--", rope_ls="-"):
    sns.set_style("whitegrid", {"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"]})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax, s, t, labs in [(axes[0], "large", r"A) Coordinate Disruption vs. Depth (Large Scale)", ["Wide\n(L=12)", "Balanced\n(L=24)", "Deep\n(L=48)"]),
                           (axes[1], "medium", "B) Consistent Equivariance Regularization (Medium Scale)", ["Wide\n(L=6)", "Balanced\n(L=12)", "Deep\n(L=24)"])]:
        ya = [R[f"{s}_{sh}_dfs_ape"]["dfs"] for sh in ["wide", "balanced", "deep"]]
        yr = [R[f"{s}_{sh}_dfs_rope"]["dfs"] for sh in ["wide", "balanced", "deep"]]
        ax.plot(range(3), ya, ls=ape_ls, marker="X", ms=13, lw=3, color=RED, label="APE")
        ax.plot(range(3), yr, ls=rope_ls, marker="o", ms=13, lw=3, color=BLUE, label="RoPE")
        if s == "large":
            lo, hi = ax.get_ylim(); pad = 0.08 * (hi - lo); ax.set_ylim(lo - pad, hi + pad)  # keep the value labels inside the frame
            for i in (0, 2):
                ax.annotate(f"{yr[i]:.4f}", (i, yr[i]), textcoords="offset points", xytext=(-10 if i == 0 else 0, 10), ha="center", color=BLUE, fontweight="bold", family=SERIF, fontsize=12)
                ax.annotate(f"{ya[i]:.4f}", (i, ya[i]), textcoords="offset points", xytext=(28 if i == 0 else 0, -20), ha="center", color=RED, fontweight="bold", family=SERIF, fontsize=12)
        ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=12)
        ax.set_xlabel("Architectural Aspect Ratio", fontsize=12); ax.set_ylabel(r"Accuracy on $\mu_{DFS}$", fontsize=12)
        ax.set_title(t, fontsize=14, fontweight="bold")
        ax.legend(title="Positional Encoding", loc="lower right" if s == "large" else "upper right", prop={"family": SERIF, "size": 12}, title_fontproperties={"family": SERIF, "size": 12})
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_supp_rope_mitigation.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--extra", default=None)
    ap.add_argument("--beta", type=float, default=0.035); ap.add_argument("--outdir", required=True)  # beta: calibration of App. G.3
    ap.add_argument("--h1_match_caption", action="store_true", default=True, help="draw APE dashed / RoPE solid (default)")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    R = load(a.results, a.extra)
    print("configs:", len(R))
    fig_h2(R, a.outdir)
    v = fig1(R, a.outdir, a.beta)
    print(f"envelope (A=0.538, beta={a.beta}, alpha=0.15) violations: {len(v)}", v)
    fig_h1(R, a.outdir, *(("--", "-") if a.h1_match_caption else ("-", "--")))
    for f in sorted(os.listdir(a.outdir)):
        print("wrote", os.path.join(a.outdir, f))
