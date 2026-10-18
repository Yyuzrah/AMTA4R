"""Supplementary figure of the attention-score results of Experiment 4 (App. M.3.1). Reads results/summary.json, writes fig_lemma2.pdf
(+ results/fig_lemma2_preview.png). Color = positional encoding (Okabe-Ito vermillion / blue), marker = shift k,
filled = fp32, hollow = fp64.
Each point is the value pooled over the four input families; the thin vertical bar spans the four per-family
values."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/lemma2_pretrained
RES = os.environ.get('LEMMA2_RES', f'{HERE}/results')
OUTPDF = os.environ.get('LEMMA2_FIG', f'{HERE}/fig_lemma2.pdf')
FAMILIES = ('countdown_dfs', 'countdown_bfs', 'dyck', 'code')
MODELS = [('gpt2', 'APE'), ('gpt2-medium', 'APE'), ('gpt-neo-125m', 'APE'), ('gpt-neo-1.3B', 'APE'),
          ('pythia-160m', 'RoPE'), ('pythia-410m', 'RoPE'), ('pythia-1.4b', 'RoPE'), ('smollm2-360m', 'RoPE')]
NAME = {'gpt2': 'GPT-2 124M', 'gpt2-medium': 'GPT-2 355M', 'gpt-neo-125m': 'GPT-Neo 125M', 'gpt-neo-1.3B': 'GPT-Neo 1.3B',
        'pythia-160m': 'Pythia 160M', 'pythia-410m': 'Pythia 410M', 'pythia-1.4b': 'Pythia 1.4B', 'smollm2-360m': 'SmolLM2 360M'}
COL = {'APE': '#D55E00', 'RoPE': '#0072B2'}
KS = (16, 64, 256)
MK = {16: 'o', 64: 's', 256: '^'}
DX = {16: -0.22, 64: 0.0, 256: 0.22}
INK, MUTED, GRID = '#222222', '#666666', '#DDDDDD'

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

S = json.load(open(f'{RES}/summary.json'))
C = S['cells']
models = [(t, e) for t, e in MODELS if f'{t}|fp32|pooled|16' in C]
xs, x = [], 0.0
for i, (t, e) in enumerate(models):
    if i > 0 and e != models[i - 1][1]:
        x += 0.7
    xs.append(x); x += 1.0


def get(t, prec, g, k, m):
    c = C.get(f'{t}|{prec}|{g}|{k}')
    return None if c is None else c.get(m)


def panel(ax, metric, prec='fp32', hollow=False, scale=1.0, label=None):
    for xi, (t, e) in zip(xs, models):
        for k in KS:
            v = get(t, prec, 'pooled', k, metric)
            if v is None:
                continue
            fam = [get(t, prec, f, k, metric) for f in FAMILIES]
            fam = [f * scale for f in fam if f is not None]
            xx = xi + DX[k]
            if fam:
                ax.plot([xx, xx], [min(fam), max(fam)], color=COL[e], lw=0.9, alpha=0.8, zorder=2)
            ax.scatter([xx], [v * scale], marker=MK[k], s=26, zorder=3,
                       facecolor='white' if hollow else COL[e], edgecolor=COL[e], linewidth=1.1)


def deco(ax, title, ylabel, log=True, bottom=False):
    if log:
        ax.set_yscale('log')
    ax.set_title(title, loc='left', color=INK)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    if bottom:
        ax.set_xticklabels([NAME[t] for t, _ in models], rotation=45, ha='right', rotation_mode='anchor')
    else:
        ax.set_xticklabels([])
    ax.grid(axis='y', color=GRID, lw=0.6, which='major')
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5))
ax = axs[0, 0]
panel(ax, 'm1_first_max')
ax.axhline(1e-5, color=MUTED, lw=0.8, ls='--', zorder=1)
ax.text(xs[-1] + 0.5, 1e-5 * 1.6, 'decision bound $10^{-5}$', ha='right', va='bottom', color=MUTED, fontsize=6.5)
ax.axhline(2.0 ** -24, color=MUTED, lw=0.8, ls=':', zorder=1)
ax.text(xs[-1] + 0.5, 2.0 ** -24 * 1.6, 'fp32 unit roundoff', ha='right', va='bottom', color=MUTED, fontsize=6.5)
deco(ax, '(a) $m_1$: layer-1 score change (fp32)', r'max $|\Delta A|$ / max $|A|$')

ax = axs[0, 1]
panel(ax, 'm1_all_max')
panel(ax, 'm1_all_max', prec='fp64diag', hollow=True)
for xi, (t, e) in zip(xs, models):
    f = get(t, 'fp64diag', 'pooled', 16, 'F_base_all_max')
    if f is not None:
        ax.plot([xi - 0.36, xi + 0.36], [f, f], color=INK, lw=1.0, zorder=4)
ax.axhline(1e-10, color=MUTED, lw=0.8, ls='--', zorder=1)
ax.text(xs[-1] + 0.5, 1e-10 * 1.6, 'fp64 bound $10^{-10}$', ha='right', va='bottom', color=MUTED, fontsize=6.5)
deco(ax, '(b) $m_1$: all layers (filled fp32, hollow fp64)', r'max over layers of (a)')

ax = axs[1, 0]
panel(ax, 'm2', scale=100.0)
deco(ax, '(c) $m_2$: top-1 next-token flips (fp32)', 'positions flipped (%)', log=False, bottom=True)
ax.set_ylim(bottom=-1)
rope_cells = [C[k_]['m2'] for k_ in C if k_.split('|')[1] == 'fp32' and k_.split('|')[3] in ('16', '64', '256')
              and dict(MODELS).get(k_.split('|')[0]) == 'RoPE']
rx = [xi for xi, (t, e) in zip(xs, models) if e == 'RoPE']
if rope_cells and rx:
    ax.text(np.mean(rx), ax.get_ylim()[1] * 0.12, f'RoPE: every cell\n$\\leq${100 * max(rope_cells):.2f}%',
            ha='center', va='bottom', color=MUTED, fontsize=6.5)

ax = axs[1, 1]
panel(ax, 'm3_abs')
deco(ax, r'(d) $m_3$: mean $|\Delta \log p|$ per token (fp32)', 'nats / token', bottom=True)

for a in axs.flat:
    ape_x = [xi for xi, (t, e) in zip(xs, models) if e == 'APE']
    rope_x = [xi for xi, (t, e) in zip(xs, models) if e == 'RoPE']
    if ape_x and rope_x:
        a.axvline((max(ape_x) + min(rope_x)) / 2, color=GRID, lw=0.8, zorder=0)

handles = [Patch(facecolor=COL['APE'], label='learned APE'), Patch(facecolor=COL['RoPE'], label='RoPE')]
handles += [Line2D([], [], marker=MK[k], ls='', color=INK, markerfacecolor=INK, markersize=5, label=f'k = {k}') for k in KS]
handles += [Line2D([], [], marker='o', ls='', color=INK, markerfacecolor='white', markersize=5, label='fp64'),
            Line2D([], [], color=INK, lw=1.0, label='fp32 rounding floor (no shift)')]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=0.8, w_pad=1.0)
# layout self-check: overlapping text boxes and text outside the figure
fig.canvas.draw()
r = fig.canvas.get_renderer()
texts = []
for a in fig.axes:
    ab = a.get_window_extent(r)
    for t in a.get_xticklabels():
        e = t.get_window_extent(r)
        if ab.x0 - 2 <= (e.x0 + e.x1) / 2 <= ab.x1 + 2 and t.get_visible():
            texts.append(t)
    for t in a.get_yticklabels():
        e = t.get_window_extent(r)
        if ab.y0 - 2 <= (e.y0 + e.y1) / 2 <= ab.y1 + 2 and t.get_visible():
            texts.append(t)
    texts += [a.title, a._left_title, a.xaxis.label, a.yaxis.label] + list(a.texts)
texts += [t for lg in fig.legends for t in lg.get_texts()]
texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
boxes = [(t.get_text(), t.get_window_extent(r)) for t in texts]
W, H = fig.bbox.width, fig.bbox.height
issues = []
for i in range(len(boxes)):
    a = boxes[i][1]
    if a.x0 < -1 or a.y0 < -1 or a.x1 > W + 1 or a.y1 > H + 1:
        issues.append(('outside', boxes[i][0]))
    for j in range(i + 1, len(boxes)):
        b = boxes[j][1]
        if texts[i].get_rotation() and texts[j].get_rotation():
            continue
        if a.overlaps(b) and min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
            issues.append(('overlap', boxes[i][0], boxes[j][0]))
for a in fig.axes:
    pts = []
    for coll in a.collections:
        off = coll.get_offsets()
        if len(off):
            pts += list(a.transData.transform(off))
    for t in list(a.texts) + [a._left_title]:
        e = t.get_window_extent(r).expanded(1.0, 1.0)
        for (px, py) in pts:
            if e.x0 - 4 <= px <= e.x1 + 4 and e.y0 - 4 <= py <= e.y1 + 4:
                issues.append(('marker under text', t.get_text()))
print('layout issues:', issues if issues else 'none')
for a in fig.axes:
    print('axes', a.get_title(loc='left')[:20], 'ylim', a.get_ylim(), 'bbox', a.get_window_extent(r).bounds)
fig.savefig(OUTPDF)
fig.savefig(os.path.join(RES, 'fig_lemma2_preview.png'), dpi=200)
print('wrote', OUTPDF)
