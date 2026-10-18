"""Figure 5 (Appendix M.3.2): change in accuracy of 33 public models when all position indices of a request are
shifted. Reads ../shifted_positions_serving/results/table_pooled.csv, writes fig_shift.pdf, fig_shift.png and
fig_shift_values.json, and prints every plotted number.
Rows = models grouped by position encoding; colour = encoding (Okabe-Ito: APE #D55E00, ALiBi #009E73,
RoPE #0072B2); filled marker = paired bootstrap 95% CI excludes 0, hollow = CI includes 0; thin bar = CI.
(a) all positions shifted by 256 in fp32 (circle) and the same shift with a sliding window (triangle);
(b) control: cached prefix at its own positions, rest of the prompt 64 positions later (gap of 64);
(c) all positions shifted by 256 (circle) and 1024 (square) in bf16 (1024 only where the context allows;
Pythia 160M omitted, bf16 reference accuracy 0.012).
Usage: python make_fig_shift.py [output folder, default: the folder of this script]"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
POOLED = os.path.join(os.path.dirname(HERE), 'shifted_positions_serving', 'results', 'table_pooled.csv')
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
OUTPDF = os.path.join(OUT, 'fig_shift.pdf')

GROUPS = [
    ('APE', 'learned position\nembeddings (APE)',
     [('gpt2', 'GPT-2 124M'), ('gpt2-medium', 'GPT-2 355M'), ('gpt2-large', 'GPT-2 774M'), ('gpt2-xl', 'GPT-2 1.5B'),
      ('gpt-neo-125m', 'GPT-Neo 125M'), ('gpt-neo-1.3B', 'GPT-Neo 1.3B'), ('gpt-neo-2.7B', 'GPT-Neo 2.7B'),
      ('opt-125m', 'OPT 125M'), ('opt-350m', 'OPT 350M'), ('opt-1.3b', 'OPT 1.3B'), ('opt-2.7b', 'OPT 2.7B'),
      ('opt-6.7b', 'OPT 6.7B')]),
    ('ALiBi', 'ALiBi',
     [('bloom-560m', 'BLOOM 560M'), ('bloom-1b1', 'BLOOM 1.1B'), ('bloom-1b7', 'BLOOM 1.7B'), ('bloom-3b', 'BLOOM 3B'),
      ('bloom-7b1', 'BLOOM 7.1B')]),
    ('RoPE', 'rotary positions (RoPE)',
     [('pythia-160m', 'Pythia 160M'), ('pythia-410m', 'Pythia 410M'), ('pythia-1b', 'Pythia 1B'),
      ('pythia-1.4b', 'Pythia 1.4B'), ('pythia-2.8b', 'Pythia 2.8B'), ('pythia-6.9b', 'Pythia 6.9B'),
      ('gpt-j-6b', 'GPT-J 6B'), ('qwen2.5-0.5b', 'Qwen2.5 0.5B'), ('qwen2.5-1.5b', 'Qwen2.5 1.5B'),
      ('qwen2.5-3b', 'Qwen2.5 3B'), ('qwen2.5-7b', 'Qwen2.5 7B'), ('olmo-2-1b', 'OLMo-2 1B'), ('olmo-2-7b', 'OLMo-2 7B'),
      ('smollm2-135m', 'SmolLM2 135M'), ('smollm2-360m', 'SmolLM2 360M'), ('smollm2-1.7b', 'SmolLM2 1.7B')])]
COL = {'APE': '#D55E00', 'ALiBi': '#009E73', 'RoPE': '#0072B2'}
INK, MUTED, GRID, ROWLINE = '#222222', '#666666', '#DDDDDD', '#EFEFEF'
DY = 0.2          # vertical separation of the two markers that share a row
XLIM = (-0.62, 0.07)

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

# P['<model>/<scenario>']['<row>']: delta, ci, n and acc_ref of one row of table_pooled.csv
P = {}
for c in csv.DictReader(open(POOLED, newline='')):
    P.setdefault(f"{c['model']}/{c['scenario']}", {})[c['row']] = dict(
        delta=float(c['delta']), ci=[float(c['ci_lo']), float(c['ci_hi'])], n=int(c['n']), acc_ref=float(c['acc_ref']))
MODELS = {k.split('/')[0] for k in P}
# panel (c) shows a model only if its bf16 reference accuracy is at least half of its fp32 reference accuracy
# (HALF_ADMIT in shifted_positions_serving/code/analyze.py); the row 'rep' covers every item of the scenario
bf16_ok = {t: P[f'{t}/offset_bf16']['rep']['acc_ref'] >= 0.5 * P[f'{t}/offset']['rep']['acc_ref'] for t in MODELS}
assert sum(len(g[2]) for g in GROUPS) == 33 and set(t for g in GROUPS for t, _ in g[2]) == MODELS


def get(tag, scen, key):
    c = P.get(f'{tag}/{scen}', {}).get(key)
    if not c:
        return None
    lo, hi = c['ci']
    return dict(delta=c['delta'], lo=lo, hi=hi, sig=(hi < 0 or lo > 0), n=c['n'], acc_ref=c['acc_ref'])


# row positions (top to bottom), with a gap between encoding groups
rows, y = [], 0.0
for gi, (enc, glabel, ms) in enumerate(GROUPS):
    if gi:
        y += 0.7
    for tag, name in ms:
        rows.append(dict(tag=tag, name=name, enc=enc, y=y))
        y += 1.0
YMAX = y - 1.0

fig, axs = plt.subplots(1, 3, figsize=(5.5, 4.3), sharey=True)
fig.subplots_adjust(left=0.205, right=0.985, top=0.927, bottom=0.072, wspace=0.10)

printed = []


def point(ax, x, yy, enc, marker, label, tag, ms=3.9):
    if x is None:
        return
    c = COL[enc]
    ax.plot([x['lo'], x['hi']], [yy, yy], color=c, lw=0.7, zorder=2, solid_capstyle='butt')
    ax.plot([x['delta']], [yy], marker=marker, ls='', markersize=ms, zorder=3, markeredgewidth=0.8,
            markeredgecolor=c, markerfacecolor=c if x['sig'] else 'white')
    printed.append((label, tag, round(x['delta'], 4), round(x['lo'], 4), round(x['hi'], 4), x['sig']))


for r in rows:
    t, e, yy = r['tag'], r['enc'], r['y']
    # (a) fp32: all positions shifted by 256; same shift with a sliding window
    point(axs[0], get(t, 'offset', 'k256'), yy - DY, e, 'o', 'a:shift256_fp32', t)
    point(axs[0], get(t, 'window', 'k256'), yy + DY, e, '^', 'a:window256_fp32', t)
    # (b) control: cache at its own positions, gap of 64 before the rest of the prompt
    point(axs[1], get(t, 'cache', 'gap64'), yy, e, 'D', 'b:gap64_fp32', t, ms=3.5)
    # (c) bf16 shift by 256 and 1024 (Pythia 160M omitted: bf16 reference accuracy 0.012)
    if bf16_ok.get(t, True):
        point(axs[2], get(t, 'offset_bf16', 'k256'), yy - DY, e, 'o', 'c:shift256_bf16', t)
        point(axs[2], get(t, 'offset_bf16', 'k1024'), yy + DY, e, 's', 'c:shift1024_bf16', t, ms=3.5)
    else:
        axs[2].text(XLIM[0] + 0.01, yy, 'not shown', ha='left', va='center', color=MUTED, fontsize=7,
                    style='italic')

titles = ['(a) all position indices\nshifted by 256, fp32', '(b) control: gap of 64\nafter a cached prefix, fp32',
          '(c) all position indices\nshifted by 256, 1024, bf16']
for ax, title in zip(axs, titles):
    ax.set_xlim(*XLIM)
    ax.set_ylim(YMAX + 0.6, -0.6)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    for r in rows:
        ax.axhline(r['y'], color=ROWLINE, lw=0.5, zorder=0)
    gy = [rows[i]['y'] for i in range(1, len(rows)) if rows[i]['enc'] != rows[i - 1]['enc']]
    for g in gy:
        ax.axhline(g - 0.85, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0])
    ax.set_xticklabels(['−0.6', '−0.4', '−0.2', '0'])
    ax.set_xlabel('change in accuracy', labelpad=1.5)
    ax.set_title(title, loc='left', color=INK, fontsize=7, pad=3)
    ax.tick_params(length=2.0, pad=1.5)
    ax.tick_params(axis='y', length=0)
axs[0].set_yticks([r['y'] for r in rows])
axs[0].set_yticklabels([r['name'] for r in rows])
for a in axs[1:]:
    a.tick_params(axis='y', labelleft=False)
    a.spines['left'].set_visible(False)

# encoding group labels, rotated, left of the model names
tr = mtransforms.blended_transform_factory(fig.transFigure, axs[0].transData)
for enc, glabel, ms in GROUPS:
    ys = [r['y'] for r in rows if r['enc'] == enc]
    fig.text(0.012, (min(ys) + max(ys)) / 2, glabel, transform=tr, rotation=90, ha='left', va='center',
             color=COL[enc], fontsize=7, fontweight='bold', linespacing=1.0)

# legends inside the empty lower-left regions (ALiBi/RoPE rows sit at 0 there)
h_a = [Line2D([], [], marker='o', ls='', color=INK, markerfacecolor=INK, markersize=3.9,
              label='all position indices\nshifted by 256'),
       Line2D([], [], marker='^', ls='', color=INK, markerfacecolor=INK, markersize=3.9,
              label='same, with a\nsliding window'),
       Line2D([], [], marker='o', ls='', color=INK, markerfacecolor=INK, markersize=3.9,
              label='filled: 95% CI\nexcludes 0'),
       Line2D([], [], marker='o', ls='', color=INK, markerfacecolor='white', markersize=3.9,
              label='hollow: 95% CI\nincludes 0')]
axs[0].legend(handles=h_a, loc='lower left', frameon=False, handlelength=0.8, handletextpad=0.4,
              labelspacing=0.55, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.0))
h_c = [Line2D([], [], marker='o', ls='', color=INK, markerfacecolor=INK, markersize=3.9, label='shifted by 256'),
       Line2D([], [], marker='s', ls='', color=INK, markerfacecolor=INK, markersize=3.5, label='shifted by 1024')]
axs[2].legend(handles=h_c, loc='lower left', frameon=False, handlelength=0.8, handletextpad=0.4,
              labelspacing=0.4, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.0))

# ---------------- layout self-check ----------------
fig.canvas.draw()
r_ = fig.canvas.get_renderer()
texts = []
for a in fig.axes:
    ab = a.get_window_extent(r_)
    for t in a.get_xticklabels():
        e = t.get_window_extent(r_)
        if ab.x0 - 2 <= (e.x0 + e.x1) / 2 <= ab.x1 + 2 and t.get_visible():
            texts.append(t)
    for t in a.get_yticklabels():
        e = t.get_window_extent(r_)
        if ab.y0 - 2 <= (e.y0 + e.y1) / 2 <= ab.y1 + 2 and t.get_visible():
            texts.append(t)
    texts += [a.title, a._left_title, a.xaxis.label, a.yaxis.label] + list(a.texts)
    lg = a.get_legend()
    if lg:
        texts += list(lg.get_texts())
texts += list(fig.texts)
texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
boxes = [(t.get_text(), t.get_window_extent(r_)) for t in texts]
W, H = fig.bbox.width, fig.bbox.height
issues = []
for i in range(len(boxes)):
    a = boxes[i][1]
    if a.x0 < -1 or a.y0 < -1 or a.x1 > W + 1 or a.y1 > H + 1:
        issues.append(('outside', boxes[i][0]))
    for j in range(i + 1, len(boxes)):
        b = boxes[j][1]
        if a.overlaps(b) and min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
            issues.append(('overlap', boxes[i][0], boxes[j][0]))
# markers under legend/text
for a in fig.axes:
    pts = []
    for ln in a.lines:
        if ln.get_marker() not in (None, 'None', '') and len(ln.get_xdata()) == 1:
            pts.append(a.transData.transform((ln.get_xdata()[0], ln.get_ydata()[0])))
    boxes_t = [t.get_window_extent(r_) for t in a.texts]
    lg = a.get_legend()
    if lg:
        boxes_t.append(lg.get_window_extent(r_))
    for e in boxes_t:
        for (px, py) in pts:
            if e.x0 - 2 <= px <= e.x1 + 2 and e.y0 - 2 <= py <= e.y1 + 2:
                issues.append(('marker under text/legend', a.get_title(loc='left')[:12], (round(px), round(py))))
# vertical spacing of model names
yl = sorted(axs[0].transData.transform((0, r['y']))[1] for r in rows)
print('row spacing (pt):', round(min(np.diff(yl)) * 72 / fig.dpi, 2))
print('layout issues:', issues if issues else 'none')
for a in fig.axes:
    print('axes', a.get_title(loc='left').split('\n')[0], 'bbox (in)',
          tuple(round(v / fig.dpi, 2) for v in a.get_window_extent(r_).bounds))
print('figure size (in):', fig.get_size_inches())

fig.savefig(OUTPDF)
fig.savefig(OUTPDF.replace('.pdf', '.png'), dpi=220)
print('wrote', OUTPDF)

# ---------------- plotted numbers ----------------
name = {r['tag']: r['name'] for r in rows}
print('\nlabel | model | delta | ci_lo | ci_hi | ci_excludes_0')
for p in printed:
    print('%-18s %-14s %+.4f [%+.4f, %+.4f] %s' % (p[0], name[p[1]], p[2], p[3], p[4], p[5]))
json.dump([dict(panel=p[0], model=name[p[1]], tag=p[1], delta=p[2], ci=[p[3], p[4]], ci_excludes_0=p[5])
           for p in printed], open(os.path.join(OUT, 'fig_shift_values.json'), 'w'), indent=1)

# label box height vs row spacing
hs = [t.get_window_extent(r_).height * 72 / fig.dpi for t in axs[0].get_yticklabels()]
print('model-name box height (pt): %.2f' % max(hs))
for a in (axs[0], axs[2]):
    e = a.get_legend().get_window_extent(r_)
    x0, _ = a.transData.inverted().transform((e.x0, e.y0)); x1, _ = a.transData.inverted().transform((e.x1, e.y1))
    print('legend x-extent (data units):', round(x0, 3), round(x1, 3))
