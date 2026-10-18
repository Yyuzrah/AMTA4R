"""Figure 6 (Appendix M.4): splitting deep nested tasks into separate model calls.
(a) nested plans: d_ext (largest depth with single-call success >= 0.5 on the calibration plans; column
    d_ext_calibration of runs.csv) against the training depth, one dot per run (120 runs; RoPE left, APE right of
    each training depth, spread horizontally), hollow = the 6 runs of APE two-layer width-624 at training depth 2
    and 4 (single-call success below 0.9 at a depth up to the training depth);
(b) nested plans, 12-layer RoPE model: fraction of plans solved against nesting depth, mean over 3 seeds,
    one call (single_call_success of per_depth.csv, dashed) and split into calls (split_success, solid), one colour
    per training depth;
(c) nested tool calls: accuracy against depth, mean over the 8 Qwen, Phi-4 and Mistral models for one call (I)
    and one level per call (A), thin lines = single models with one level per call; OLMo-2 mean in light grey;
(d) the 4 models whose d_hat (threshold 0.8) is 2 or 3: one call, one level per call and calls of
    d_hat levels (H0.8), mean over the 4 models.
Reads ../nested_plans/results/runs.csv and per_depth.csv, and ../nested_tool_calls/results/metrics.json (checked
against test_by_depth.csv of the same folder). Writes fig_split.pdf, fig_split.png and fig_split_values.json, and
prints every plotted number.
Usage: python make_fig_split.py [output folder, default: the folder of this script]"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
OUTPDF = os.path.join(OUT, 'fig_split.pdf')
COL = {'ape': '#D55E00', 'rope': '#0072B2'}
DTCOL = {2: '#4A90C8', 4: '#2B6CB0', 6: '#1A4D8F', 8: '#0B2E5C'}     # blue ramp, contrast >= 3.4:1 on white
C_I, C_A, C_H, C_OLMO = '#222222', '#009E73', '#CC79A7', '#A0A0A0'
INK, MUTED, GRID = '#222222', '#666666', '#DDDDDD'

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

# ---------------- data: nested plans ----------------
# runs.csv and per_depth.csv, read into the keys of analysis/summary.json written by nested_plans/code/analyze.py
RUNS, BELOW_09 = {}, set()
for c in csv.DictReader(open(f'{E}/nested_plans/results/runs.csv', newline='')):
    RUNS[c['run']] = {'pe': c['position_encoding'].lower(), 'model': c['model'], 'dtrain': int(c['d_train']),
                      'd_ext': {'cal_0.5': [int(c['d_ext_calibration'])]}, 'A_single': [],
                      'policy_0.5': {'A_hand': [], 'D_call': int(c['call_depth'])}}
    if c['single_call_ge_0.9_up_to_d_train'] == 'False':
        BELOW_09.add(c['run'])
for c in csv.DictReader(open(f'{E}/nested_plans/results/per_depth.csv', newline='')):
    r = RUNS[c['run']]
    assert int(c['depth']) == len(r['A_single']) + 1
    r['A_single'].append(float(c['single_call_success']))
    r['policy_0.5']['A_hand'].append(float(c['split_success']))
assert len(RUNS) == 120
NOFIT = {n for n, r in RUNS.items() if r['pe'] == 'ape' and r['model'] == 'W624' and r['dtrain'] in (2, 4)}
assert NOFIT == BELOW_09 and len(NOFIT) == 6

# ---------------- data: nested tool calls ----------------
T = json.load(open(f'{E}/nested_tool_calls/results/metrics.json'))['results']
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
MAIN = ['qwen2.5-7b', 'qwen2.5-14b', 'qwen2.5-32b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'phi4-14b', 'mistral-small-24b']
OLMO = ['olmo2-7b', 'olmo2-13b', 'olmo2-32b']
DHAT = {m: T[m]['T']['cal']['tau0.8']['dhat'] for m in T}
FOUR = [m for m in MAIN if DHAT[m] >= 2]
assert sorted(FOUR) == sorted(['qwen2.5-32b', 'qwen3-14b', 'phi4-14b', 'mistral-small-24b'])
assert all(T[m]['T']['budgets']['0.8']['D'] == DHAT[m] for m in FOUR)


def curve(m, cond):
    return np.array([T[m]['T']['curves'][cond][str(d)] for d in DEPTHS])


# cross-check against the column accuracy of test_by_depth.csv
RES = {}
for c in csv.DictReader(open(f'{E}/nested_tool_calls/results/test_by_depth.csv', newline='')):
    RES.setdefault(c['model'], {}).setdefault(c['condition'] + c['threshold'], {})[int(c['depth'])] = float(c['accuracy'])
dev = max(np.max(np.abs(curve(m, c) - np.array([RES[m][c][d] for d in DEPTHS]))) for m in T for c in ('I', 'A', 'H0.8'))
print('max |metrics.json - test_by_depth.csv| over all per-model values: %.4f' % dev)
assert dev == 0

fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.3), gridspec_kw=dict(height_ratios=[1.0, 1.17]))
fig.subplots_adjust(left=0.085, right=0.99, top=0.958, bottom=0.083, wspace=0.22, hspace=0.36)
out = {}

# ---------------- (a) deepest depth one call solves, per run ----------------
ax = axs[0, 0]
ax.plot([1, 9], [1, 9], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
pa = []
for dt in (2, 4, 6, 8):
    for pe, side in (('rope', -1), ('ape', +1)):
        rs = sorted([(n, r) for n, r in RUNS.items() if r['dtrain'] == dt and r['pe'] == pe],
                    key=lambda nr: (nr[1]['d_ext']['cal_0.5'][0], nr[0] in NOFIT, nr[0]))
        byv = {}
        for n, r in rs:
            byv.setdefault(r['d_ext']['cal_0.5'][0], []).append(n)
        for v, names in byv.items():
            k = len(names)
            span = min(0.55, 0.05 * (k - 1))
            xs = dt + side * 0.42 + (np.linspace(-span / 2, span / 2, k) if k > 1 else np.zeros(1))
            for x, n in zip(xs, names):
                hollow = n in NOFIT
                ax.scatter([x], [v], s=11, marker='o', facecolor='white' if hollow else COL[pe],
                           edgecolor=COL[pe] if hollow else 'white', linewidth=0.9 if hollow else 0.35,
                           zorder=4 if hollow else 3)
                pa.append((n, dt, pe, v, hollow))
ax.set_xlim(0.9, 9.1)
ax.set_ylim(0, 14)
ax.set_xticks([2, 4, 6, 8])
ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
ax.set_xlabel('training depth (deepest nesting in training)', labelpad=1.5)
ax.set_ylabel('$d_{\\mathrm{ext}}$ (single-call\nsuccess at least 0.5)', labelpad=2)
ax.set_title('(a) nested plans: $d_{\\mathrm{ext}}$ of each run', loc='left', pad=3)
ax.grid(axis='y', color=GRID, lw=0.5)
ax.set_axisbelow(True)
ha = [Line2D([], [], marker='o', ls='', markerfacecolor=COL['rope'], markeredgecolor='white', markersize=4.2,
             label='RoPE run'),
      Line2D([], [], marker='o', ls='', markerfacecolor=COL['ape'], markeredgecolor='white', markersize=4.2,
             label='APE run'),
      Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=COL['ape'], markersize=4.0,
             markeredgewidth=0.9, label='APE run below 0.9\nat a depth $\\leq d_{\\mathrm{train}}$')]
ax.legend(handles=ha, loc='upper left', frameon=False, handlelength=1.0, handletextpad=0.4, labelspacing=0.35,
          borderaxespad=0.1)
ax.text(9.0, 0.5, 'runs spread sideways;\ndashed line: equal to\nthe training depth', ha='right', va='bottom', color=MUTED, fontsize=7)
out['a'] = pa

# ---------------- (b) 12-layer RoPE model, mean over 3 seeds ----------------
ax = axs[0, 1]
pb = {}
for dt in (8, 6, 4, 2):
    rs = [RUNS[f'rope_dt{dt}_L12_s{s}'] for s in range(3)]
    single = np.mean([r['A_single'] for r in rs], 0)
    split = np.mean([r['policy_0.5']['A_hand'] for r in rs], 0)
    d = np.arange(1, len(single) + 1)
    assert len(d) == 4 * dt + 8
    ax.plot(d, split, color=DTCOL[dt], lw=1.3, ls='-', zorder=3, solid_capstyle='butt')
    ax.plot(d, single, color=DTCOL[dt], lw=1.0, ls=(0, (3, 1.6)), zorder=4)
    pb[dt] = dict(single=single, split=split, d_ext=[r['d_ext']['cal_0.5'][0] for r in rs],
                  D=[r['policy_0.5']['D_call'] for r in rs])
ax.set_xlim(0, 41)
ax.set_ylim(-0.03, 1.05)
ax.set_xticks([1, 8, 16, 24, 32, 40])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
ax.set_xlabel('nesting depth of the plan', labelpad=1.5)
ax.set_ylabel('fraction of plans solved', labelpad=2)
ax.set_title('(b) nested plans: 12-layer RoPE model', loc='left', pad=3)
ax.grid(axis='y', color=GRID, lw=0.5)
ax.set_axisbelow(True)
hb = [Line2D([], [], color=DTCOL[dt], lw=1.3, label=f'training depth {dt}') for dt in (2, 4, 6, 8)]
hb += [Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.6)), label='one call'),
       Line2D([], [], color=INK, lw=1.3, label='split into calls')]
ax.legend(handles=hb, loc='center right', frameon=False, handlelength=1.8, handletextpad=0.5, labelspacing=0.3,
          borderaxespad=0.2, bbox_to_anchor=(1.0, 0.47))
out['b'] = pb

# ---------------- (c) nested tool calls, 8 models ----------------
ax = axs[1, 0]
for m in MAIN:
    ax.plot(DEPTHS, curve(m, 'A'), color=C_A, lw=0.5, alpha=0.45, zorder=2)
mo_I, mo_A = np.mean([curve(m, 'I') for m in OLMO], 0), np.mean([curve(m, 'A') for m in OLMO], 0)
ax.plot(DEPTHS, mo_I, color=C_OLMO, lw=1.0, ls=(0, (3, 1.6)), zorder=3)
ax.plot(DEPTHS, mo_A, color=C_OLMO, lw=1.0, zorder=3)
m_I, m_A = np.mean([curve(m, 'I') for m in MAIN], 0), np.mean([curve(m, 'A') for m in MAIN], 0)
ax.plot(DEPTHS, m_I, color=C_I, lw=1.4, ls=(0, (3, 1.6)), zorder=5)
ax.plot(DEPTHS, m_A, color=C_A, lw=1.6, zorder=5)
out['c'] = dict(mean_I=m_I, mean_A=m_A, olmo_I=mo_I, olmo_A=mo_A,
                per_model_A={m: curve(m, 'A') for m in MAIN}, per_model_I={m: curve(m, 'I') for m in MAIN})
hc = [Line2D([], [], color=C_I, lw=1.4, ls=(0, (3, 1.6)), label='one call'),
      Line2D([], [], color=C_A, lw=1.6, label='one level per call'),
      Line2D([], [], color=C_A, lw=0.5, alpha=0.6, label='each model'),
      Line2D([], [], color=C_OLMO, lw=1.0, label='OLMo-2, mean of 3')]
ax.legend(handles=hc, loc='upper left', frameon=False, handlelength=1.3, handletextpad=0.4, labelspacing=0.3,
          columnspacing=0.8, borderaxespad=0.0, borderpad=0.1, bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title('(c) nested tool calls: mean of 8 models', loc='left', pad=3)

# ---------------- (d) the 4 models whose single-call limit is 2 or 3 levels ----------------
ax = axs[1, 1]
f_I, f_A, f_H = (np.mean([curve(m, c) for m in FOUR], 0) for c in ('I', 'A', 'H0.8'))
ax.plot(DEPTHS, f_I, color=C_I, lw=1.4, ls=(0, (3, 1.6)), zorder=5)
ax.plot(DEPTHS, f_A, color=C_A, lw=1.6, zorder=5)
ax.plot(DEPTHS, f_H, color=C_H, lw=1.6, zorder=5)
out['d'] = dict(mean_I=f_I, mean_A=f_A, mean_H=f_H, per_model_H={m: curve(m, 'H0.8') for m in FOUR},
                dhat={m: DHAT[m] for m in FOUR})
hd = [Line2D([], [], color=C_I, lw=1.4, ls=(0, (3, 1.6)), label='one call'),
      Line2D([], [], color=C_A, lw=1.6, label='one level per call'),
      Line2D([], [], color=C_H, lw=1.6, label='calls of $\\hat d$ levels')]
ax.legend(handles=hd, loc='upper left', frameon=False, handlelength=1.3, handletextpad=0.4, labelspacing=0.3,
          columnspacing=0.8, borderaxespad=0.0, borderpad=0.1, bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title('(d) 4 models with $\\hat d$ of 2\u20133', loc='left', pad=3)

for ax in (axs[1, 0], axs[1, 1]):
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.03, 1.36)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.spines['left'].set_bounds(0, 1)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('nesting depth (functions in the call chain)', labelpad=1.5)
    ax.set_ylabel('fraction answered correctly')
    ax.yaxis.set_label_coords(-0.135, (0.5 + 0.03) / 1.39)
    ax.grid(axis='y', color=GRID, lw=0.5)
    for gl, yv in zip(ax.get_ygridlines(), ax.get_yticks()):
        gl.set_visible(yv <= 1.0)
    ax.set_axisbelow(True)
for ax in axs.flat:
    ax.tick_params(length=2.0, pad=1.5)

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
    if a.get_legend():
        texts += list(a.get_legend().get_texts())
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
# data (markers and line vertices, densified) under legends
for a in fig.axes:
    lg = a.get_legend()
    cover = ([lg.get_window_extent(r_)] if lg else []) + [t.get_window_extent(r_) for t in a.texts]
    pts, mpts = [], []
    for coll in a.collections:
        mpts += list(a.transData.transform(coll.get_offsets()))
    for ln in a.lines:
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        for k in range(len(xd) - 1):
            for t_ in np.linspace(0, 1, 25):
                pts.append(a.transData.transform((xd[k] + t_ * (xd[k + 1] - xd[k]), yd[k] + t_ * (yd[k + 1] - yd[k]))))
    rad = 3.0 / 72 * fig.dpi   # marker radius plus a margin, in pixels
    for e in cover:
        hit = [p for p in pts if e.x0 + 1 <= p[0] <= e.x1 - 1 and e.y0 + 1 <= p[1] <= e.y1 - 1]
        hit += [p for p in mpts if e.x0 - rad <= p[0] <= e.x1 + rad and e.y0 - rad <= p[1] <= e.y1 + rad]
        if hit:
            issues.append(('data under legend/text', a.get_title(loc='left')[:14], len(hit)))
    if lg:
        e = lg.get_window_extent(r_)
        lo = a.transData.inverted().transform((e.x0, e.y0)); hi = a.transData.inverted().transform((e.x1, e.y1))
        print('legend data extent', a.get_title(loc='left')[:14], 'x %.2f..%.2f y %.2f..%.2f' % (lo[0], hi[0], lo[1], hi[1]))
for a in fig.axes:
    lg = a.get_legend()
    if lg:
        e, ab = lg.get_window_extent(r_), a.get_window_extent(r_)
        if e.x0 < ab.x0 - 2 or e.x1 > ab.x1 + 2:
            issues.append(('legend wider than axes', a.get_title(loc='left')[:14], round((e.x1 - ab.x1) / fig.dpi, 3)))
        print('legend', a.get_title(loc='left')[:14], 'width %.2f in, axes width %.2f in' % (e.width / fig.dpi, ab.width / fig.dpi))
print('layout issues:', issues if issues else 'none')
for a in fig.axes:
    print('axes', a.get_title(loc='left'), 'bbox (in)', tuple(round(v / fig.dpi, 2) for v in a.get_window_extent(r_).bounds))
print('figure size (in):', fig.get_size_inches())
fig.savefig(OUTPDF)
fig.savefig(OUTPDF.replace('.pdf', '.png'), dpi=220)
print('wrote', OUTPDF)

# ---------------- plotted numbers ----------------
f3 = lambda v: ' '.join('%.3f' % x for x in v)
print('\n(a) d_ext (calibration set, threshold 0.5) per run, by training depth and encoding')
for dt in (2, 4, 6, 8):
    for pe in ('rope', 'ape'):
        vals = sorted(v for n, d, p, v, h in pa if d == dt and p == pe and not h)
        hol = sorted(v for n, d, p, v, h in pa if d == dt and p == pe and h)
        print(f'  dtrain {dt} {pe:4s}: n={len(vals)} range {min(vals)}-{max(vals)} values {vals}' +
              (f'  hollow {hol}' if hol else ''))
print('\n(b) 12-layer RoPE, mean of 3 seeds (depth 1..4*dtrain+8)')
for dt in (2, 4, 6, 8):
    print(f'  dtrain {dt}: d_ext {pb[dt]["d_ext"]} call depth {pb[dt]["D"]}')
    print(f'    one call : {f3(pb[dt]["single"])}')
    print(f'    split    : {f3(pb[dt]["split"])}')
    print(f'    split, min over depths > max d_ext ({max(pb[dt]["d_ext"])}): {pb[dt]["split"][max(pb[dt]["d_ext"]):].min():.3f};'
          f' min over all depths {pb[dt]["split"].min():.3f} at depth {int(np.argmin(pb[dt]["split"])) + 1}')
print('\ndepths', DEPTHS)
print('(c) mean of 8, one call        :', f3(out['c']['mean_I']))
print('(c) mean of 8, one level/call  :', f3(out['c']['mean_A']))
print('(c) OLMo-2 mean, one call      :', f3(mo_I))
print('(c) OLMo-2 mean, one level/call:', f3(mo_A))
for m in MAIN:
    print('    %-18s A: %s' % (m, f3(curve(m, 'A'))))
print('(d) models', FOUR, 'single-call limit', [DHAT[m] for m in FOUR])
print('(d) mean of 4, one call        :', f3(f_I))
print('(d) mean of 4, one level/call  :', f3(f_A))
print('(d) mean of 4, single-call limit:', f3(f_H))
print('(d) at d=8: calls per chain with the single-call limit:',
      {m: T[m]['T']['costs']['H0.8']['calls']['8'] for m in FOUR})
json.dump(dict(depths=DEPTHS, c={k: (v.tolist() if hasattr(v, 'tolist') else {kk: vv.tolist() for kk, vv in v.items()})
                                 for k, v in out['c'].items()},
               d={k: (v.tolist() if hasattr(v, 'tolist') else ({kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                                                                  for kk, vv in v.items()}))
                  for k, v in out['d'].items()},
               b={str(dt): dict(single=v['single'].tolist(), split=v['split'].tolist(), d_ext=v['d_ext'], D=v['D'])
                  for dt, v in pb.items()},
               a=[dict(run=n, dtrain=d, pe=p, d_ext=v, hollow=h) for n, d, p, v, h in pa]),
          open(os.path.join(OUT, 'fig_split_values.json'), 'w'), indent=1)
