"""Analysis (CPU) of results/raw/*: the metrics m1-m3 for every model x k x input family and pooled over the families,
the controls V1-V3, the position profile D2, the logit margin at flipped positions D3, and the threshold checks
R1-R4. Writes results/summary.json, results/table_cells.csv and results/table_pooled.tex.
Checks, made in each of the 96 cells (model x k x family; comparisons at the same k and family):
  R1  RoPE: fp32 m1_first_max <= 1e-5 and float64 m1_all_max <= 1e-10
  R2  APE: fp32 m1_first_max >= 100 x the largest RoPE fp32 m1_all_max, and the same for the means;
      R2_fp64: APE float64 m1_first_max >= 100 x the largest RoPE float64 m1_all_max
  R3  RoPE: m2 < 1%
  R4  APE: m2 >= 1% and m2 >= 10 x the largest RoPE m2
  R1_all_layers: RoPE fp32 m1_all_max <= 1e-5 (all layers); listed in summary.json only.
Controls: V1 explicit position ids reproduce the default forward pass (max abs logit difference <= 1e-5); V2 the
replicate row (k=0) gives exactly 0, or an m1_all_max at most 0.1 x that of every shifted row; V3 the independent
layer-1 scores of results/verify_layer1.json agree (APE within 1% relative; RoPE <= 1e-5).
summary.json also lists the cells that fail each check, the RoPE cells whose float64 m1_all_max exceeds 1e-6, and
all_checks_pass (R1-R4 hold in every cell and the controls V1-V3 pass)."""
import json, os, csv, math
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/lemma2_pretrained
RAW = os.environ.get('LEMMA2_RAW', f'{HERE}/results/raw')
RES = os.environ.get('LEMMA2_RES', f'{HERE}/results')
KS = (0, 0, 16, 64, 256)
ROWS = {16: 2, 64: 3, 256: 4}
FAMILIES = ('countdown_dfs', 'countdown_bfs', 'dyck', 'code')
MODELS = [('gpt2', 'APE'), ('gpt2-medium', 'APE'), ('gpt-neo-125m', 'APE'), ('gpt-neo-1.3B', 'APE'),
          ('pythia-160m', 'RoPE'), ('pythia-410m', 'RoPE'), ('pythia-1.4b', 'RoPE'), ('smollm2-360m', 'RoPE')]
LICENSE = {'gpt2': 'MIT', 'gpt2-medium': 'MIT', 'gpt-neo-125m': 'MIT', 'gpt-neo-1.3B': 'MIT',
           'pythia-160m': 'Apache-2.0', 'pythia-410m': 'Apache-2.0', 'pythia-1.4b': 'Apache-2.0',
           'smollm2-360m': 'Apache-2.0'}


def load(tag, prec):
    j = json.load(open(f'{RAW}/{tag}_{prec}.json'))
    z = dict(np.load(f'{RAW}/{tag}_{prec}.npz'))
    return j, z


def cat(z, fams, key):
    return np.concatenate([z[f'{f}/{key}'] for f in fams], axis=0)


def cell(z, j, fams, b):
    maxA, sumA, n = cat(z, fams, 'maxA'), cat(z, fams, 'sumA'), cat(z, fams, 'n')
    maxD, sumD = cat(z, fams, 'maxD')[:, :, b], cat(z, fams, 'sumD')[:, :, b]
    L = maxA.shape[1]
    lay_max = [float(maxD[:, l].max() / maxA[:, l].max()) for l in range(L)]
    lay_mean = [float(sumD[:, l].sum() / sumA[:, l].sum()) for l in range(L)]
    flips = cat(z, fams, 'flip_pos')[:, b, :]
    dlp = cat(z, fams, 'dlp_pos')[:, b, :]
    nll = sum(sum(j['families'][f]['base_nll_sum']) for f in fams) / sum(j['families'][f]['n_seq'] * j['families'][f]['n_lp'] for f in fams)
    out = {
        'm1_first_max': lay_max[0], 'm1_first_mean': lay_mean[0],
        'm1_all_max': max(lay_max), 'm1_all_mean': float(sumD.sum() / sumA.sum()),
        'm1_first_max_abs': float(maxD[:, 0].max()), 'm1_first_mean_abs': float(sumD[:, 0].sum() / n[:, 0].sum()),
        'm1_all_max_abs': float(maxD.max()), 'm1_all_mean_abs': float(sumD.sum() / n.sum()),
        'score_scale_first_max': float(maxA[:, 0].max()), 'score_scale_first_mean': float(sumA[:, 0].sum() / n[:, 0].sum()),
        'm1_layer_max': lay_max, 'm1_layer_mean': lay_mean, 'argmax_layer': int(np.argmax(lay_max)),
        'm2': float(flips.mean()), 'm2_count': int(flips.sum()), 'n_positions': int(flips.size),
        'm3_signed': float(dlp.mean()), 'm3_abs': float(np.abs(dlp).mean()), 'base_nll': float(nll),
        'n_seq': int(maxA.shape[0]),
        # D2: position profile, 8 bins of 64 positions
        'D2_flip_by_pos64': [float(flips[:, i:i + 64].mean()) for i in range(0, flips.shape[1], 64)],
        'D2_absdlp_by_pos64': [float(np.abs(dlp[:, i:i + 64]).mean()) for i in range(0, dlp.shape[1], 64)],
    }
    if f'{fams[0]}/maxF' in z:
        maxF, sumF = cat(z, fams, 'maxF'), cat(z, fams, 'sumF')
        for r, name in ((0, 'base'), (b, 'shift')):
            out[f'F_{name}_first_max'] = float(maxF[:, 0, r].max() / maxA[:, 0].max())
            out[f'F_{name}_all_max'] = max(float(maxF[:, l, r].max() / maxA[:, l].max()) for l in range(L))
            out[f'F_{name}_all_mean'] = float(sumF[:, :, r].sum() / sumA.sum())
        mlog = [x for f in fams for x in j['families'][f]['logit_fp32_vs_fp64_maxabs']]
        out['logit_fp32_vs_fp64_maxabs_base'] = float(max(x[0] for x in mlog))
    # D3: base top1-top2 logit margin at flipped positions (max over sequences)
    mm = [s[b] for f in fams for s in j['families'][f]['flip_margin_max'] if s[b] is not None]
    out['D3_flip_margin_max'] = float(max(mm)) if mm else None
    return out


def main():
    global MODELS
    if os.environ.get('LEMMA2_TEST'):
        MODELS = [(t, e) for t, e in MODELS if os.path.exists(f'{RAW}/{t}_fp32.json')]
    data = {}
    for tag, enc in MODELS:
        data[tag] = {p: load(tag, p) for p in ('fp32', 'fp64diag') if os.path.exists(f'{RAW}/{tag}_{p}.json')}
    groups = list(FAMILIES) + ['pooled']
    cells = {}
    for tag, enc in MODELS:
        for prec, (j, z) in data[tag].items():
            for g in groups:
                fams = FAMILIES if g == 'pooled' else (g,)
                for k, b in list(ROWS.items()) + [('rep0', 1)]:
                    cells[(tag, prec, g, k)] = cell(z, j, fams, b)
    # ---------------- controls ----------------
    V1 = {f'{tag}/{p}': data[tag][p][0]['S1_maxabs_logit_diff_default_vs_arange'] for tag, _ in MODELS for p in data[tag]}
    V1_pass = all(v <= 1e-5 for v in V1.values())
    rep = {f'{tag}/{p}/{g}': {m: cells[(tag, p, g, 'rep0')][m] for m in ('m1_all_max', 'm1_all_mean', 'm2', 'm3_abs')}
           for tag, _ in MODELS for p in data[tag] for g in FAMILIES}
    V2_exact0 = all(all(v == 0 for v in d.values()) for d in rep.values())
    V2_floor_ok = all(cells[(t, p, g, 'rep0')]['m1_all_max'] <= 0.1 * min(cells[(t, p, g, k)]['m1_all_max'] for k in ROWS)
                      for t, e in MODELS for p in data[t] for g in FAMILIES)
    V2_pass = V2_exact0 or V2_floor_ok
    v3 = json.load(open(f'{HERE}/results/verify_layer1.json')) if os.path.exists(f'{HERE}/results/verify_layer1.json') else None
    V3_detail, V3_pass = {}, None
    if v3:
        V3_pass = True
        for tag, d in v3.items():
            enc = dict(MODELS)[tag]
            for k in ('16', '64', '256'):
                ind, capv = d['independent_fp64_m1_first_max'][k], d['captured_fp32_m1_first_max'][k]
                if enc == 'APE':
                    ok = abs(ind - capv) / capv <= 0.01
                else:
                    ok = ind <= 1e-5 and capv <= 1e-5
                V3_detail[f'{tag}/k={k}'] = {'independent': ind, 'captured': capv, 'pass': ok}
                V3_pass &= ok
            V3_detail[f'{tag}/scale_rel_diff'] = {'maxA': d['scale_maxA_rel_diff'], 'sumA': d['scale_sumA_rel_diff']}
    # ---------------- checks R1-R4 ----------------
    ape = [t for t, e in MODELS if e == 'APE']; rope = [t for t, e in MODELS if e == 'RoPE']
    have64 = all('fp64diag' in data[t] for t, _ in MODELS)
    fails = {'R1': [], 'R1_all_layers': [], 'R2': [], 'R2_fp64': [], 'R3': [], 'R4': []}
    rope64_above = []
    for g in FAMILIES:
        for k in ROWS:
            c32 = {t: cells[(t, 'fp32', g, k)] for t, _ in MODELS}
            c64 = {t: cells[(t, 'fp64diag', g, k)] for t, _ in MODELS} if have64 else None
            rmax32 = max(c32[t]['m1_all_max'] for t in rope); rmean32 = max(c32[t]['m1_all_mean'] for t in rope)
            rm2 = max(c32[t]['m2'] for t in rope)
            for t in rope:
                c = c32[t]
                if not (c['m1_first_max'] <= 1e-5 and (c64 is not None and c64[t]['m1_all_max'] <= 1e-10)):
                    fails['R1'].append(f'{t}/{g}/k={k}')
                if not c['m1_all_max'] <= 1e-5:
                    fails['R1_all_layers'].append(f'{t}/{g}/k={k}')
                if not c['m2'] < 0.01:
                    fails['R3'].append(f'{t}/{g}/k={k}')
                if c64 is not None and c64[t]['m1_all_max'] > 1e-6:
                    rope64_above.append(f'{t}/{g}/k={k}')
            for t in ape:
                c = c32[t]
                if not (c['m1_first_max'] >= 100 * rmax32 and c['m1_first_mean'] >= 100 * rmean32):
                    fails['R2'].append(f'{t}/{g}/k={k}')
                if c64 is not None and not c64[t]['m1_first_max'] >= 100 * max(c64[r]['m1_all_max'] for r in rope):
                    fails['R2_fp64'].append(f'{t}/{g}/k={k}')
                if not (c['m2'] >= 0.01 and c['m2'] >= 10 * rm2):
                    fails['R4'].append(f'{t}/{g}/k={k}')
    rules_ok = have64 and not any(fails[r] for r in ('R1', 'R2', 'R2_fp64', 'R3', 'R4'))
    verified = V1_pass and V2_pass and bool(V3_pass)
    all_checks_pass = rules_ok and verified
    summary = {
        'all_checks_pass': all_checks_pass, 'rules_all_cells_pass': rules_ok, 'failing_cells': fails,
        'rope_fp64_cells_above_1e-6': rope64_above, 'R1_all_layers_holds_all_cells': not fails['R1_all_layers'], 'V1': V1, 'V1_pass': V1_pass,
        'V2_replicate_exact_zero': V2_exact0, 'V2_pass': V2_pass, 'V2_replicate_values': rep, 'V3_pass': V3_pass, 'V3': V3_detail,
        'licenses': LICENSE, 'devices': {f'{t}/{p}': data[t][p][0]['device'] for t, _ in MODELS for p in data[t]},
        'cells': {f'{t}|{p}|{g}|{k}': v for (t, p, g, k), v in cells.items()},
    }
    json.dump(summary, open(f'{RES}/summary.json', 'w'), indent=1)
    # ---------------- tables ----------------
    cols = ['m1_first_max', 'm1_first_mean', 'm1_all_max', 'm1_all_mean', 'm1_first_max_abs', 'm1_first_mean_abs',
            'm1_all_max_abs', 'm1_all_mean_abs', 'm2', 'm2_count', 'n_positions', 'm3_signed', 'm3_abs', 'base_nll', 'n_seq']
    fcols = ['F_base_all_max', 'F_shift_all_max', 'F_base_all_mean', 'F_shift_all_mean']
    with open(f'{RES}/table_cells.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'encoding', 'license', 'precision', 'family', 'k'] + cols + fcols)
        for (t, p, g, k), v in cells.items():
            w.writerow([t, dict(MODELS)[t], LICENSE[t], p, g, k] + [v[c] for c in cols] + [v.get(c, '') for c in fcols])

    def sci(x):
        if x == 0:
            return '$0$'
        e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
        if -2 <= e <= 0:
            return f'${x:.2f}$' if e >= -1 else f'${x:.3f}$'
        if round(abs(m), 1) >= 10:
            m, e = m / 10, e + 1
        return f'${m:.1f}\\text{{e}}{e}$'

    lines = [r'\begin{tabular}{lr ccccc r rr}', r'\toprule',
             r' & & \multicolumn{2}{c}{$m_1$, layer 1} & \multicolumn{2}{c}{$m_1$, all layers} & $m_1$, all & $m_2$ & \multicolumn{2}{c}{$m_3$ (nats/token)} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}',
             r'Model & $k$ & max & mean & max & mean & max, fp64 & flips (\%) & signed & $|\cdot|$ \\', r'\midrule']
    for i, (t, enc) in enumerate(MODELS):
        if i == 0 or enc != MODELS[i - 1][1]:
            if i > 0:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{10}{l}{\emph{%s}} \\' % ('learned absolute position embeddings (APE)' if enc == 'APE' else 'rotary position embeddings (RoPE)'))
        for k in ROWS:
            c = cells[(t, 'fp32', 'pooled', k)]
            c64 = cells.get((t, 'fp64diag', 'pooled', k), {})
            name = t if k == 16 else ''
            m2 = c['m2'] * 100
            m2s = '$0$' if c['m2_count'] == 0 else (f'${m2:.1f}$' if m2 >= 0.1 else f'${m2:.3f}$')
            lines.append(f'{name} & {k} & {sci(c["m1_first_max"])} & {sci(c["m1_first_mean"])} & {sci(c["m1_all_max"])} & '
                         f'{sci(c["m1_all_mean"])} & {sci(c64["m1_all_max"]) if c64 else "--"} & {m2s} & '
                         f'{sci(c["m3_signed"])} & {sci(c["m3_abs"])} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(f'{RES}/table_pooled.tex', 'w').write('\n'.join(lines) + '\n')
    print('all_checks_pass', all_checks_pass, '| rules_ok', rules_ok, '| V1', V1_pass, 'V2', V2_exact0, 'V3', V3_pass)
    print('fails', {k: len(v) for k, v in fails.items()}, 'rope_fp64_above_1e-6', len(rope64_above))
    for t, enc in MODELS:
        for k in ROWS:
            c = cells[(t, 'fp32', 'pooled', k)]; c64 = cells.get((t, 'fp64diag', 'pooled', k), {})
            print(f'{t:13s} {enc:4s} k={k:3d} first_max={c["m1_first_max"]:.2e} first_mean={c["m1_first_mean"]:.2e} '
                  f'all_max={c["m1_all_max"]:.2e} all_mean={c["m1_all_mean"]:.2e} fp64_all_max={c64.get("m1_all_max", float("nan")):.2e} '
                  f'F_base_all_max={c64.get("F_base_all_max", float("nan")):.2e} m2={c["m2"]*100:.3f}% m3={c["m3_signed"]:.2e} |m3|={c["m3_abs"]:.2e}')


if __name__ == '__main__':
    main()
