"""LaTeX rows of two supplementary result tables: all 52 pairs of Experiment 2 (App. M.2.1, structural_audit)
and the ranges per model of Experiment 4 (App. M.3.1, lemma2_pretrained)."""
import json, csv, os
HERE = os.path.dirname(os.path.abspath(__file__))  # extensions/
M2 = f'{HERE}/structural_audit/results'
M3 = f'{HERE}/lemma2_pretrained/results'

# ---------------- App. M.2: all pairs ----------------
d = json.load(open(f'{M2}/results.json'))
main = d['main']['pairs']; nos = {(p['family'], p['a'], p['b']): p for p in d['noS']['pairs']}
fam_dec = d['main']['separation']['per_family']
NAME = {'pick_and_place_simple': 'simple', 'look_at_obj_in_light': 'look', 'pick_clean_then_place_in_recep': 'clean',
        'pick_heat_then_place_in_recep': 'heat', 'pick_cool_then_place_in_recep': 'cool', 'pick_two_obj_and_place': 'two',
        'alfworld': 'ALFWorld', 'sciworld': 'ScienceWorld', 'webshop': 'WebShop', 'human': 'human', 'gpt': 'GPT-4',
        'SWEagent_llama70b': 'SWE-agent', 'OpenHands_qwen3coder480b': 'OpenHands',
        'PICKPLACE': 'pick-place', 'OPEN': 'open', 'CLOSE': 'close', 'FOLD': 'fold', 'SWEEP': 'sweep', 'FLIP': 'flip', 'TURN': 'turn',
        'goto': 'go to', 'pickup': 'pick up', 'open': 'open', 'putnext': 'put next', 'pick up seq go to': 'sequence'}
FAM = [('A', 'A: ALFWorld task types (ETO), $d_\\Phi=14$'), ('B', 'B: ETO environments, $d_\\Phi=12$'),
       ('B2', 'B2: WebShop demonstrators (ETO), $d_\\Phi=12$'), ('C_H1', 'C: harness and model, shared issues, $d_\\Phi=16$'),
       ('C_H2', 'C: harness and model, all resolved issues, $d_\\Phi=16$'), ('D', 'D: Bridge skills (ECoT, VLA), $d_\\Phi=16$'),
       ('E', 'E: BabyAI instruction kinds, $d_\\Phi=13$')]
lines = []
for f, title in FAM:
    s, n = fam_dec[f]['separated'], fam_dec[f]['pairs']
    lines.append(r'\multicolumn{6}{@{}l}{\emph{%s; %d of %d separated}} \tabularnewline' % (title, s, n))
    for p in main:
        if p['family'] != f: continue
        q = nos[(f, p['a'], p['b'])]
        assert p['complete'] and p['warnings'] == 0
        sep = p['separated']
        assert sep == (p['R'] >= 3 and p['cv'] <= 0.10)
        R = '%.2f' % p['R']
        R = r'\textbf{%s}' % R if sep else R
        lines.append(r'\quad %s vs %s & %d & %.2f (%.3f) & %.2f / %.2f & %s & %.2f \tabularnewline' % (
            NAME[p['a']], NAME[p['b']], p['N'], p['W'], p['W_sd'], p['B_a'], p['B_b'], R, q['R']))
    lines.append(r'\addlinespace[1pt]')
lines = lines[:-1]
open(f'{M2}/paper_table_pairs.tex', 'w').write('\n'.join(lines) + '\n')
print('pairs', len(main), 'separated', sum(p['separated'] for p in main), 'noS', d['noS']['separation']['n_separated'],
      'pairz', d['pairz']['separation']['n_separated'], 'maxcv', max(p['cv'] for p in main))
# summary numbers used in the text
import statistics
for f, _ in FAM:
    rs = [p['R'] for p in main if p['family'] == f]
    bs = [x for p in main if p['family'] == f for x in (p['B_a'], p['B_b'])]
    ws = [p['W'] for p in main if p['family'] == f]
    print(f, 'R %.2f-%.2f median %.2f' % (min(rs), max(rs), statistics.median(rs)), 'B %.2f-%.2f' % (min(bs), max(bs)),
          'W %.2f-%.2f' % (min(ws), max(ws)), 'N', sorted(set(p['N'] for p in main if p['family'] == f)))
print('meanbase sep', sum(1 for p in main if p['R_meanbase'] >= 3 and p['cv'] <= 0.10))

# ---------------- App. M.3.1: ranges per model ----------------
rows = list(csv.DictReader(open(f'{M3}/table_cells.csv')))
fams = ['countdown_dfs', 'countdown_bfs', 'dyck', 'code']
MOD = [('gpt2', 'GPT-2 124M', 'APE'), ('gpt2-medium', 'GPT-2 355M', 'APE'), ('gpt-neo-125m', 'GPT-Neo 125M', 'APE'),
       ('gpt-neo-1.3B', 'GPT-Neo 1.3B', 'APE'), ('pythia-160m', 'Pythia 160M', 'RoPE'), ('pythia-410m', 'Pythia 410M', 'RoPE'),
       ('pythia-1.4b', 'Pythia 1.4B', 'RoPE'), ('smollm2-360m', 'SmolLM2 360M', 'RoPE')]

def sel(m, p):
    s = [r for r in rows if r['model'] == m and r['precision'] == p and r['family'] in fams and r['k'] in ('16', '64', '256')]
    assert len(s) == 12, (m, p, len(s))
    return s

def sci(x):
    if x == 0: return '0'
    m, e = ('%.1e' % x).split('e')
    return r'%s\text{e}{%d}' % (m, int(e))

def fmt(x, kind):
    return {'f2': lambda v: '%.2f' % v, 'f1': lambda v: '%.1f' % v, 'sci': sci,
            'g2': lambda v: ('0' if v == 0 else ('%.2g' % v))}[kind](x)

def rng(vals, kind, scale=1.0, sep='--'):
    lo, hi = min(vals) * scale, max(vals) * scale
    a, b = fmt(lo, kind), fmt(hi, kind)
    if a == b: return '$%s$' % a
    return ('$%s$--$%s$' % (a, b)) if sep == '--' else ('$%s$ to $%s$' % (a, b))

out = []
for i, (m, name, pe) in enumerate(MOD):
    if i == 0: out.append(r'\multicolumn{7}{@{}l}{\emph{learned absolute position embeddings (APE)}} \tabularnewline')
    if i == 4: out.append(r'\midrule' + '\n' + r'\multicolumn{7}{@{}l}{\emph{rotary position embeddings (RoPE)}} \tabularnewline')
    s32, s64 = sel(m, 'fp32'), sel(m, 'fp64diag')
    g = lambda s, k: [float(r[k]) for r in s]
    if pe == 'APE':
        cells = (rng(g(s32, 'm1_first_max'), 'f2'), rng(g(s32, 'm1_all_max'), 'f2'), rng(g(s64, 'm1_all_max'), 'f2'),
                 rng(g(s64, 'F_base_all_max'), 'sci'), rng(g(s32, 'm2'), 'f1', 100), rng(g(s32, 'm3_signed'), 'f2', sep='to'))
    else:
        cells = (rng(g(s32, 'm1_first_max'), 'sci'), rng(g(s32, 'm1_all_max'), 'sci'), rng(g(s64, 'm1_all_max'), 'sci'),
                 rng(g(s64, 'F_base_all_max'), 'sci'), rng(g(s32, 'm2'), 'g2', 100), rng(g(s32, 'm3_signed'), 'sci', sep='to'))
    out.append(name + ' & ' + ' & '.join(cells) + r' \tabularnewline')
open(f'{M3}/paper_table_models.tex', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
