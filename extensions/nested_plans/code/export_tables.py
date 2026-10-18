"""Write the result tables of results/ from the output of analyze.py and the analysed run directories.
  runs.csv       one row per analysed run (120), from BASE/analysis/summary.json
  per_depth.csv  one row per run and test depth, from BASE/analysis/summary.json
  selection_30000_steps.csv  one row per run of the 30,000-step grid, from BASE/runs/gatecheck.json (gatecheck.py)
  runs/<name>.json  BASE/<dir>/results.json of every analysed run (runs_ext/ for the re-trained cells), with
                 config.root (the absolute --root of train.py) replaced by that directory relative to BASE
Values are copied unchanged (floats written with Python's repr); runs are sorted by PE, d_train, model and seed.
Usage: python export_tables.py [--base BASE] [--out OUT]   (defaults: this experiment's folder and its results/)"""
import argparse, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--base', default=os.path.dirname(HERE), help='folder with analysis/, runs/ and runs_ext/')
ap.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'results'))
args = ap.parse_args()

S = json.load(open(os.path.join(args.base, 'analysis', 'summary.json')))
runs = S['runs']
PE_NAME = {'rope': 'RoPE', 'ape': 'APE'}
order = sorted(runs, key=lambda n: (C.PES.index(runs[n]['pe']), runs[n]['dtrain'], C.MODELS.index(runs[n]['model']),
                                    runs[n]['seed']))


def cell(v):
    if v is None:
        return ''
    if isinstance(v, float):
        return repr(v)
    return v


def write(name, header, rows):
    with open(os.path.join(args.out, name), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        for r in rows:
            w.writerow([cell(v) for v in r])


os.makedirs(os.path.join(args.out, 'runs'), exist_ok=True)
head = ['run', 'position_encoding', 'd_train', 'model', 'layers', 'width', 'seed', 'non_embedding_parameters',
        'training_steps', 'd_test', 'd_ext_calibration', 'd_ext_test', 'call_depth', 'split_possible',
        'single_call_success_beyond_d_ext', 'split_success_beyond_d_ext_mean', 'split_success_beyond_d_ext_min',
        'calls_per_plan_beyond_d_ext', 'single_call_ge_0.9_up_to_d_train']
rows, rows_d = [], []
for n in order:
    x = runs[n]
    P = x['policy_0.5']
    R = P['R']
    rows.append([n, PE_NAME[x['pe']], x['dtrain'], x['model'], x['L'], x['width'], x['seed'], x['params_nonembed'],
                 x['steps'], x['dtest'], x['d_ext']['cal_0.5'][0], x['d_ext']['test_0.5'][0], P['D_call'],
                 P['constructible'], R.get('mean_single'), R.get('mean_hand'), R.get('min_hand'), P['mean_calls_B'],
                 x['gate_a']])
    for i in range(x['dtest']):
        rows_d.append([n, i + 1, x['A_single'][i], P['A_hand'][i], x['A_single_cal'][i], P['calls'][i]])
    res = json.load(open(os.path.join(args.base, x['dir'], 'results.json')))
    res['config']['root'] = os.path.dirname(x['dir'])
    with open(os.path.join(args.out, 'runs', n + '.json'), 'w') as f:
        json.dump(res, f, indent=1)
        f.write('\n')
write('runs.csv', head, rows)
write('per_depth.csv', ['run', 'depth', 'single_call_success', 'split_success', 'single_call_success_calibration',
                        'calls_per_plan'], rows_d)
G = json.load(open(os.path.join(args.base, 'runs', 'gatecheck.json')))
dmax = max(C.D_TRAINS)
rows_g = []
for n in sorted(G['per_run'], key=lambda n: (C.PES.index(C.cell_of(n)[0]), C.cell_of(n)[1],
                                             C.MODELS.index(C.cell_of(n)[2]), n)):
    pe, dt, m = C.cell_of(n)
    a = G['per_run'][n]['A_in']
    rows_g.append([n, PE_NAME[pe], dt, m, C.STEPS] + a + [None] * (dmax - len(a)) +
                  [G['per_run'][n]['gate_a'], n in G['rescue_names']])
write('selection_30000_steps.csv', ['run', 'position_encoding', 'd_train', 'model', 'training_steps'] +
      [f'single_call_success_d{d}' for d in range(1, dmax + 1)] +
      ['single_call_ge_0.9_up_to_d_train', 'cell_retrained_60000_steps'], rows_g)
print(len(rows), 'runs;', len(rows_d), 'run-depth rows;', len(rows_g), 'runs of 30,000 steps; written to', args.out)
