"""Writes the result tables (CSV) from the output of `analyze2.py metrics` (metrics.json) and the per-model files
of run2.py (dhat_T.json, budgets_T.json, T_rerun.json). Values are copied unchanged. The script also checks that the
calibration values in metrics.json equal those in dhat_T.json and that budgets_T.json uses the calibrated D.
Usage (from the folder nested_tool_calls/): python code/export_tables.py [--metrics results/metrics.json]
       [--per-model results/per_model] [--out results]
--per-model is a folder with one subfolder per model, as results/ of a run."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from models import MODELS, ORDER, FAMILY
from analyze2 import BIG_B

TAUS = ('0.8', '0.9')
# (key in metrics.json, condition, threshold)
CONDS = [('I', 'I', ''), ('E', 'E', ''), ('A', 'A', ''), ('H0.8', 'H', '0.8'), ('H0.9', 'H', '0.9')]


def rd(p):
    with open(p) as f:
        return json.load(f)


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        for r in rows:
            assert len(r) == len(header), (path, r)
            w.writerow(['' if v is None else v for v in r])
    print(f'{path}: {len(rows)} rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default='results/metrics.json')
    ap.add_argument('--per-model', default='results/per_model')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    M = rd(a.metrics)['results']
    os.makedirs(a.out, exist_ok=True)
    DS = C.DEPTHS
    cal_rows, dhat_rows, test_rows, rerun_rows = [], [], [], []
    for m in ORDER:
        r = M[m]['T']
        assert r['complete'], m
        dj = rd(os.path.join(a.per_model, m, 'dhat_T.json'))
        bud = rd(os.path.join(a.per_model, m, 'budgets_T.json'))
        rr = rd(os.path.join(a.per_model, m, 'T_rerun.json'))
        assert dj['depths'] == DS and dj['included'] == r['cal']['included'], m
        assert dj['tau0.8']['fit'] == dj['tau0.9']['fit'], m
        for j, d in enumerate(DS):
            assert dj['acc'][j] == r['cal']['acc'][str(d)] and dj['n'][j] == r['cal']['n'][str(d)], (m, d)
            assert dj['tau0.8']['fit'][j] == r['cal']['tau0.8']['fit'][j], (m, d)
            cal_rows.append([m, d, dj['n'][j], dj['acc'][j], dj['tau0.8']['fit'][j]])
        row = [m, MODELS[m][0], FAMILY[m], BIG_B[m]]
        for tau in TAUS:
            c = dj[f'tau{tau}']
            assert c['dhat'] == r['cal'][f'tau{tau}']['dhat'] and c['D'] == bud[tau]['D'] == r['cal'][f'tau{tau}']['D']
            assert c['ci95'] == r['cal'][f'tau{tau}']['ci95'], m
            row += [c['dhat'], c['ci95'][0], c['ci95'][1], c['D']]
        row += [r['dhat_test_0.8']['dhat'], r['dhat_test_0.9']['dhat'], dj['included']]
        dhat_rows.append(row)
        for key, cond, tau in CONDS:
            D = {'I': None, 'E': None, 'A': 1}.get(cond) if cond != 'H' else bud[tau]['D']
            cv, co = r['curves'], r['costs'][key]
            for d in DS:
                s = str(d)
                test_rows.append([m, cond, tau, D, d, r['n_test'][key][s], cv[key][s], cv[key + '_strict'][s],
                                  cv[key + '_allcalls'][s], co['calls'][s], co['tokens'][s], co['prompt_tokens'][s]])
        assert rr['model'] == m
        rerun_rows.append([m, rr['n'], rr['acc_stored'], rr['acc_rerun'], rr['abs_diff'], rr['identical_outputs'],
                           rr['pass']])
    write(os.path.join(a.out, 'calibration_by_depth.csv'), ['model', 'depth', 'n_items', 'accuracy', 'fit'], cal_rows)
    write(os.path.join(a.out, 'dhat.csv'),
          ['model', 'hub_id', 'family', 'parameters_b'] +
          [f'{k}_{tau}' for tau in TAUS for k in ('dhat', 'dhat_ci95_low', 'dhat_ci95_high', 'levels_per_call')] +
          ['dhat_test_0.8', 'dhat_test_0.9', 'cal_accuracy_d1_d2_ge_0.8'], dhat_rows)
    write(os.path.join(a.out, 'test_by_depth.csv'),
          ['model', 'condition', 'threshold', 'levels_per_call', 'depth', 'n_items', 'accuracy',
           'accuracy_exact_output', 'all_calls_correct', 'calls_per_item', 'tokens_per_item',
           'prompt_tokens_per_item'], test_rows)
    write(os.path.join(a.out, 'rerun.csv'),
          ['model', 'n_items', 'accuracy_stored', 'accuracy_rerun', 'abs_difference', 'identical_outputs',
           'difference_le_0.03'], rerun_rows)


if __name__ == '__main__':
    main()
