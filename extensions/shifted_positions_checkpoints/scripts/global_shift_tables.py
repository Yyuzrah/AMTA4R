"""Result tables for the shift of all position indices on the two checkpoints (CPU, a few seconds).

Without --raw: reads teacher_forced_per_item.csv and free_running_per_problem.csv from --out (default: ../results)
and writes teacher_forced.csv, free_running.csv, free_running_unedited.csv and tables.md there.
With --raw DIR: first rewrites the two per-item and per-problem files from the outputs of tf_run.py and fr_run.py in
DIR (tf_<model>.jsonl, fr_<model>_base.jsonl, fr_<model>_strict*.jsonl), then writes the tables.
Only the conditions that shift all position indices are read: A16, A64 and Akc (all indices shifted by 16, 64 and k_c)
and, for free running, the unshifted resumption base_resume that they are compared with.
Teacher-forced rates are pooled over routing tokens (sums over items divided by the number of routing tokens).
Free-running success and its change are means over the problems that have a site (strict rule)."""
import argparse, csv, glob, json, os
import numpy as np

MODELS = [('f1', 'small_deep_mixed_rope', 'RoPE'), ('f2', 'large_deep_bfs_ape', 'APE')]
SHIFTS = [('A16', '16'), ('A64', '64'), ('Akc', 'k_c')]
FR_CONDS = [('base_resume', '0')] + SHIFTS
TYPES = ['all', 'bfs', 'dfs']
TF_ITEM_COLS = ['model', 'eval_set', 'trace_index', 'site_type', 'k_c', 'routing_tokens', 'base_correct'] + \
    [f'{c}_{x}' for c, _ in SHIFTS for x in ('flips', 'correct', 'max_abs_dlogit')]
FR_PROB_COLS = ['model', 'problem', 'unedited_success', 'site', 'site_type', 'k_c', 'budget', 'resume_token_identical'] + \
    [f'{c}_success' for c, _ in FR_CONDS]


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def write_csv(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in rows])
    print('wrote', path, len(rows), 'rows')


def from_raw(raw, out):
    tf_rows, fr_rows = [], []
    for m, _, _ in MODELS:
        for r in read_jsonl(f'{raw}/tf_{m}.jsonl'):
            row = dict(model=m, eval_set=r['set'], trace_index=r['idx'], site_type=r['type'], k_c=r['kc'],
                       routing_tokens=len(r['base_ok']), base_correct=sum(r['base_ok']))
            for c, _ in SHIFTS:
                row[f'{c}_flips'] = sum(r['cond'][c]['flip'])
                row[f'{c}_correct'] = sum(r['cond'][c]['ok'])
                row[f'{c}_max_abs_dlogit'] = r['cond'][c]['maxdlogit']
            tf_rows.append(row)
        base = {r['pid']: r for r in read_jsonl(f'{raw}/fr_{m}_base.jsonl')}
        by = {}
        for p in sorted(glob.glob(f'{raw}/fr_{m}_strict*.jsonl')):
            for r in read_jsonl(p):
                by.setdefault(r['pid'], {})[r['cond']] = r
        n_incomplete = 0
        for pid, b in base.items():
            row = {c: '' for c in FR_PROB_COLS}
            row.update(model=m, problem=pid, unedited_success=b['ok'], site=0)
            d = by.get(pid)
            if d is not None and all(c in d for c, _ in FR_CONDS):
                row.update(site=1, site_type=d['base_resume']['type'], k_c=d['base_resume']['kc'], budget=d['base_resume']['G'],
                           resume_token_identical=d['base_resume']['identical'])
                for c, _ in FR_CONDS:
                    row[f'{c}_success'] = d[c]['ok']
            elif d is not None:
                n_incomplete += 1
            fr_rows.append(row)
        if n_incomplete:
            print(m, n_incomplete, 'problems with a site lack a global-shift condition and are left out')
    write_csv(f'{out}/teacher_forced_per_item.csv', TF_ITEM_COLS, tf_rows)
    write_csv(f'{out}/free_running_per_problem.csv', FR_PROB_COLS, fr_rows)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def tables(out):
    tf_items = read_csv(f'{out}/teacher_forced_per_item.csv')
    fr_probs = read_csv(f'{out}/free_running_per_problem.csv')
    tf, fr, un = [], [], []
    for m, ck, pe in MODELS:
        for ty in TYPES:
            rr = [r for r in tf_items if r['model'] == m and (ty == 'all' or r['site_type'] == ty)]
            if not rr:
                continue
            den = sum(int(r['routing_tokens']) for r in rr)
            base_ok = sum(int(r['base_correct']) for r in rr)
            kc_med = float(np.median([int(r['k_c']) for r in rr]))
            for c, k in SHIFTS:
                fl = sum(int(r[f'{c}_flips']) for r in rr)
                ok = sum(int(r[f'{c}_correct']) for r in rr)
                tf.append(dict(model=m, checkpoint=ck, encoding=pe, site_type=ty, condition=c, shift=k, items=len(rr),
                               routing_tokens=den, k_c_median=kc_med, base_accuracy=base_ok / den, flips=fl,
                               flip_rate=fl / den, accuracy=ok / den, accuracy_change=(ok - base_ok) / den,
                               max_abs_dlogit=max(float(r[f'{c}_max_abs_dlogit']) for r in rr)))
        mp = [r for r in fr_probs if r['model'] == m]
        if not mp:
            continue
        sites = [r for r in mp if r['site'] == '1']
        un.append(dict(model=m, checkpoint=ck, encoding=pe, problems=len(mp),
                       unedited_success=float(np.mean([float(r['unedited_success']) for r in mp])), problems_with_site=len(sites)))
        for ty in TYPES:
            P = [r for r in sites if ty == 'all' or r['site_type'] == ty]
            if not P:
                continue
            Y = {c: np.array([float(r[f'{c}_success']) for r in P]) for c, _ in FR_CONDS}
            b0 = Y['base_resume']
            for c, k in FR_CONDS:
                fr.append(dict(model=m, checkpoint=ck, encoding=pe, site_type=ty, condition=c, shift=k, problems=len(P),
                               k_c_median=float(np.median([int(r['k_c']) for r in P])),
                               unedited_success=float(np.mean([float(r['unedited_success']) for r in P])),
                               resume_token_identical=float(np.mean([float(r['resume_token_identical']) for r in P])),
                               success=float(Y[c].mean()), change=float((Y[c] - b0).mean()),
                               lost=int(((b0 == 1) & (Y[c] == 0)).sum()), gained=int(((b0 == 0) & (Y[c] == 1)).sum())))
    write_csv(f'{out}/teacher_forced.csv', list(tf[0]), tf)
    write_csv(f'{out}/free_running.csv', list(fr[0]), fr)
    write_csv(f'{out}/free_running_unedited.csv', list(un[0]), un)

    md = ['# Shift of all position indices on the two checkpoints', '',
          'Generated by `scripts/global_shift_tables.py` from the per-item and per-problem files in this folder. '
          'Rates are rounded here; the CSV files give them at full precision. Columns are explained in README.md.', '',
          '## Teacher-forced routing predictions', '',
          '| model | site type | condition | items | routing tokens | flips | flip rate | base accuracy | accuracy change | max abs logit change |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in tf:
        md.append(f"| {r['model']} ({r['encoding']}) | {r['site_type']} | {r['condition']} | {r['items']} | {r['routing_tokens']} | "
                  f"{r['flips']} | {r['flip_rate']:.4f} | {r['base_accuracy']:.4f} | {r['accuracy_change']:+.4f} | {r['max_abs_dlogit']:.2e} |")
    md += ['', '## Free-running generation resumed with shifted position indices (strict site rule)', '',
           '| model | problems | unedited success | problems with a site |', '|---|---|---|---|']
    for r in un:
        md.append(f"| {r['model']} ({r['encoding']}) | {r['problems']} | {r['unedited_success']:.4f} | {r['problems_with_site']} |")
    md += ['', '| model | site type | condition | problems | success | change | lost | gained | base_resume token-identical |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in fr:
        md.append(f"| {r['model']} ({r['encoding']}) | {r['site_type']} | {r['condition']} | {r['problems']} | {r['success']:.4f} | "
                  f"{r['change']:+.4f} | {r['lost']} | {r['gained']} | {r['resume_token_identical']:.4f} |")
    open(f'{out}/tables.md', 'w').write('\n'.join(md) + '\n')
    print('wrote', f'{out}/tables.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=None, help='directory with the outputs of tf_run.py and fr_run.py')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
    a = ap.parse_args()
    if a.raw:
        from_raw(a.raw, a.out)
    tables(a.out)
