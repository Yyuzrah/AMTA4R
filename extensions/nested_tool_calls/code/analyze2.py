"""Analysis of the nested tool-call runs.
  python analyze2.py gate     after calibration: 'included' flag per model, gate status, analysis/d_star.json (cal only)
  python analyze2.py metrics  all metrics -> analysis/metrics.json (does not read the checks of verify2.py)
  python analyze2.py final    reads analysis/verify.json (verify2.py) and the re-run files, evaluates the checks
                              (a), (b), (c), noharm and the gate -> analysis/summary.json ("status"), RESULTS.md,
                              d_star.json, curves.png
Reads results/<model>/ under config.BASE (smoke/ in quick test mode)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from stats import dhat, dhat_boot
from models import MODELS, ORDER, FAMILY

BIG_B = {'qwen2.5-32b': 32, 'olmo2-32b': 32, 'qwen3-32b': 32, 'mistral-small-24b': 24, 'qwen2.5-14b': 14,
         'qwen3-14b': 14, 'phi4-14b': 14, 'olmo2-13b': 13, 'qwen3-8b': 8, 'qwen2.5-7b': 7, 'olmo2-7b': 7}
DS = C.DEPTHS


def load(p):
    with open(p) as f:
        L = [json.loads(l) for l in f]
    return L[0]['_meta'], L[1:]


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p + '.tmp', 'w'), indent=1, default=str)
    os.replace(p + '.tmp', p)


def by_depth(recs, f):
    out = {d: [] for d in DS}
    for r in recs:
        out[r['d']].append(f(r))
    return out


def mean(v):
    return float(np.mean(v)) if len(v) else None


def cal_block(m, task):
    p = f'{C.RES}/{m}/{task}_I_cal.jsonl'
    if not os.path.exists(p):
        return None
    recs = load(p)[1]
    byc = by_depth(recs, lambda r: 1.0 if r['ok'] else 0.0)
    acc = [mean(byc[d]) for d in DS]
    n = [len(byc[d]) for d in DS]
    res = {'acc': dict(zip(DS, acc)), 'n': dict(zip(DS, n))}
    for tau in C.TAUS:
        dh, fit = dhat(DS, acc, n, tau)
        lo, hi, _ = dhat_boot(DS, [byc[d] for d in DS], tau, reps=C.BOOT)
        res[f'tau{tau}'] = {'dhat': int(dh), 'D': max(1, int(dh)), 'ci95': [lo, hi], 'fit': [float(x) for x in fit],
                            'censored': bool(dh >= DS[-1])}
    res['included'] = bool(all(res['acc'][d] >= C.TAU for d in C.INCL_DEPTHS))
    return res


def boot_diff(a_by, b_by, Bs, reps, seed=0):
    """Paired percentile bootstrap (items resampled within depth) of mean over depths in Bs of mean(a - b)."""
    if not Bs:
        return None
    rng = np.random.default_rng(seed)
    diffs = {d: np.asarray(a_by[d], float) - np.asarray(b_by[d], float) for d in Bs}
    vals = []
    for _ in range(reps):
        vals.append(np.mean([diffs[d][rng.integers(0, len(diffs[d]), len(diffs[d]))].mean() for d in Bs]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def model_task(m, task):
    md = f'{C.RES}/{m}'
    cal = cal_block(m, task)
    files = {'I': f'{md}/{task}_I_test.jsonl', 'E': f'{md}/{task}_E_test.jsonl', 'A': f'{md}/{task}_A_test.jsonl',
             'bud': f'{md}/budgets_{task}.json'}
    if cal is None or not all(os.path.exists(p) for p in files.values()):
        return {'complete': False, 'cal': cal, 'missing': [k for k, p in files.items() if not os.path.exists(p)]}
    bud = json.load(open(files['bud']))
    R = {k: load(files[k])[1] for k in ('I', 'E', 'A')}
    for tau in C.TAUS:
        b = bud[str(tau)]
        assert b['D'] == cal[f'tau{tau}']['D'], f'{m} {task}: budget file D != recomputed calibration D'
        R[f'H{tau}'] = load(f"{md}/{b['file']}")[1]
    res = {'complete': True, 'cal': cal, 'budgets': bud, 'curves': {}, 'costs': {}, 'n_test': {}}
    keyed = {}
    for k, recs in R.items():
        keys = sorted((r['d'], r['idx']) for r in recs)
        assert len(keys) == len(set(keys)), f'duplicate items {m} {task} {k}'
        keyed[k] = {(r['d'], r['idx']): r for r in recs}
        acc = by_depth(recs, lambda r: 1.0 if r['ok'] else 0.0)
        res['curves'][k] = {d: mean(acc[d]) for d in DS}
        res['curves'][k + '_strict'] = {d: mean(v) for d, v in by_depth(recs, lambda r: 1.0 if r['ok_strict'] else 0.0).items()}
        res['curves'][k + '_allcalls'] = {d: mean(v) for d, v in by_depth(recs, lambda r: 1.0 if r['all_ok'] else 0.0).items()}
        res['costs'][k] = {'calls': {d: mean(v) for d, v in by_depth(recs, lambda r: r['n_calls']).items()},
                           'tokens': {d: mean(v) for d, v in by_depth(recs, lambda r: r['ptok'] + r['otok']).items()},
                           'prompt_tokens': {d: mean(v) for d, v in by_depth(recs, lambda r: r['ptok']).items()}}
        res['n_test'][k] = {d: len(v) for d, v in acc.items()}
        pc = {}
        for r in recs:
            for c in r['calls']:
                pc.setdefault(c['nest'], []).append(1.0 if c['ok'] else 0.0)
        res['costs'][k]['percall_acc_by_nest'] = {int(k2): [mean(v), len(v)] for k2, v in sorted(pc.items())}
    # same items in every condition
    ref = set(keyed['I'])
    for k in keyed:
        assert set(keyed[k]) == ref, f'{m} {task}: item sets differ between I and {k}'
    # test-split d_hat (descriptive)
    bt = by_depth(R['I'], lambda r: 1.0 if r['ok'] else 0.0)
    acct = [mean(bt[d]) for d in DS]
    for tau in C.TAUS:
        dh, fit = dhat(DS, acct, [len(bt[d]) for d in DS], tau)
        res[f'dhat_test_{tau}'] = {'dhat': int(dh), 'fit': [float(x) for x in fit]}
    okd = {k: by_depth(R[k], lambda r: 1.0 if r['ok'] else 0.0) for k in R}
    # align items for paired bootstrap
    okd_al = {k: {d: [1.0 if keyed[k][(d, i)]['ok'] else 0.0 for (dd, i) in sorted(ref) if dd == d] for d in DS} for k in R}
    for tau in C.TAUS:
        hk = f'H{tau}'
        dh = cal[f'tau{tau}']['dhat']
        Bs = [d for d in DS if d > dh]
        s = {'tau': tau, 'dhat_cal': dh, 'D_H': cal[f'tau{tau}']['D'], 'B': Bs}
        for k in ('I', 'E', 'A', hk):
            nm = 'H' if k == hk else k
            s[f'acc_{nm}'] = mean([res['curves'][k][d] for d in Bs]) if Bs else None
            s[f'calls_{nm}'] = mean([res['costs'][k]['calls'][d] for d in Bs]) if Bs else None
            s[f'tokens_{nm}'] = mean([res['costs'][k]['tokens'][d] for d in Bs]) if Bs else None
            s[f'full_acc_{nm}'] = mean([res['curves'][k][d] for d in DS])
            s[f'full_calls_{nm}'] = mean([res['costs'][k]['calls'][d] for d in DS])
            s[f'full_tokens_{nm}'] = mean([res['costs'][k]['tokens'][d] for d in DS])
        if Bs:
            s['H_minus_A'] = s['acc_H'] - s['acc_A']
            s['H_minus_I'] = s['acc_H'] - s['acc_I']
            s['ci95_H_minus_I'] = boot_diff(okd_al[hk], okd_al['I'], Bs, C.BOOT, seed=1)
            s['ci95_H_minus_A'] = boot_diff(okd_al[hk], okd_al['A'], Bs, C.BOOT, seed=2)
            s['b_acc'] = bool(s['acc_H'] >= s['acc_A'] - C.TOL_A)
            s['b'] = bool(s['b_acc'] and s['calls_H'] < s['calls_A'])
            s['c'] = bool(s['H_minus_I'] >= C.MARGIN_I and s['ci95_H_minus_I'][0] > 0)
            s['call_ratio_H_over_A'] = s['calls_H'] / s['calls_A']
            s['token_ratio_H_over_A'] = s['tokens_H'] / s['tokens_A']
        else:
            s['b'] = s['c'] = s['b_acc'] = False
        s['noharm'] = bool(s['full_acc_H'] >= s['full_acc_A'] - C.NOHARM)
        s['a'] = bool(res[f'dhat_test_{C.TAU}']['fit'][-1] < C.TAU and dh < DS[-1]) if tau == C.TAU else None
        res[f'summary_{tau}'] = s
    res['a'] = res[f'summary_{C.TAU}']['a']
    return res


def decide(R, task, verify=None, reruns=None):
    have = [m for m in ORDER if R[m].get(task, {}).get('complete')]
    inc = [m for m in have if R[m][task]['cal']['included']]
    fams = sorted({FAMILY[m] for m in inc})
    gate = bool(len(inc) >= C.GATE_MIN_MODELS and len(fams) >= C.GATE_MIN_FAMILIES and
                any(BIG_B[m] >= C.GATE_BIG for m in inc))
    per = {}
    for m in inc:
        r = R[m][task]
        s, s2 = r[f'summary_{C.TAU}'], r[f'summary_{C.TAU_SENS}']
        per[m] = {'a': r['a'], 'b': s['b'], 'c': s['c'], 'b_acc_sens': s2['b_acc'], 'c_sens': s2['c'],
                  'noharm': s['noharm'],
                  'D_H': s['D_H'], 'dhat': s['dhat_cal'], 'acc_H': s.get('acc_H'), 'acc_A': s.get('acc_A'),
                  'acc_I': s.get('acc_I'), 'calls_H': s.get('calls_H'), 'calls_A': s.get('calls_A')}
    if len(have) < C.GATE_MIN_MODELS:
        out = 'too_few_models'
    elif not gate:
        out = 'not_all_checks_hold'
    elif all(p['a'] and p['b'] and p['c'] for p in per.values()):
        out = 'abc_hold'
    else:
        withA = [p for p in per.values() if p['a']]
        nfail = sum(1 for p in withA if not (p['b'] and p['c']))
        out = ('abc_fail_for_half_or_more' if (len(per) - len(withA) >= len(per) / 2 or
                                              (withA and nfail >= len(withA) / 2)) else 'not_all_checks_hold')
    checks = {}
    if out == 'abc_hold':
        checks['tau_sens'] = all(p['b_acc_sens'] and p['c_sens'] for p in per.values())
        checks['noharm'] = all(p['noharm'] for p in per.values())
        checks['verify_pass'] = bool(verify and verify.get('pass'))
        checks['rerun_pass'] = bool(reruns is not None and all(reruns.get(m, {}).get(task, {}).get('pass') for m in inc))
    final = ('all_checks_hold' if out == 'abc_hold' and all(checks.values()) else
             'abc_hold_other_checks_fail' if out == 'abc_hold' else out)
    return {'models_with_results': have, 'included': inc, 'excluded': [m for m in have if m not in inc],
            'families_included': fams, 'gate': gate, 'per_model': per, 'status_before_checks': out,
            'verification_checks': checks, 'status_task': final}


def gate_stage():
    """After calibration only: 'included' flags, gate status and d_star.json (no test data is read)."""
    lic = json.load(open(f'{C.EXP}/meta/licenses.json'))
    out, models = {}, {}
    for m in ORDER:
        e = {'hub': MODELS[m][0], 'num_hidden_layers': lic[m]['num_hidden_layers'], 'hidden_size': lic[m]['hidden_size'],
             'params_b': BIG_B[m], 'license': lic[m]['license_tag'], 'sha': lic[m]['sha']}
        for t in C.TASKS:
            cb = cal_block(m, t)
            if cb is None:
                continue
            e[t] = {'included': cb['included'], 'cal_acc': {str(d): cb['acc'][d] for d in DS}, 'n_cal': cb['n'][DS[0]]}
            for tau in C.TAUS:
                e[t][f'dhat_tau{tau}'] = cb[f'tau{tau}']['dhat']
                e[t][f'ci95_tau{tau}'] = cb[f'tau{tau}']['ci95']
                e[t][f'D_H_tau{tau}'] = cb[f'tau{tau}']['D']
                e[t][f'censored_tau{tau}'] = cb[f'tau{tau}']['censored']
        models[m] = e
    for t in C.TASKS:
        inc = [m for m in ORDER if models[m].get(t, {}).get('included')]
        fams = sorted({FAMILY[m] for m in inc})
        out[t] = {'calibrated': [m for m in ORDER if t in models[m]], 'included': inc, 'families': fams,
                  'gate': bool(len(inc) >= C.GATE_MIN_MODELS and len(fams) >= C.GATE_MIN_FAMILIES and
                               any(BIG_B[m] >= C.GATE_BIG for m in inc))}
    note = ('d_hat*(tau) = largest depth of the grid %s whose fit (weighted least squares, not increasing with '
            'depth) of single-call (answer only) accuracy on the calibration split is >= tau; 0 if none; 95%% '
            'percentile bootstrap CI (items resampled within depth). D_H = max(1, d_hat*) is the number of levels '
            'per call of condition H: chains deeper than D_H are cut into calls of <= D_H levels. Main tau %s, '
            'second tau %s. included = calibration accuracy at d = 1 and d = 2 both >= %s. '
            'Task: T = nested tool-call returns (a Python tool registry). Format: the d chain tools + %d distractor tools + (24 - d) note lines, '
            'callee-first order, 4 fixed demonstrations, answer-only after an "Answer:" prefill, greedy, bf16, '
            'vLLM 0.31.0; in a split, the answer of the previous call replaces the definition of the tool it '
            'computed.' % (DS, C.TAU, C.TAU_SENS, C.TAU, C.N_TOOLS_EXTRA))
    ds = {'_note': note, 'experiment': 'nested_tool_calls', 'stage': 'calibration', 'smoke': C.SMOKE,
          'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tau_primary': C.TAU, 'tau_sens': C.TAU_SENS,
          'depths': DS, 'gate': out, 'models': models}
    jdump(ds, f'{C.ANA}/d_star.json')
    jdump({'generated': ds['generated'], 'gate': out}, f'{C.ANA}/gate.json')
    print(json.dumps(out, indent=1))


def metrics_stage():
    R = {m: {} for m in ORDER}
    for m in ORDER:
        for t in C.TASKS:
            R[m][t] = model_task(m, t)
    jdump({'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'smoke': C.SMOKE, 'results': R}, f'{C.ANA}/metrics.json')
    for t in C.TASKS:
        d = decide(R, t)
        print(t, json.dumps({k: d[k] for k in ('included', 'gate', 'status_before_checks')}))


def final_stage():
    M = json.load(open(f'{C.ANA}/metrics.json'))
    R = {m: {t: _intkeys(v) for t, v in M['results'][m].items()} for m in ORDER}
    verify = json.load(open(f'{C.ANA}/verify.json')) if os.path.exists(f'{C.ANA}/verify.json') else None
    reruns = {}
    for m in ORDER:
        for t in C.TASKS:
            p = f'{C.RES}/{m}/{t}_rerun.json'
            if os.path.exists(p):
                reruns.setdefault(m, {})[t] = json.load(open(p))
    dec = {t: decide(R, t, verify, reruns) for t in C.TASKS}
    status = dec[C.PRIMARY]['status_task']
    if status == 'abc_hold_other_checks_fail':
        status = 'not_all_checks_hold'         # (a)-(c) hold, but a check of verify2.py or a re-run failed
    summary = {'experiment': 'nested_tool_calls', 'status': status, 'primary_task': C.PRIMARY,
               'status_primary_detail': dec[C.PRIMARY]['status_task'],
               'status_secondary': {t: dec[t]['status_task'] for t in C.TASKS if t != C.PRIMARY}, 'per_task': dec,
               'verify_pass': bool(verify and verify.get('pass')), 'reruns': reruns,
               'design': {'depths': DS, 'M': C.M_FUN, 'order': C.ORDER, 'n_cal': C.N_CAL, 'n_test': C.N_TEST,
                          'tau': C.TAU, 'tau_sens': C.TAU_SENS, 'tol_A': C.TOL_A, 'margin_I': C.MARGIN_I,
                          'noharm': C.NOHARM, 'gate': [C.GATE_MIN_MODELS, C.GATE_MIN_FAMILIES, C.GATE_BIG]},
               'smoke': C.SMOKE, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'results': R}
    write_dstar(R)
    write_md(R, dec, summary, verify, reruns)
    plot(R)
    jdump(summary, f'{C.ANA}/summary.json')       # written last
    print('STATUS', status, json.dumps({t: dec[t]['status_task'] for t in C.TASKS}))


def _intkeys(o):
    if isinstance(o, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k): _intkeys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_intkeys(v) for v in o]
    return o


def write_dstar(R):
    p = f'{C.ANA}/d_star.json'
    ds = json.load(open(p)) if os.path.exists(p) else {'models': {}}
    for m in ORDER:
        for t in C.TASKS:
            r = R[m].get(t) or {}
            if r.get('complete') and t in ds['models'].get(m, {}):
                for tau in C.TAUS:
                    ds['models'][m][t][f'dhat_test_tau{tau}'] = r[f'dhat_test_{tau}']['dhat']
                    assert ds['models'][m][t][f'dhat_tau{tau}'] == r['cal'][f'tau{tau}']['dhat'], 'd_star mismatch'
    ds['stage'] = 'final'
    ds['generated_final'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    jdump(ds, p)


def f2(v):
    return '-' if v is None else f'{v:.2f}'


def write_md(R, dec, summary, verify, reruns):
    L = ['# Nested tool calls: results (generated by code/analyze2.py; do not edit)', '',
         f"**Status of the checks: {summary['status']}** (primary task T NestTool; detail {summary['status_primary_detail']}; "
         f"secondary: {summary['status_secondary'] or 'none'}). Quick test mode: {C.SMOKE}. Generated {summary['generated']}.", '',
         f"Independent verification (verify2.py): pass = {summary['verify_pass']}"
         + (f"; failed checks: {verify.get('failed')}" if verify else ' (verify.json missing)'), '']
    for t in C.TASKS:
        d = dec[t]
        L += [f"## Task {t} ({'primary' if t == C.PRIMARY else 'secondary'})", '',
              f"models with results {len(d['models_with_results'])}; included {d['included']}; excluded "
              f"{d['excluded']}; families {d['families_included']}; gate {d['gate']}; status before the checks "
              f"**{d['status_before_checks']}**; verification checks {d['verification_checks']}; task status "
              f"**{d['status_task']}**", '',
              f'| model | incl | cal acc d1/d2 | dhat{C.TAU} [CI] | dhat{C.TAU_SENS} | D_H | dhat_test | (a) | (b) | (c) | '
              f'b_acc@{C.TAU_SENS} | c@{C.TAU_SENS} | noharm |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for m in ORDER:
            r = R[m].get(t)
            if not r or not r.get('complete'):
                L.append(f'| {m} | incomplete: {r.get("missing") if r else "-"} |')
                continue
            c, s, s2 = r['cal'], r[f'summary_{C.TAU}'], r[f'summary_{C.TAU_SENS}']
            L.append(f"| {m} | {c['included']} | {f2(c['acc'][1])}/{f2(c['acc'][2])} | {c[f'tau{C.TAU}']['dhat']} "
                     f"{c[f'tau{C.TAU}']['ci95']} | {c[f'tau{C.TAU_SENS}']['dhat']} | {s['D_H']} | "
                     f"{r[f'dhat_test_{C.TAU}']['dhat']} | {r['a']} | {s['b']} | {s['c']} | {s2.get('b_acc')} | {s2['c']} | {s['noharm']} |")
        L += ['', 'Accuracy by depth (test split; end-to-end = final answer correct). Depths ' + str(DS), '',
              '| model | cond | ' + ' | '.join(str(x) for x in DS) + ' |', '|---|---|' + '---|' * len(DS)]
        for m in ORDER:
            r = R[m].get(t)
            if not r or not r.get('complete'):
                continue
            for k in ['I', 'E', 'A', f'H{C.TAU}', f'H{C.TAU_SENS}']:
                L.append(f'| {m} | {k} | ' + ' | '.join(f2(r['curves'][k][dd]) for dd in DS) + ' |')
            L.append(f'| {m} | calls H{C.TAU} | ' + ' | '.join(f2(r['costs'][f"H{C.TAU}"]['calls'][dd]) for dd in DS) + ' |')
        L += ['', f'Deep-set summaries (B_m = depths > d_hat*_cal(tau)); costs = mean calls / tokens per task:', '']
        for m in ORDER:
            r = R[m].get(t)
            if r and r.get('complete'):
                for tau in C.TAUS:
                    s = r[f'summary_{tau}']
                    keep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                    L.append(f'- {m} tau {tau}: `{json.dumps(keep)}`')
        L.append('')
    L += ['## Re-run reproducibility (verification iv)', '']
    for m, v in reruns.items():
        for t, x in v.items():
            L.append(f"- {m} {t}: n {x['n']}, stored {x['acc_stored']:.3f}, re-run {x['acc_rerun']:.3f}, "
                     f"identical outputs {x['identical_outputs']:.3f}, pass {x['pass']}")
    open(f'{C.ANA}/RESULTS.md', 'w').write('\n'.join(L) + '\n')


def plot(R):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    ms = [m for m in ORDER if any((R[m].get(t) or {}).get('complete') for t in C.TASKS)]
    if not ms:
        return
    fig, axes = plt.subplots(len(C.TASKS), len(ms), figsize=(2.6 * len(ms), 2.4 * len(C.TASKS)), squeeze=False)
    for i, t in enumerate(C.TASKS):
        for j, m in enumerate(ms):
            ax = axes[i][j]
            r = R[m].get(t)
            if r and r.get('complete'):
                for k, st in (('I', 'k-'), ('E', 'b--'), ('A', 'g-'), (f'H{C.TAU}', 'r-'), (f'H{C.TAU_SENS}', 'm:')):
                    ax.plot(DS, [r['curves'][k][d] for d in DS], st, label=k, lw=1)
                ax.axhline(C.TAU, color='gray', lw=0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f'{m} {t}', fontsize=7)
            ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(f'{C.ANA}/curves.png', dpi=130)


if __name__ == '__main__':
    {'gate': gate_stage, 'metrics': metrics_stage, 'final': final_stage}[sys.argv[1]]()
