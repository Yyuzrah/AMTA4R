"""Analysis of the serving experiment. Recomputes everything from the unit outputs
(results/raw|rescue/<model>/<scenario>/{checks.json,<task>.jsonl,DONE.json}).
Writes analysis/summary.json, analysis/RESULTS.md, analysis/verify.json (consistency checks of the records and of the
bootstrap), analysis/cards/<model>.json and .md (one card per model), analysis/cards_table.md,
analysis/table_cells.csv and analysis/table_pooled.csv. export_results.py copies the tables and cards to results/.
Usage: analyze.py <base> [split]"""
import csv, json, os, sys, time, zlib
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import (MODELS, ORDER, CLASSES, SCENARIOS, FP32_SCEN, TASKS, FEWSHOT_TASKS, KS, K_PRIMARY, WINDOW, EXCLUDED,
                      units, scen_tasks, FEWSHOT_DIR, PARAMS_B)
from common_io import unit_dirs, rescue_set, REP_MAX, V4_MIN

B = 10000                       # bootstrap resamples (percentile 95% CI)
B_VERIFY = 4000                 # independent re-implementation (index resampling) in the self-check
G1_REL = 1e-3                   # max|default HF forward - explicit position path| / max|logit| (fp32)
G2_MIN, G2_NOISE = 1e-2, 10.0   # gap control changes the logits by >= 1e-2 (absolute) and >= 10 x the default-vs-explicit difference
G2B_MIN, G2B_FRAC = 1e-3, 0.99  # APE: k=256 changes the first-step logits by > 1e-3 on >= 99% of items
W1_REL = 1e-3                   # window check: band wider than the sequence == full attention (relative)
W2_MIN = 1e-2                   # window check: a 128-token band changes the logits (absolute)
PATH_REL = 1e-3                 # leftpad / cache reference row == plain single pass (no pads, one phase), relative
G4_POOLED_MIN, G4_TASK_MIN, G4_MIN_TASKS = 0.10, 0.05, 2   # lower bounds on the unshifted accuracy only
WIN_MIN_ACC = 0.05              # smallest pooled reference accuracy under the sliding window
MATERIAL = 0.02                 # |Delta| >= 0.02 with a CI that excludes 0 is flagged as 'material'
MIN_ADMIT = {'APE': (3, 0.6), 'ALiBi': (3, 0.6), 'RoPE': (3, 0.6)}   # per class: >= n models and >= frac admitted
HALF_ADMIT = 0.5                # bf16 summary only for models with bf16 pooled ref acc >= 0.5 x fp32 pooled ref acc
FEWSHOT_RUN_MODELS = ('gpt2', 'gpt2-medium', 'gpt-neo-125m', 'gpt-neo-1.3B', 'pythia-160m', 'pythia-410m', 'pythia-1.4b',
             'smollm2-360m')
FEWSHOT_RUN_ROWS = {'ref': 0, 'k16': 2, 'k64': 3, 'k256': 4}


def seed(*a):
    return zlib.crc32('|'.join(map(str, a)).encode())


def r4(x):
    return None if x is None else float(round(x, 4))


# ------------------------------------------------------------------ bootstrap
def boot_ci(dlist, s):
    """Paired percentile bootstrap of the mean of d over all items, stratified by task. d in {-1, 0, 1}, so each
    stratum's resampled sum is (#+1) - (#-1) of a multinomial draw with the observed proportions (identical in
    distribution to resampling items with replacement)."""
    n_tot = sum(len(d) for d in dlist)
    if all((d == 0).all() for d in dlist):
        return 0.0, 0.0
    rng = np.random.default_rng(s)
    tot = np.zeros(B)
    for d in dlist:
        n = len(d)
        p = np.array([(d == -1).sum(), (d == 0).sum(), (d == 1).sum()], dtype=float) / n
        draw = rng.multinomial(n, p, size=B)
        tot += draw[:, 2] - draw[:, 0]
    lo, hi = np.percentile(tot / n_tot, [2.5, 97.5])
    return float(lo), float(hi)


def boot_ci_index(dlist, s, nb=B_VERIFY):
    """Independent implementation for the self-check: plain item-index resampling within each task."""
    rng = np.random.default_rng(s + 7919)
    n_tot = sum(len(d) for d in dlist)
    tot = np.zeros(nb)
    for d in dlist:
        idx = rng.integers(0, len(d), size=(nb, len(d)))
        tot += d[idx].sum(1)
    lo, hi = np.percentile(tot / n_tot, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ loading
def load_records(d, scen):
    out = {}
    for t in scen_tasks(scen):
        p = f'{d}/{t}.jsonl'
        if not os.path.exists(p):
            continue
        rr = []
        for line in open(p):
            r = json.loads(line)
            r.pop('gen', None)
            rr.append(r)
        out[t] = rr
    return out


def load_units(base):
    rs = rescue_set(base)
    U = {}
    for u, (uid, tag, scen) in enumerate(units()):
        raw, res = unit_dirs(base, u)
        eff, src = None, None
        if u in rs and os.path.exists(f'{res}/DONE.json'):
            eff, src = res, 'rescue'
        elif os.path.exists(f'{raw}/DONE.json'):
            eff, src = raw, 'raw'
        if eff is None:
            U[(tag, scen)] = {'u': u, 'status': 'missing', 'rescued': u in rs}
            continue
        U[(tag, scen)] = {'u': u, 'status': 'done', 'src': src, 'dir': eff, 'rescued': u in rs,
                          'done': json.load(open(f'{eff}/DONE.json')),
                          'checks': json.load(open(f'{eff}/checks.json')),
                          'recs': load_records(eff, scen)}
    return U, rs


# ------------------------------------------------------------------ metrics
def row_kind(label):
    for p in ('block', 'gap', 'k', 'p'):
        if label.startswith(p) and label[len(p):].isdigit():
            return p, int(label[len(p):])
    return label, None


def cell(recs, label, key):
    """Metrics of one row label vs 'ref' over a list of records (one task)."""
    rr = [r for r in recs if label in r['labels']]
    if not rr:
        return None
    i0 = [r['labels'].index('ref') for r in rr]
    ii = [r['labels'].index(label) for r in rr]
    c0 = np.array([r['correct'][a] for r, a in zip(rr, i0)])
    c1 = np.array([r['correct'][b] for r, b in zip(rr, ii)])
    d = c1 - c0
    lo, hi = boot_ci([d], seed(*key))
    md = np.array([r['first']['maxdiff'][b] for r, b in zip(rr, ii)])
    kl = np.array([r['first']['kl'][b] for r, b in zip(rr, ii)])
    return {'n': len(rr), 'acc_ref': float(c0.mean()), 'acc': float(c1.mean()), 'delta': float(d.mean()),
            'ci': [lo, hi], 'ci_contains_0': bool(lo <= 0 <= hi), 'ci_below_0': bool(hi < 0), 'ci_above_0': bool(lo > 0),
            'answer_flip_rate': float(np.mean([r['ans'][b] != r['ans'][a] for r, a, b in zip(rr, i0, ii)])),
            'n_correct_to_wrong': int((d < 0).sum()), 'n_wrong_to_correct': int((d > 0).sum()),
            'first_token_flip_rate': float(np.mean([r['first']['argmax'][b] != r['first']['argmax'][a]
                                                    for r, a, b in zip(rr, i0, ii)])),
            'logit_maxabs_median': float(np.median(md)), 'logit_maxabs_max': float(md.max()),
            'frac_logit_changed_gt_1e-3': float((md > 1e-3).mean()),
            'kl_median': float(np.median(kl)), 'kl_max': float(kl.max()), '_d': d}


def pooled(cells_by_task, key):
    tasks = [t for t, c in cells_by_task.items() if c is not None]
    if not tasks:
        return None
    ds = [cells_by_task[t]['_d'] for t in tasks]
    n = sum(len(d) for d in ds)
    lo, hi = boot_ci(ds, seed(*key))
    w = lambda f: float(sum(cells_by_task[t][f] * cells_by_task[t]['n'] for t in tasks) / n)
    return {'n': n, 'tasks': tasks, 'acc_ref': w('acc_ref'), 'acc': w('acc'), 'delta': float(np.concatenate(ds).mean()),
            'ci': [lo, hi], 'ci_contains_0': bool(lo <= 0 <= hi), 'ci_below_0': bool(hi < 0), 'ci_above_0': bool(lo > 0),
            'material': bool(not (lo <= 0 <= hi) and abs(float(np.concatenate(ds).mean())) >= MATERIAL),
            'answer_flip_rate': w('answer_flip_rate'), 'first_token_flip_rate': w('first_token_flip_rate'),
            'n_correct_to_wrong': int(sum(cells_by_task[t]['n_correct_to_wrong'] for t in tasks)),
            'n_wrong_to_correct': int(sum(cells_by_task[t]['n_wrong_to_correct'] for t in tasks)),
            'frac_logit_changed_gt_1e-3': w('frac_logit_changed_gt_1e-3'),
            'logit_maxabs_median_of_tasks': float(np.median([cells_by_task[t]['logit_maxabs_median'] for t in tasks])),
            'logit_maxabs_max': float(max(cells_by_task[t]['logit_maxabs_max'] for t in tasks)),
            'kl_median_of_tasks': float(np.median([cells_by_task[t]['kl_median'] for t in tasks])), '_ds': ds}


def unit_metrics(U, tag, scen):
    x = U.get((tag, scen))
    if not x or x['status'] != 'done':
        return None
    recs = x['recs']
    labels = []
    for t in recs:
        for r in recs[t]:
            for l in r['labels']:
                if l != 'ref' and l not in labels:
                    labels.append(l)
    cells = {l: {t: cell(recs[t], l, (tag, scen, t, l)) for t in recs} for l in labels}
    pool = {l: pooled(cells[l], (tag, scen, 'pooled', l)) for l in labels}
    acc_ref_task = {t: float(np.mean([r['correct'][r['labels'].index('ref')] for r in recs[t]])) for t in recs}
    n_ref = sum(len(v) for v in recs.values())
    acc_ref = float(sum(acc_ref_task[t] * len(recs[t]) for t in recs) / max(1, n_ref))
    return {'cells': cells, 'pooled': pool, 'acc_ref_task': acc_ref_task, 'acc_ref': acc_ref, 'labels': labels}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith('_')}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


# ------------------------------------------------------------------ gates
def unit_gate(U, M, tag, scen, cls):
    """Per-unit validity (fp32 scenarios): cached decoding agrees with full recomputation (V4 >= V4_MIN), replicate
    answer flips <= REP_MAX, finite logits, plus the reference-path check (leftpad, cache) or the window-mask check
    (window). offset_bf16: finite logits only; V4 and replicate flips are reported."""
    x = U.get((tag, scen))
    if not x or x['status'] != 'done':
        return False, ['no finished result'], {}
    ck = x['checks']['summary']
    why, val = [], {}
    v4 = ck.get('V4_cached_vs_full_agree')
    val['V4'] = v4
    rep = {t: c['answer_flip_rate'] for t, c in M[(tag, scen)]['cells'].get('rep', {}).items() if c}
    val['rep_flip_max'] = max(rep.values()) if rep else None
    fin = bool(ck.get('finite', False)) and all(r.get('finite', True) for v in x['recs'].values() for r in v)
    val['finite'] = fin
    if not fin:
        why.append('non-finite logits')
    if scen == 'offset_bf16':
        return not why, why, val
    if v4 is None or v4 < V4_MIN:
        why.append(f'G3a V4 {v4} < {V4_MIN}')
    if not rep or max(rep.values()) > REP_MAX:
        why.append(f'G3b replicate flip {val["rep_flip_max"]} > {REP_MAX}')
    if scen in ('leftpad', 'cache'):
        pv = ck.get('path_rel')
        val['path_rel'] = pv
        if pv is None or pv > PATH_REL:
            why.append(f'reference path differs from a plain pass (rel {pv})')
    if scen == 'window':
        w1, w2 = ck.get('W1_rel'), ck.get('W2_band_vs_full')
        val['W1_rel'], val['W2'] = w1, w2
        if w1 is None or w1 > W1_REL or w2 is None or w2 < W2_MIN:
            why.append(f'window mask not honoured (W1 rel {w1}, W2 {w2}): scenario not supported')
        val['acc_ref'] = M[(tag, scen)]['acc_ref']
        if M[(tag, scen)]['acc_ref'] < WIN_MIN_ACC:
            why.append(f'window reference accuracy {M[(tag, scen)]["acc_ref"]:.3f} < {WIN_MIN_ACC} (degenerate)')
    return not why, why, val


def model_admission(U, M, tag, cls):
    """Admission of a model to the summary over models ('offset', fp32): the unit checks of unit_gate, the explicit
    position path matches the default HF forward (key G1_rel), the gap control changes the logits (G2a), k = K_PRIMARY
    changes the logits of APE models (G2b), and the unshifted accuracy is above the lower bounds (G4)."""
    why, val = [], {}
    ok_u, why_u, val_u = unit_gate(U, M, tag, 'offset', cls)
    if not ok_u:
        return False, why_u, val_u
    ck = U[(tag, 'offset')]['checks']['summary']
    g1, g1a, g2 = ck.get('G1_rel'), ck.get('G1_default_vs_explicit'), ck.get('G2_gap_maxdiff')
    val.update(val_u, G1_rel=g1, G1_abs=g1a, G2a=g2)
    if g1 is None or g1 > G1_REL:
        why.append(f'G1_rel {g1} > {G1_REL}')
    if g2 is None or g2 < G2_MIN or g2 < G2_NOISE * (g1a or 0):
        why.append(f'G2a gap control {g2} < max({G2_MIN}, {G2_NOISE} x G1_default_vs_explicit {g1a})')
    m = M[(tag, 'offset')]
    if cls == 'APE':
        p = m['pooled'].get(f'k{K_PRIMARY}')
        frac = p['frac_logit_changed_gt_1e-3'] if p else None      # threshold G2B_MIN = 1e-3
        val['G2b'] = frac
        if frac is None or frac < G2B_FRAC:
            why.append(f'G2b APE k{K_PRIMARY} logit change on {frac} < {G2B_FRAC} of items')
    a0 = m['acc_ref_task']
    nt = sum(v >= G4_TASK_MIN for v in a0.values())
    val['G4'] = {'pooled_acc_ref': m['acc_ref'], 'n_tasks_above_floor': int(nt)}
    if m['acc_ref'] < G4_POOLED_MIN or nt < G4_MIN_TASKS:
        why.append(f'G4 non-degenerate tasks failed (pooled {m["acc_ref"]:.3f}, tasks with acc >= {G4_TASK_MIN}: {nt})')
    return not why, why, val


# ------------------------------------------------------------------ checks R1, R2, N1, N2 over the admitted models
def decide(M, admitted, cls_of, scen='offset'):
    rel = [t for t in admitted if cls_of[t] in ('RoPE', 'ALiBi')]
    ape = [t for t in admitted if cls_of[t] == 'APE']
    r1_cells, r1_fail, n1 = {}, [], []
    for t in rel:
        for l, p in M[(t, scen)]['pooled'].items():
            kind, k = row_kind(l)
            if kind != 'k' or p is None:
                continue
            r1_cells[f'{t}/{l}'] = p['ci']
            if not p['ci_contains_0']:
                r1_fail.append(f'{t}/{l}')
            if p['material']:
                n1.append(f'{t}/{l}')
    r2 = {}
    for t in ape:
        p = M[(t, scen)]['pooled'].get(f'k{K_PRIMARY}')
        r2[t] = bool(p and p['ci_below_0'])
    r2_by_k = {t: {l: bool(p['ci_below_0']) for l, p in M[(t, scen)]['pooled'].items()
                   if p and row_kind(l)[0] == 'k'} for t in ape}
    R1, R2 = not r1_fail, bool(ape) and all(r2.values())
    N1, N2 = bool(n1), sum(r2.values()) < len(ape) / 2 if ape else True
    return {'relative_models': rel, 'ape_models': ape, 'R1_pass': R1, 'R1_failing': r1_fail, 'R1_cells': r1_cells,
            'R2_pass': R2, 'R2': r2, 'R2_by_k_reported': r2_by_k, 'N1_material_relative': n1, 'N1': N1, 'N2': bool(N2)}


def secondary_pure(M, U, admitted, cls_of, unit_ok):
    """Rows of the other scenarios that shift all position indices (leftpad p<k>, cache block<k>, window k<k>) for the
    admitted RoPE and ALiBi models: lists the rows flagged 'material' and the rows whose CI excludes 0. Units that fail
    their validity checks are skipped and listed."""
    out = {'checked': [], 'excluded_units': [], 'material': [], 'ci_excludes_0': []}
    for t in admitted:
        if cls_of[t] not in ('RoPE', 'ALiBi'):
            continue
        for scen, kind in (('leftpad', 'p'), ('cache', 'block'), ('window', 'k')):
            if not unit_ok.get((t, scen), (False,))[0]:
                out['excluded_units'].append(f'{t}/{scen}')
                continue
            for l, p in M[(t, scen)]['pooled'].items():
                if p is None or row_kind(l)[0] != kind:
                    continue
                out['checked'].append(f'{t}/{scen}/{l}')
                if not p['ci_contains_0']:
                    out['ci_excludes_0'].append(f'{t}/{scen}/{l}')
                if p['material']:
                    out['material'].append(f'{t}/{scen}/{l}')
    return out


# ------------------------------------------------------------------ cards
def verdict(p):
    if p is None:
        return 'n/a'
    if p['ci_contains_0']:
        return 'unchanged'
    return 'degrades' if p['ci_below_0'] else 'improves'


SCEN_TEXT = {'offset': 'all position ids shifted by k',
             'leftpad': 'k left-pad tokens with naive absolute position ids (ref: positions from the mask)',
             'cache': 'prefix KV cache: block = cache and suffix shifted by k; gap = suffix placed k after the cache',
             'window': f'banded (sliding-window, W={WINDOW}) attention, window starting at position k',
             'offset_bf16': 'all position ids shifted by k, in bf16'}


def make_card(tag, U, M, fetch, adm, unit_ok):
    hub, rev, cls, fam, ctx, lic, size = MODELS[tag]
    card = {'model': tag, 'hub_id': hub, 'revision': (fetch or {}).get('sha'), 'license': lic, 'position_class': cls,
            'hf_family': fam, 'trained_context': ctx, 'params_B': PARAMS_B[tag],
            'admitted_to_main_checks': adm[0], 'admission_reasons': adm[1], 'gate_values': adm[2], 'scenarios': {}}
    for scen in SCENARIOS:
        m = M.get((tag, scen))
        if m is None:
            card['scenarios'][scen] = {'status': U.get((tag, scen), {}).get('status', 'missing')}
            continue
        ok, why, val = unit_ok[(tag, scen)]
        rows = {}
        for l in m['labels']:
            p = m['pooled'][l]
            if p is None:
                continue
            rows[l] = {'verdict': verdict(p), 'delta': r4(p['delta']), 'ci': [r4(p['ci'][0]), r4(p['ci'][1])],
                       'acc_ref': r4(p['acc_ref']), 'acc': r4(p['acc']), 'answer_flip_rate': r4(p['answer_flip_rate']),
                       'first_token_flip_rate': r4(p['first_token_flip_rate']),
                       'logit_maxabs_median': r4(p['logit_maxabs_median_of_tasks']), 'kl_median': p['kl_median_of_tasks'],
                       'n': p['n'], 'tasks': p['tasks'],
                       'per_task': {t: {'delta': r4(c['delta']), 'ci': [r4(c['ci'][0]), r4(c['ci'][1])],
                                        'acc_ref': r4(c['acc_ref']), 'flip': r4(c['answer_flip_rate'])}
                                    for t, c in m['cells'][l].items() if c}}
        card['scenarios'][scen] = {'what': SCEN_TEXT[scen], 'valid': ok, 'validity_notes': why, 'gate_values': val,
                                   'source': U[(tag, scen)]['src'], 'acc_ref': r4(m['acc_ref']), 'rows': rows}
    # serving recommendations (descriptive; from the shifted rows of each fp32 scenario)
    rec = {}
    for scen, kind, what in (('offset', 'k', 'shift all position ids by a constant'),
                             ('leftpad', 'p', 'left-pad with naive absolute position ids'),
                             ('cache', 'block', 'reuse a prefix KV cache computed at a different (block) offset'),
                             ('cache', 'gap', 'append to a cached prefix with a position gap'),
                             ('window', 'k', f'start a {WINDOW}-token sliding window at a non-zero position')):
        sc = card['scenarios'].get(scen, {})
        rows = {l: r for l, r in sc.get('rows', {}).items() if row_kind(l)[0] == kind}
        if not rows or not sc.get('valid'):
            rec[f'{scen}/{kind}'] = f'{what}: not assessed' + ('' if sc.get('valid') else ' (scenario invalid or missing)')
            continue
        bad = [l for l, r in rows.items() if r['verdict'] != 'unchanged']
        maxflip = max(r['answer_flip_rate'] for r in rows.values())
        if not bad:
            rec[f'{scen}/{kind}'] = f'{what}: accuracy unchanged at every tested k (max answer-flip rate {maxflip:.3f})'
        else:
            rec[f'{scen}/{kind}'] = f'{what}: accuracy changes at {", ".join(bad)} -> recompute/reset positions'
    card['recommendations'] = rec
    return card


def card_md(c):
    L = [f"# Answers under shifted position indices: {c['model']}", '',
         f"- hub `{c['hub_id']}` rev `{(c['revision'] or '')[:12]}`; licence: {c['license']}",
         f"- position class **{c['position_class']}** ({c['hf_family']}), trained context {c['trained_context']}, "
         f"{c['params_B']} B params",
         f"- admitted to the main checks: **{c['admitted_to_main_checks']}** "
         f"{'' if c['admitted_to_main_checks'] else '(' + '; '.join(c['admission_reasons']) + ')'}", '',
         '| scenario | row | verdict | Delta acc [95% CI] | acc ref -> row | answer flips | first-token flips | median max|dlogit| |',
         '|---|---|---|---|---|---|---|---|']
    for scen, s in c['scenarios'].items():
        if 'rows' not in s:
            L.append(f"| {scen} | - | {s.get('status')} | | | | | |")
            continue
        for l, r in s['rows'].items():
            if l == 'rep':
                continue
            L.append(f"| {scen}{'' if s['valid'] else ' (invalid)'} | {l} | {r['verdict']} | {r['delta']:+.4f} "
                     f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['acc_ref']:.3f} -> {r['acc']:.3f} | "
                     f"{r['answer_flip_rate']:.4f} | {r['first_token_flip_rate']:.4f} | {r['logit_maxabs_median']:.3g} |")
    L += ['', '**Serving recommendations (descriptive)**', '']
    L += [f'- {v}' for v in c['recommendations'].values()]
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ self-check
def selfcheck(base, split, U, M, items, plan, checks, admitted, cls_of):
    iss = defaultdict(list)
    tgt = {it['id']: it['target'] for it in items['items']}
    # (a) correctness recomputed from answers and the item file's targets
    n_rec = 0
    for (tag, scen), x in U.items():
        if x['status'] != 'done':
            continue
        pm = plan['models'].get(tag)
        for t, rr in x['recs'].items():
            ids_plan = [i for i, it in ((it['id'], it) for it in items['items']) if it['task'] == t and i in pm['items']]
            if x['done'].get('cap'):
                ids_plan = ids_plan[:x['done']['cap']]
            if [r['id'] for r in rr] != ids_plan:
                iss['coverage'].append(f'{tag}/{scen}/{t}: item ids differ from the plan ({len(rr)} vs {len(ids_plan)})')
            Kp = pm['tasks'][t]['K']
            for r in rr:
                n_rec += 1
                if r['target'] != tgt.get(r['id']):
                    iss['target'].append(f'{tag}/{scen}/{r["id"]}')
                if [int(a == r['target']) for a in r['ans']] != r['correct']:
                    iss['correct_recomputed'].append(f'{tag}/{scen}/{r["id"]}')
                if r.get('K', Kp) != Kp:
                    iss['K'].append(f'{tag}/{scen}/{r["id"]}')
                T = r['T']
                if [T, r['m']] != pm['items'][r['id']]:
                    iss['T_m'].append(f'{tag}/{scen}/{r["id"]}')
                for l, (a, b) in zip(r['labels'], r['pos']):
                    kind, k = row_kind(l)
                    k = k or 0
                    ea, eb = {'ref': (0, T - 1), 'rep': (0, T - 1), 'k': (k, k + T - 1), 'p': (k, k + T - 1),
                              'block': (k, k + T - 1), 'gap': (0, k + T - 1)}[kind]
                    if (a, b) != (ea, eb):
                        iss['positions'].append(f'{tag}/{scen}/{r["id"]}/{l}')
                    if b + r['steps'] >= pm['ctx'] + 1:
                        iss['context'].append(f'{tag}/{scen}/{r["id"]}/{l}: last position {b + r["steps"]} > ctx')
                if r['labels'][0] != 'ref' or r['labels'][-1] != 'rep':
                    iss['labels'].append(f'{tag}/{scen}/{r["id"]}')
        d = x['done']
        if d.get('split') != split:
            iss['split'].append(f'{tag}/{scen}: run split {d.get("split")} != {split}')
        fp = f'{base}/meta/fetch/{tag}.json'
        if os.path.exists(fp) and json.load(open(fp))['sha'] != d.get('sha'):
            iss['sha'].append(f'{tag}/{scen}')
        if d.get('precision') != ('bf16' if scen == 'offset_bf16' else 'fp32'):
            iss['precision'].append(f'{tag}/{scen}')
        if x['src'] == 'rescue' and not d.get('strict'):
            iss['rescue_strict'].append(f'{tag}/{scen}')
    # (b) the items of this split share no id, seed or prompt text with the other split (if its item files exist):
    # ids and seeds must be disjoint; identical prompt strings may coincide by chance (the few-shot items of the small
    # test split, split label pilot, and of the main split share one countdown trace prefix, countdown_1300009 =
    # countdown_401166) and fail the check only above max(1, 0.1%).
    leak = {}
    other = 'pilot' if split == 'main' else 'main'
    E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    po = f'{E}/smoke/data/items_pilot.json' if split == 'main' else f'{E}/data/items_main.json'
    fewshot_other = f'{FEWSHOT_DIR}/data/items_{other}.json'
    for name, p in (('this_experiment_' + other, po), ('fewshot_' + other, fewshot_other)):
        if os.path.exists(p):
            oi = json.load(open(p))['items']
            ov_id = len({it['id'] for it in items['items']} & {it['id'] for it in oi})
            ov_seed = len({(it['task'], it['seed']) for it in items['items']} & {(it['task'], it['seed']) for it in oi})
            shared = sorted({it['prompt'] for it in items['items']} & {it['prompt'] for it in oi})
            ids = [it['id'] for it in items['items'] if it['prompt'] in set(shared)]
            leak[name] = {'id_overlap': ov_id, 'seed_overlap': ov_seed, 'prompt_overlap': len(shared),
                          'shared_prompt_item_ids': ids[:20]}
            if ov_id or ov_seed or len(shared) > max(1, 0.001 * len(items['items'])):
                iss['leakage'].append(f'{name}: {ov_id} ids, {ov_seed} seeds, {len(shared)} prompts shared')
    if plan.get('items_sha256') != items.get('sha256'):
        iss['items_hash'].append('plan items_sha256 != items file sha256')
    # (c) independent re-implementation of the cells of R1 and R2 (index-resampling bootstrap, other seed)
    agree, border = 0, []
    cells = []
    for t in checks['relative_models']:
        for l, p in M[(t, 'offset')]['pooled'].items():
            if p is not None and row_kind(l)[0] == 'k':
                cells.append((t, l, 'contains'))
    for t in checks['ape_models']:
        cells.append((t, f'k{K_PRIMARY}', 'below'))
    for t, l, what in cells:
        p = M[(t, 'offset')]['pooled'][l]
        lo, hi = boot_ci_index(p['_ds'], seed(t, 'verify', l))
        a = (lo <= 0 <= hi) if what == 'contains' else (hi < 0)
        b = p['ci_contains_0'] if what == 'contains' else p['ci_below_0']
        if a == b:
            agree += 1
        elif min(abs(lo), abs(hi), abs(p['ci'][0]), abs(p['ci'][1])) <= 2e-3:
            border.append(f'{t}/{l}')
        else:
            iss['check_recomputation'].append(f'{t}/{l}: {what} {b} vs re-implementation {a}')
    # recompute delta by plain loops for these cells
    for t, l, what in cells:
        tot, n = 0, 0
        for task, rr in U[(t, 'offset')]['recs'].items():
            for r in rr:
                if l in r['labels']:
                    tot += r['correct'][r['labels'].index(l)] - r['correct'][r['labels'].index('ref')]
                    n += 1
        if n == 0 or abs(tot / n - M[(t, 'offset')]['pooled'][l]['delta']) > 1e-12:
            iss['delta_recomputation'].append(f'{t}/{l}')
    # (d) agreement of the fp32 'offset' rows with the answers of another run of the five few-shot tasks on
    # FEWSHOT_RUN_MODELS, read from <FEWSHOT_DIR>/results/raw (same items, same tokenisation; skipped if absent)
    fs_agree = {}
    for tag in FEWSHOT_RUN_MODELS:
        x = U.get((tag, 'offset'))
        if not x or x['status'] != 'done' or split != 'main':
            continue
        same, tot = 0, 0
        for task in FEWSHOT_TASKS:
            p = f'{FEWSHOT_DIR}/results/raw/{tag}_fp32_{task}.jsonl'
            if not os.path.exists(p) or task not in x['recs']:
                continue
            fs_recs = {}
            for line in open(p):
                r = json.loads(line); fs_recs[r['id']] = r
            for r in x['recs'][task]:
                o = fs_recs.get(r['id'])
                if o is None or o['T'] != r['T']:
                    continue
                for l, j in FEWSHOT_RUN_ROWS.items():
                    if l in r['labels']:
                        tot += 1      # that run did not stop at EOS: apply this experiment's EOS rule to its answers
                        fs_ans = o['ans'][j].split('<|endoftext|>')[0].strip()
                        same += int(fs_ans == r['ans'][r['labels'].index(l)])
        if tot:
            fs_agree[tag] = {'compared': tot, 'agreement': same / tot}
            if same / tot < 0.98:
                iss['fewshot_run_agreement'].append(f'{tag}: answer agreement {same / tot:.4f} < 0.98')
    out = {'n_records_checked': n_rec, 'leakage': leak, 'check_cells': len(cells), 'check_cells_agree': agree,
           'check_cells_borderline': border, 'fewshot_run_agreement': fs_agree,
           'issues': {k: (v[:50] + ([f'... {len(v) - 50} more'] if len(v) > 50 else [])) for k, v in iss.items()},
           'n_issues': {k: len(v) for k, v in iss.items()}}
    out['pass'] = not iss
    return out


# ------------------------------------------------------------------ main
def main():
    base = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else 'main'
    OUT = f'{base}/analysis'
    os.makedirs(f'{OUT}/cards', exist_ok=True)
    t0 = time.time()
    S = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'split': split, 'base': base}
    ut = f'{base}/results/markers/unittest.json'
    G0 = os.path.exists(ut) and json.load(open(ut)).get('pass', False)
    S['G0_unit_tests_pass'] = bool(G0)
    items_p, plan_p = f'{base}/data/items_{split}.json', f'{base}/data/plan_{split}.json'
    if not (os.path.exists(items_p) and os.path.exists(plan_p)):
        S['status'] = 'not_run'; S['reason'] = 'no item plan'
        return finish(OUT, S, {'status': 'not_run', 'pass': False}, None, {})
    items, plan = json.load(open(items_p)), json.load(open(plan_p))
    U, rs = load_units(base)
    cls_of = {t: MODELS[t][2] for t in ORDER}
    fetched = {t: json.load(open(f'{base}/meta/fetch/{t}.json')) for t in ORDER
               if os.path.exists(f'{base}/meta/fetch/{t}.json')}
    M = {}
    for (tag, scen), x in U.items():
        if x['status'] == 'done':
            M[(tag, scen)] = unit_metrics(U, tag, scen)
    unit_ok = {k: unit_gate(U, M, k[0], k[1], cls_of[k[0]]) for k in M}
    adm = {}
    for tag in ORDER:
        if (tag, 'offset') in M:
            adm[tag] = model_admission(U, M, tag, cls_of[tag])
        else:
            adm[tag] = (False, ['offset unit missing' if tag in fetched else 'model not fetched'], {})
    admitted = [t for t in ORDER if adm[t][0]]
    cover = {}
    for c in CLASSES:
        allc = [t for t in ORDER if cls_of[t] == c]
        na = sum(t in admitted for t in allc)
        nmin, fmin = MIN_ADMIT[c]
        cover[c] = {'models': len(allc), 'fetched': sum(t in fetched for t in allc), 'admitted': na,
                    'pass': bool(na >= nmin and na >= fmin * len(allc))}
    dec = decide(M, admitted, cls_of) if admitted else None
    sec = secondary_pure(M, U, admitted, cls_of, unit_ok) if admitted else None
    # bf16 secondary (reported only)
    bf = {}
    adm_bf = []
    for t in admitted:
        if (t, 'offset_bf16') in M and unit_ok[(t, 'offset_bf16')][0]:
            a32, a16 = M[(t, 'offset')]['acc_ref'], M[(t, 'offset_bf16')]['acc_ref']
            bf[t] = {'fp32_acc_ref': a32, 'bf16_acc_ref': a16, 'admitted': bool(a16 >= HALF_ADMIT * a32)}
            if bf[t]['admitted']:
                adm_bf.append(t)
    dec_bf = decide(M, adm_bf, cls_of, scen='offset_bf16') if adm_bf else None
    # gap rows (control, descriptive): fraction of valid (model, k) cells whose CI excludes 0 / logits change
    gap = {}
    for c in CLASSES:
        rows = [(t, l, p) for t in ORDER if cls_of[t] == c and (t, 'cache') in M and unit_ok[(t, 'cache')][0]
                for l, p in M[(t, 'cache')]['pooled'].items() if p and row_kind(l)[0] == 'gap']
        gap[c] = {'cells': len(rows), 'ci_excludes_0': sum(not p['ci_contains_0'] for _, _, p in rows),
                  'mean_frac_logit_changed': float(np.mean([p['frac_logit_changed_gt_1e-3'] for _, _, p in rows]))
                  if rows else None,
                  'mean_answer_flip_rate': float(np.mean([p['answer_flip_rate'] for _, _, p in rows])) if rows else None}
    # self-check
    VJ = selfcheck(base, split, U, M, items, plan, dec, admitted, cls_of) if dec else {'pass': False,
                                                                                       'note': 'no admitted model'}
    # status of the checks
    n_done = sum(1 for x in U.values() if x['status'] == 'done')
    if not G0 or not any((t, 'offset') in M for t in ORDER):
        status, why = 'not_run', 'G0 unit tests failed or no offset results'
    elif dec is None:
        status, why = 'not_all_checks_pass', 'no model passed the admission gates (coverage fails)'
    elif dec['N1'] or dec['N2']:
        status, why = 'N1_or_N2', f"N1 {dec['N1_material_relative']} N2 {dec['N2']}"
    elif (dec['R1_pass'] and dec['R2_pass'] and all(v['pass'] for v in cover.values()) and not sec['material']
          and VJ['pass']):
        status, why = 'all_checks_pass', 'G0, coverage, R1, R2, other shifted scenarios and self-check pass'
    else:
        fails = [n for n, ok in (('R1', dec['R1_pass']), ('R2', dec['R2_pass']),
                                 ('coverage', all(v['pass'] for v in cover.values())),
                                 ('other_shift_scenarios', not sec['material']), ('self_check', VJ['pass'])) if not ok]
        status, why = 'not_all_checks_pass', 'failing: ' + ', '.join(fails)
    S.update({'status': status, 'status_reason': why, 'units_done': n_done, 'units_total': len(units()),
              'rescued_units': sorted(units()[u][0] for u in rs),
              'rescue_results_used': sorted(f'{k[0]}__{k[1]}' for k, x in U.items() if x.get('src') == 'rescue'),
              'excluded_models_registry': EXCLUDED,
              'licences': {t: MODELS[t][5] for t in ORDER},
              'fetched': {t: {'hub': MODELS[t][0], 'sha': fetched[t]['sha'], 'licence_tags': fetched[t]['license_tags']}
                          for t in fetched},
              'admission': {t: {'admitted': adm[t][0], 'reasons': adm[t][1], 'values': adm[t][2]} for t in ORDER},
              'coverage': cover, 'checks_fp32_offset': dec, 'other_shift_scenarios': sec,
              'bf16_secondary': {'admission': bf, 'checks_reported_only': dec_bf},
              'gap_negative_control': gap,
              'unit_validity': {f'{k[0]}/{k[1]}': {'valid': v[0], 'notes': v[1], 'values': v[2]}
                                for k, v in unit_ok.items()},
              'pooled': {f'{k[0]}/{k[1]}': strip(m['pooled']) for k, m in M.items()},
              'cells': {f'{k[0]}/{k[1]}': strip(m['cells']) for k, m in M.items()},
              'acc_ref': {f'{k[0]}/{k[1]}': {'pooled': m['acc_ref'], 'tasks': m['acc_ref_task']} for k, m in M.items()},
              'compute': {'gpu_hours_task_loops': round(sum(
                  sum(x['done']['stats'].get('task_sec', {}).values()) for x in U.values() if x['status'] == 'done') /
                  3600, 2), 'gpus': sorted({x['done']['gpu'] for x in U.values() if x['status'] == 'done'})}})
    cards = {}
    for tag in ORDER:
        if not any((tag, s) in M for s in SCENARIOS):
            continue
        cards[tag] = make_card(tag, U, M, fetched.get(tag), adm[tag], unit_ok)
    VJ['status'] = status
    VJ['rederived_status'] = rederive(S, VJ)
    VJ['rederived_agrees'] = VJ['rederived_status'] == status
    if not VJ['rederived_agrees']:
        VJ['pass'] = False
    return finish(OUT, S, VJ, dec, cards, t0)


def rederive(S, VJ):
    """Status re-derived only from the JSON-level summary (admission, pooled CIs, coverage, secondary rows) by a
    separate code path, as a consistency check of summary.json."""
    if not S['G0_unit_tests_pass'] or not any(k.endswith('/offset') for k in S['pooled']):
        return 'not_run'
    adm = [t for t, a in S['admission'].items() if a['admitted']]
    if not adm:
        return 'not_all_checks_pass'
    rel = [t for t in adm if MODELS[t][2] != 'APE']
    ape = [t for t in adm if MODELS[t][2] == 'APE']
    off = lambda t: S['pooled'][f'{t}/offset']
    kr = lambda d: {l: p for l, p in d.items() if p and l[0] == 'k' and l[1:].isdigit()}
    n1 = any(not (p['ci'][0] <= 0 <= p['ci'][1]) and abs(p['delta']) >= MATERIAL for t in rel for p in kr(off(t)).values())
    r2 = [bool(off(t).get(f'k{K_PRIMARY}') and off(t)[f'k{K_PRIMARY}']['ci'][1] < 0) for t in ape]
    if n1 or not ape or sum(r2) < len(ape) / 2:
        return 'N1_or_N2'
    r1 = all(p['ci'][0] <= 0 <= p['ci'][1] for t in rel for p in kr(off(t)).values())
    sec_bad = False
    for t in rel:
        for scen, pre in (('leftpad', 'p'), ('cache', 'block'), ('window', 'k')):
            v = S['unit_validity'].get(f'{t}/{scen}')
            if not v or not v['valid']:
                continue
            for l, p in S['pooled'][f'{t}/{scen}'].items():
                if p and l.startswith(pre) and l[len(pre):].isdigit() and p['material']:
                    sec_bad = True
    cov = all(v['pass'] for v in S['coverage'].values())
    if r1 and all(r2) and cov and not sec_bad and VJ.get('pass'):
        return 'all_checks_pass'
    return 'not_all_checks_pass'


def finish(OUT, S, VJ, dec, cards, t0=None):
    os.makedirs(f'{OUT}/cards', exist_ok=True)
    for tag, c in cards.items():
        json.dump(c, open(f'{OUT}/cards/{tag}.json', 'w'), indent=1)
        open(f'{OUT}/cards/{tag}.md', 'w').write(card_md(c))
    write_tables(OUT, S)
    write_cards_table(OUT, cards)
    json.dump(VJ, open(f'{OUT}/verify.json.tmp', 'w'), indent=1)
    os.replace(f'{OUT}/verify.json.tmp', f'{OUT}/verify.json')
    write_results(OUT, S, VJ, cards)
    S['seconds'] = round(time.time() - t0, 1) if t0 else None
    json.dump(S, open(f'{OUT}/summary.json.tmp', 'w'), indent=1)      # summary.json is written last
    os.replace(f'{OUT}/summary.json.tmp', f'{OUT}/summary.json')
    print('status:', S['status'], '|', S.get('status_reason'), '| self-check', VJ.get('pass'))
    return 0


def write_tables(OUT, S):
    if 'cells' not in S:
        return
    with open(f'{OUT}/table_cells.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'class', 'scenario', 'task', 'row', 'n', 'acc_ref', 'acc_row', 'delta', 'ci_lo', 'ci_hi',
                    'answer_flip_rate', 'c_to_w', 'w_to_c', 'first_token_flip_rate', 'logit_maxabs_median',
                    'logit_maxabs_max', 'kl_median'])
        for key, cl in S['cells'].items():
            tag, scen = key.split('/')
            for l, bt in cl.items():
                for t, c in bt.items():
                    if c:
                        w.writerow([tag, MODELS[tag][2], scen, t, l, c['n'], f"{c['acc_ref']:.4f}", f"{c['acc']:.4f}",
                                    f"{c['delta']:+.4f}", f"{c['ci'][0]:+.4f}", f"{c['ci'][1]:+.4f}",
                                    f"{c['answer_flip_rate']:.4f}", c['n_correct_to_wrong'], c['n_wrong_to_correct'],
                                    f"{c['first_token_flip_rate']:.4f}", f"{c['logit_maxabs_median']:.4g}",
                                    f"{c['logit_maxabs_max']:.4g}", f"{c['kl_median']:.3g}"])
    with open(f'{OUT}/table_pooled.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'class', 'scenario', 'row', 'n', 'tasks', 'acc_ref', 'acc_row', 'delta', 'ci_lo', 'ci_hi',
                    'material', 'answer_flip_rate', 'first_token_flip_rate', 'frac_logit_changed'])
        for key, pl in S['pooled'].items():
            tag, scen = key.split('/')
            for l, p in pl.items():
                if p:
                    w.writerow([tag, MODELS[tag][2], scen, l, p['n'], ' '.join(p['tasks']), f"{p['acc_ref']:.4f}",
                                f"{p['acc']:.4f}", f"{p['delta']:+.4f}", f"{p['ci'][0]:+.4f}", f"{p['ci'][1]:+.4f}",
                                int(p['material']), f"{p['answer_flip_rate']:.4f}", f"{p['first_token_flip_rate']:.4f}",
                                f"{p['frac_logit_changed_gt_1e-3']:.4f}"])


def write_cards_table(OUT, cards):
    L = ['# Answers under shifted position indices, all models (pooled over tasks; verdict = 95% CI of Delta accuracy vs the k = 0 reference)', '',
         '| model | class | admitted | offset k16/k64/k256/k1024 | leftpad p16/p64/p256/p1024 | cache block | cache gap | '
         'window | bf16 offset |', '|---|---|---|---|---|---|---|---|---|']
    ab = {'unchanged': '=', 'degrades': 'v', 'improves': '^', 'n/a': '.'}
    for tag, c in cards.items():
        def seq(scen, kind):
            s = c['scenarios'].get(scen, {})
            rows = s.get('rows', {})
            v = ''.join(ab[rows[f'{kind}{k}']['verdict']] if f'{kind}{k}' in rows else '.' for k in KS)
            return v + ('' if s.get('valid', False) else ' (inv)') if rows else '-'
        L.append(f"| {tag} | {c['position_class']} | {'yes' if c['admitted_to_main_checks'] else 'no'} | "
                 f"{seq('offset', 'k')} | {seq('leftpad', 'p')} | {seq('cache', 'block')} | {seq('cache', 'gap')} | "
                 f"{seq('window', 'k')} | {seq('offset_bf16', 'k')} |")
    L += ['', 'Legend per k in (16, 64, 256, 1024): = unchanged (CI contains 0), v degrades (CI < 0), ^ improves, '
          '. not tested (context too short); (inv) = unit failed its validity gate.']
    open(f'{OUT}/cards_table.md', 'w').write('\n'.join(L) + '\n')


def write_results(OUT, S, VJ, cards):
    L = ['# Serving experiment - results (generated by code/analyze.py)', '',
         f"## Status of the checks: **{S['status']}**", '', f"Reason: {S.get('status_reason', S.get('reason'))}", '']
    if 'checks_fp32_offset' not in S or S['checks_fp32_offset'] is None:
        open(f'{OUT}/RESULTS.md', 'w').write('\n'.join(L) + '\n')
        return
    d = S['checks_fp32_offset']
    L += [f"- G0 unit tests: {S['G0_unit_tests_pass']}; units done {S['units_done']}/{S['units_total']}; "
          f"strict re-runs: {len(S['rescued_units'])} selected, {len(S['rescue_results_used'])} used",
          f"- Coverage: " + '; '.join(f"{c}: {v['admitted']}/{v['models']} admitted (pass {v['pass']})"
                                      for c, v in S['coverage'].items()),
          f"- R1 (every admitted RoPE/ALiBi model unchanged within CI at every k, fp32, all position indices shifted): **{d['R1_pass']}**"
          f" (failing: {d['R1_failing'] or 'none'})",
          f"- R2 (every admitted APE model degrades at k={K_PRIMARY}, CI < 0): **{d['R2_pass']}** "
          f"(not degrading: {[t for t, v in d['R2'].items() if not v] or 'none'})",
          f"- N1 material change in a relative model: {d['N1_material_relative'] or 'none'}; N2: {d['N2']}",
          f"- Other scenarios that shift all position indices (leftpad p<k>, cache block<k>, window k<k>): material changes "
          f"{S['other_shift_scenarios']['material'] or 'none'}; CI excluding 0 (reported) "
          f"{S['other_shift_scenarios']['ci_excludes_0'] or 'none'}; excluded units "
          f"{S['other_shift_scenarios']['excluded_units'] or 'none'}",
          f"- Self-check (verify.json): pass {VJ.get('pass')}; issues {VJ.get('n_issues')}; cells of R1 and R2 re-implemented "
          f"{VJ.get('check_cells_agree')}/{VJ.get('check_cells')} agree (borderline {VJ.get('check_cells_borderline')})",
          f"- Agreement with the other run of the few-shot tasks: "
          + ', '.join(f"{t} {v['agreement']:.4f}" for t, v in VJ.get('fewshot_run_agreement', {}).items()),
          f"- Gap rows (negative control, relative distances change): " +
          '; '.join(f"{c}: {v['ci_excludes_0']}/{v['cells']} cells CI excl. 0, mean logit-change frac "
                    f"{v['mean_frac_logit_changed'] if v['mean_frac_logit_changed'] is None else round(v['mean_frac_logit_changed'], 3)}"
                    for c, v in S['gap_negative_control'].items()),
          f"- bf16 secondary (reported only): admitted {[t for t, v in S['bf16_secondary']['admission'].items() if v['admitted']]}; "
          f"R1 {S['bf16_secondary']['checks_reported_only'] and S['bf16_secondary']['checks_reported_only']['R1_pass']}, "
          f"R2 {S['bf16_secondary']['checks_reported_only'] and S['bf16_secondary']['checks_reported_only']['R2_pass']}",
          f"- Compute: {S['compute']['gpu_hours_task_loops']} GPU-h in the item loops (excl. loading/checks); GPUs {S['compute']['gpus']}", '',
          '## Primary table (fp32, all position indices shifted; pooled over tasks; Delta accuracy vs k = 0 with 95% paired bootstrap CI)',
          '', '| model | class | admitted | acc k=0 | k16 | k64 | k256 | k1024 | flips k256 |', '|---|---|---|---|---|---|---|---|---|']
    for tag in ORDER:
        p = S['pooled'].get(f'{tag}/offset')
        if not p:
            continue
        def f(l):
            x = p.get(l)
            return '.' if not x else f"{x['delta']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]"
        a = S['acc_ref'][f'{tag}/offset']['pooled']
        fl = p.get(f'k{K_PRIMARY}', {}) or {}
        L.append(f"| {tag} | {MODELS[tag][2]} | {S['admission'][tag]['admitted']} | {a:.3f} | {f('k16')} | {f('k64')} | "
                 f"{f('k256')} | {f('k1024')} | {fl.get('answer_flip_rate', float('nan')):.3f} |")
    L += ['', 'Per-model cards: analysis/cards/<model>.md (JSON alongside); overview: analysis/cards_table.md; all cells: '
          'analysis/table_cells.csv, analysis/table_pooled.csv.']
    open(f'{OUT}/RESULTS.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    sys.exit(main())
