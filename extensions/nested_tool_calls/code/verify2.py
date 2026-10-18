"""Checks of the stored runs, written separately from run2.py / analyze2.py.
It re-derives everything it can from prompt TEXT and raw model outputs:
  V-gold    every I prompt is rendered and parsed back from text only; a separate evaluator recomputes the gold value
            and the nesting depth d of every test and calibration item
  V-hash    md5 of the rendered I / E prompt equals the stored prompt hash of each record
  V-ctrl    a separate text-level implementation of the split rebuilds every A / H call prompt from the I prompt text
            and the stored answers of the earlier calls; md5 must equal the stored hash of every call; number of
            calls and nesting of every call recomputed
  V-score   a separate extractor (character scan, no regex) re-scores every stored raw output
  V-dhat    d_hat*(tau), D_H and the 'included' flag recomputed from the calibration file with a separate fit that
            may not increase with depth (min-max formula) and matched to dhat_<task>.json, budgets_<task>.json and
            analysis/metrics.json
  V-metric  per-depth accuracies (I, E, A, H) and mean calls/tokens recomputed with plain loops and matched to
            analysis/metrics.json
  V-leak    no item of the calibration split equals a test item (structure, constants and input); the item sets
            generated with the split labels 'pilot2', 'pilot3', 'pilot4', 'pilot6', 'diag' (40 items per depth) and
            'smokecal', 'smoketest' (4 items per depth) share no item with the test and calibration items;
            demonstrations use names outside the task pool; d_hat* uses no test data
  V-sanity  record counts per depth, prompt length constant over depth, answer distribution not degenerate,
            calls of a split never exceed their budget
Writes analysis/verify.json ({"pass": bool, "failed": [...], "checks": {...}}). Usage: python verify2.py"""
import hashlib, json, math, os, re, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import tasks2 as T
from models import ORDER

DS = C.DEPTHS
failed = []
checks = collections.OrderedDict()


def note(name, ok, detail=None):
    checks.setdefault(name, {'pass': True, 'n': 0, 'examples': []})
    checks[name]['n'] += 1
    if not ok:
        checks[name]['pass'] = False
        if len(checks[name]['examples']) < 5:
            checks[name]['examples'].append(detail)


def rd(p):
    with open(p) as f:
        L = [json.loads(l) for l in f]
    return L[0]['_meta'], L[1:]


def md5(u, p):
    return hashlib.md5((u + '\x00' + p).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ text parsing (independent of tasks2 internals)
def parse_real(task, txt):
    real = txt.split('Now the real question.\n', 1)[1]
    lines = real.strip('\n').split('\n')
    assert lines[0] in ('Registry:', 'To-do notes:'), lines[0]
    defs, order, known = {}, [], None
    i = 1
    while i < len(lines) and lines[i].startswith(('def ', '- ', '# note')):
        ln = lines[i]
        i += 1
        if ln.startswith(('# note', '- (note)')):
            continue
        if task == 'T':
            name = ln[4:ln.index('(')]
            body = ln.split(': return ', 1)[1]
            if '  # result for x = ' in body:          # substituted helper result
                known = (name, body.split('  #', 1)[0])
                continue
            left, c = body.rsplit(' + ', 1)
            callee = None if left == 'x' else left[:left.index('(')]
        else:
            name = ln[2:ln.index(':')]
            rest = ln[ln.index(':') + 2:]
            if rest.startswith('already done by a helper agent'):
                known = (name, rest.rsplit('finished at step ', 1)[1].rstrip('.'))
                continue
            if rest.startswith('needs '):
                callee = rest[6:rest.index(' first')]
                c = rest.split('then ', 1)[1].split(' ')[0]
            else:
                callee = None
                c = rest.split('takes ', 1)[1].split(' ')[0]
        defs[name] = (callee, int(c))
        order.append(name)
    rest = lines[i:]
    for ln in rest:
        if ln.startswith('Known result'):
            if task == 'T':
                seg = ln.split('): ', 1)[1]
                g = seg[:seg.index('(')]
                v = seg.split(' = ', 1)[1].split('. Use', 1)[0]
            else:
                seg = ln.split(', ', 1)[1]
                g = seg.split(' is finished', 1)[0]
                v = seg.split('at step ', 1)[1].split('. Use', 1)[0]
            known = (g, v)
    q = rest[-1]
    if task == 'T':
        f = q[len('Question: What does '):q.index('(')]
        x = int(q[q.index('(') + 1:q.index(')')])
    else:
        f = q[len('Question: The agent starts '):].split(' at step')[0]
        x = int(q.split(' at step ', 1)[1].split('.', 1)[0])
    return defs, order, known, f, x, rest


def evaluate(defs, known, f, x):
    """(value or None if a known value is not an integer, nesting levels resolved implicitly, chain names)"""
    total, depth, chain = 0, 0, [f]
    while True:
        if known is not None and f == known[0]:
            try:
                return int(known[1]) + total, depth, chain
            except ValueError:
                return None, depth, chain
        cal, c = defs[f]
        total += c
        depth += 1
        if cal is None:
            return x + total, depth, chain
        f = cal
        chain.append(f)


def first_int(s):
    out, i = '', 0
    while i < len(s) and not (s[i].isdigit() or (s[i] == '-' and i + 1 < len(s) and s[i + 1].isdigit())):
        i += 1
    if i < len(s) and s[i] == '-':
        out, i = '-', i + 1
    while i < len(s) and s[i].isdigit():
        out += s[i]
        i += 1
    return out or None


def known_line(task, g, x, v):
    if task == 'T':
        return f'Known result (already computed by a helper): {g}({x}) = {v}. Use it instead of recomputing {g}.'
    return (f'Known result (reported by a helper agent): started at step {x}, {g} is finished at step {v}. '
            f'Use it instead of working out {g} again.')


def question(task, f, x):
    return (f'Question: What does {f}({x}) return?' if task == 'T' else
            f'Question: The agent starts {f} at step {x}. At which step is {f} finished?')


def subst_line(task, g, x, v):
    if task == 'T':
        return f'def {g}(x): return {v}  # result for x = {x}, returned by a helper call'
    return f'- {g}: already done by a helper agent; started at step {x}, it was finished at step {v}.'


def ctrl_prompts(task, i_txt, D, answers):
    """Text-level implementation of the split, separate from tasks2.build_calls (the answer of the previous call
    replaces the definition line of the tool it computed). answers: stored extracted answers of the calls in order
    (None -> '?').
    Returns list of (prompt, nest, chain-name queried)."""
    defs, order, known, f, x, rest = parse_real(task, i_txt)
    _, d, chain = evaluate(defs, None, f, x)
    pre, real = i_txt.split('Now the real question.\n', 1)
    body = real[:real.rindex('\nQuestion: ') + 1]
    if d <= D:
        return [(i_txt, d, f)]
    cuts, top = [], d
    while top > 0:
        lo = max(0, top - D)
        cuts.append((lo, top if top < d else None))
        top = lo
    out = []
    for k, (lo, kn) in enumerate(cuts):
        b = body
        if kn is not None:
            a = answers[k - 1]
            g = chain[kn]
            prefix = ('def ' + g + '(x): ') if task == 'T' else ('- ' + g + ': ')
            lines = b.split('\n')
            hit = [j for j, ln in enumerate(lines) if ln.startswith(prefix)]
            assert len(hit) == 1, (g, hit)
            lines[hit[0]] = subst_line(task, g, x, a if a is not None else '?')
            b = '\n'.join(lines)
        txt = pre + 'Now the real question.\n' + b + question(task, chain[lo], x)
        out.append((txt, (kn if kn is not None else d) - lo, chain[lo]))
    return out


# ------------------------------------------------------------------ separate fit (not increasing with depth) and d_hat
def iso_dec(acc, n):
    """Weighted least-squares non-increasing fit by the min-max formula."""
    k = len(acc)
    fit = []
    for t in range(k):
        best = None
        for i in range(t + 1):
            mx = None
            for j in range(t, k):
                w = sum(n[i:j + 1])
                m = sum(a * b for a, b in zip(acc[i:j + 1], n[i:j + 1])) / w
                mx = m if mx is None else max(mx, m)
            best = mx if best is None else min(best, mx)
        fit.append(best)
    return fit


def dhat_ind(acc, n, tau):
    fit = iso_dec(acc, n)
    ok = [d for d, f in zip(DS, fit) if f >= tau - 1e-12]
    return (max(ok) if ok else 0), fit


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def main():
    t0 = time.time()
    metrics = json.load(open(f'{C.ANA}/metrics.json'))['results'] if os.path.exists(f'{C.ANA}/metrics.json') else {}
    test_items = {t: {(d, i): T.make_item(t, C.SPLIT_TEST, d, i, M=C.M_FUN, order=C.ORDER, dis_mode=C.DIS_MODE)
                      for d in DS for i in range(C.N_TEST)} for t in C.TASKS}
    cal_items = {t: {(d, i): T.make_item(t, C.SPLIT_CAL, d, i, M=C.M_FUN, order=C.ORDER, dis_mode=C.DIS_MODE)
                     for d in DS for i in range(C.N_CAL)} for t in C.TASKS}
    # V-leak: fingerprints
    for t in C.TASKS:
        fp_test = {T.fingerprint(it) for it in test_items[t].values()}
        fp_cal = {T.fingerprint(it) for it in cal_items[t].values()}
        note('leak_cal_vs_test', not (fp_test & fp_cal), f'{t}: {len(fp_test & fp_cal)} shared items')
        other = set()
        for sp in ('pilot2', 'pilot3', 'pilot4', 'pilot6', 'diag') + (('cal', 'test') if C.SMOKE else ('smokecal', 'smoketest')):
            for d in DS:
                for i in range(40 if sp.startswith(('pilot', 'diag')) else 4):
                    other.add(T.fingerprint(T.make_item(t, sp, d, i, M=C.M_FUN, order=C.ORDER, dis_mode=C.DIS_MODE)))
        note('leak_vs_other_item_sets', not ((fp_test | fp_cal) & other), t)
        note('leak_demo_names', not (set(T.DEMO_NAMES) & set(T.NAMES)), t)
    # texts of I prompts and gold from text
    i_txt = {}
    for t in C.TASKS:
        for split, items in (('test', test_items[t]), ('cal', cal_items[t])):
            for key, it in items.items():
                c = T.build_calls(it, None)[-1]
                u, p = T.render(t, it, c, 'I', {})
                defs, order, known, f, x, _ = parse_real(t, u)
                v, d, chain = evaluate(defs, known, f, x)
                note('gold_from_text', str(v) == T.gold_of(it, c) and d == key[0] and
                     len(defs) == key[0] + C.N_TOOLS_EXTRA,
                     f'{t} {split} {key}')
                i_txt[(t, split, key)] = (u, p, str(v))
    # per model / task
    for m in ORDER:
        for t in C.TASKS:
            md = f'{C.RES}/{m}/{t}'
            mt = metrics.get(m, {}).get(t, {})
            pc = f'{md}_I_cal.jsonl'
            if not os.path.exists(pc):
                continue
            # calibration: hashes, scores, d_hat and the 'included' flag recomputed
            _, cal = rd(pc)
            byd = collections.defaultdict(list)
            for r in cal:
                u, p, g = i_txt[(t, 'cal', (r['d'], r['idx']))]
                c0 = r['calls'][0]
                note('hash_I_cal', c0['phash'] == md5(u, p), f'{m} {t} {r["d"]} {r["idx"]}')
                ok = first_int(c0['out']) == g
                note('rescore', ok == r['ok'] and c0['gold'] == g, f'{m} {t} cal {r["d"]} {r["idx"]}')
                byd[r['d']].append(1.0 if ok else 0.0)
            note('count_cal', all(len(byd[d]) == C.N_CAL for d in DS), f'{m} {t}')
            acc = [sum(byd[d]) / len(byd[d]) for d in DS]
            n = [len(byd[d]) for d in DS]
            dj = json.load(open(f'{C.RES}/{m}/dhat_{t}.json'))
            incl = all(acc[DS.index(d)] >= C.TAU for d in C.INCL_DEPTHS)
            note('dhat_included_flag', incl == dj['included'] and (not mt or mt['cal']['included'] == incl), f'{m} {t}')
            Dind = {}
            for tau in C.TAUS:
                dh, _ = dhat_ind(acc, n, tau)
                Dind[tau] = max(1, dh)
                note('dhat_recomputed', dh == dj[f'tau{tau}']['dhat'] and
                     (not mt or mt['cal'][f'tau{tau}']['dhat'] == dh), f'{m} {t} tau {tau}: {dh} vs {dj[f"tau{tau}"]["dhat"]}')
            pb = f'{C.RES}/{m}/budgets_{t}.json'
            files = {'I': f'{md}_I_test.jsonl', 'E': f'{md}_E_test.jsonl', 'A': f'{md}_A_test.jsonl'}
            if not (os.path.exists(pb) and all(os.path.exists(p) for p in files.values())):
                continue
            bud = json.load(open(pb))
            for tau in C.TAUS:
                note('budget_from_cal_only', bud[str(tau)]['D'] == Dind[tau], f'{m} {t} {tau}')
                files[f'H{tau}'] = f"{C.RES}/{m}/{bud[str(tau)]['file']}"
            for k, p in files.items():
                _, recs = rd(p)
                D = None if k in ('I', 'E') else (1 if k == 'A' else Dind[float(k[1:])])
                acc_d = collections.defaultdict(list)
                calls_d = collections.defaultdict(list)
                tok_d = collections.defaultdict(list)
                ptok_d = collections.defaultdict(list)
                gold_d = collections.defaultdict(list)
                for r in recs:
                    u, p_, g = i_txt[(t, 'test', (r['d'], r['idx']))]
                    if k == 'I':
                        note('hash_I_test', r['calls'][0]['phash'] == md5(u, p_), f'{m} {t} {r["d"]}')
                    elif k == 'E':
                        it = test_items[t][(r['d'], r['idx'])]
                        ue, pe = T.render(t, it, T.build_calls(it, None)[-1], 'E', {})
                        chain_line = [ln for ln in ue.split('Now the real question.\n', 1)[1].split('\n')
                                      if ln.startswith('Call chain') or ln.startswith('Open to-do')]
                        pr = parse_real(t, u)
                        _, _, chain = evaluate(pr[0], None, pr[3], pr[4])
                        note('E_chain_line', len(chain_line) == 1 and all(nm in chain_line[0] for nm in chain) and
                             chain_line[0].count('->') == len(chain) - 1, f'{m} {t} {r["d"]}')
                        note('hash_E_test', r['calls'][0]['phash'] == md5(ue, pe), f'{m} {t} {r["d"]}')
                    else:
                        cps = ctrl_prompts(t, u, D, [c['ex'] for c in r['calls']])
                        defs_I = parse_real(t, u)[0]
                        note('ctrl_ncalls', len(cps) == len(r['calls']) == r['n_calls'] and
                             r['n_calls'] == (1 if r['d'] <= D else math.ceil(r['d'] / D)), f'{m} {t} {k} {r["d"]}')
                        for (txt, nest, f), c in zip(cps, r['calls']):
                            note('ctrl_hash', c['phash'] == md5(txt, T.PREFILL), f'{m} {t} {k} {r["d"]} {r["idx"]} call {c["id"]}')
                            note('ctrl_nest_le_D', nest <= D and nest == c['nest'], f'{m} {t} {k}')
                            defs_c, _, known_c, ff, x, _ = parse_real(t, txt)
                            _, nest_txt, _ = evaluate(defs_c, known_c, ff, x)   # implicit levels of this call (text)
                            v, _, _ = evaluate(defs_I, None, ff, x)            # true value of its tool (I registry)
                            note('ctrl_call_gold', str(v) == c['gold'] and nest_txt == c['nest'],
                                 f'{m} {t} {k} {r["d"]}')
                    for c in r['calls']:
                        note('rescore', (first_int(c['out']) == c['gold']) == c['ok'] and first_int(c['out']) == c['ex'],
                             f'{m} {t} {k} {r["d"]} {r["idx"]} {c["out"]!r}')
                    main_ok = first_int(r['calls'][-1]['out']) == g
                    note('main_gold', r['calls'][-1]['gold'] == g and main_ok == r['ok'], f'{m} {t} {k} {r["d"]}')
                    acc_d[r['d']].append(1.0 if main_ok else 0.0)
                    calls_d[r['d']].append(len(r['calls']))
                    tok_d[r['d']].append(sum(c['ptok'] + c['otok'] for c in r['calls']))
                    ptok_d[r['d']].append(sum(c['ptok'] for c in r['calls']))
                    gold_d[r['d']].append(g)
                note('count_test', all(len(acc_d[d]) == C.N_TEST for d in DS), f'{m} {t} {k}')
                if mt.get('complete'):
                    for d in DS:
                        a = sum(acc_d[d]) / len(acc_d[d])
                        note('metric_acc', close(a, mt['curves'][k][str(d)]), f'{m} {t} {k} d={d}')
                        note('metric_calls', close(sum(calls_d[d]) / len(calls_d[d]), mt['costs'][k]['calls'][str(d)]), f'{m} {t} {k}')
                        note('metric_tokens', close(sum(tok_d[d]) / len(tok_d[d]), mt['costs'][k]['tokens'][str(d)]), f'{m} {t} {k}')
                if k == 'I':
                    pm = [sum(ptok_d[d]) / len(ptok_d[d]) for d in DS]
                    note('sanity_length_constant', max(pm) / min(pm) < 1.10, f'{m} {t}: {min(pm):.0f}..{max(pm):.0f}')
                    top = max(collections.Counter(gold_d[d]).most_common(1)[0][1] / len(gold_d[d]) for d in DS)
                    note('sanity_gold_not_degenerate', top < 0.35 or C.SMOKE, f'{m} {t}: top share {top:.2f}')
    fails = [k for k, v in checks.items() if not v['pass']]
    out = {'pass': not fails and len(checks) > 0, 'failed': fails, 'checks': checks, 'smoke': C.SMOKE,
           'wall_s': time.time() - t0, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S')}
    os.makedirs(C.ANA, exist_ok=True)
    json.dump(out, open(f'{C.ANA}/verify.json.tmp', 'w'), indent=1)
    os.replace(f'{C.ANA}/verify.json.tmp', f'{C.ANA}/verify.json')
    print('VERIFY pass' if out['pass'] else f'VERIFY FAIL {fails}', {k: v['n'] for k, v in checks.items()})


if __name__ == '__main__':
    main()
