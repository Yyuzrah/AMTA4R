"""Unit tests of the item generator, renderer, split into calls and extractor (CPU only). Usage: python test_tasks2.py [n_rep]
Exit code 0 iff every check passes."""
import math, re, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import config as C
from tasks2 import make_item, build_calls, render, gold_of, extract, demos, fingerprint, DEMO_NAMES
from names import NAMES

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def text_eval(task, txt):
    """Tiny independent evaluator of the real-question part of a prompt: returns (value, implicit nesting depth)."""
    real = txt.rsplit('Now the real question.', 1)[1]
    if task == 'T':
        defs = {m.group(1): (m.group(3), int(m.group(4))) for m in
                re.finditer(r'^def ([a-z]+)\(x\): return (([a-z]+)\(x\)|x) \+ (\d)$', real, re.M)}
        defs = {k: (None if v[0] is None else v[0], v[1]) for k, v in defs.items()}
        q = re.search(r'Question: What does ([a-z]+)\((\d+)\) return\?', real)
        kn = re.search(r'^def ([a-z]+)\(x\): return (-?\d+)  # result for x = (\d+), returned by a helper call$', real, re.M)
        if kn:
            kn = (None, kn.group(1), kn.group(3), kn.group(2))
    else:
        defs = {}
        for m in re.finditer(r'^- ([a-z]+): (?:needs ([a-z]+) first, then (\d) more steps?|takes (\d) steps?)\.$', real, re.M):
            defs[m.group(1)] = (m.group(2), int(m.group(3) or m.group(4)))
        q = re.search(r'The agent starts ([a-z]+) at step (\d+)\.', real)
        kn = re.search(r'^- ([a-z]+): already done by a helper agent; started at step (\d+), it was finished at step (-?\d+)\.$',
                       real, re.M)
        if kn:
            kn = (None, kn.group(1), kn.group(2), kn.group(3))
    known = {}
    if kn:
        known[kn[1]] = int(kn[3])
    f, x = q.group(1), int(q.group(2))
    depth = 0
    total = 0
    while True:
        if f in known:
            return known[f] + total, depth, len(defs)
        cal, c = defs[f]
        total += c
        depth += 1
        if cal is None:
            return x + total, depth, len(defs)
        f = cal


for task in C.TASKS:
    lens = {}
    for d in C.DEPTHS:
        for i in range(NREP):
            it = make_item(task, 'unittest', d, i, M=C.M_FUN, order=C.ORDER, dis_mode=C.DIS_MODE)
            nm = [x for x in it['names'] if x is not None]
            NT = d + C.N_TOOLS_EXTRA
            check(len(set(nm)) == len(nm) == NT and set(nm) <= set(NAMES), f'names {task} {d} {i}')
            pos = {u: k for k, u in enumerate(it['shown'])}
            check(sorted(it['shown']) == list(range(C.M_FUN + C.N_TOOLS_EXTRA)), 'shown is a permutation')
            for u, v in it['callee'].items():
                if v is not None:
                    check(pos[v] < pos[u], f'callee-first order violated {task} {d} {i}')
                    check(u < d or v >= d, f'distractor calls a chain tool {task} {d} {i}')
            # chain depth from the query
            u, k = 0, 0
            while u is not None:
                k += 1
                u = it['callee'][u]
            check(k == d, f'chain depth {k} != {d}')
            check(all(c in C.CONST if hasattr(C, 'CONST') else c in (1, 2, 3) for c in it['const'].values()), 'const')
            check(it['val'][0] == it['x0'] + sum(it['const'][j] for j in range(d)), 'gold sum')
            # I prompt: independent text evaluation
            c0 = build_calls(it, None)[-1]
            txt, pre = render(task, it, c0, 'I')
            v, dep, nd = text_eval(task, txt)
            check(str(v) == gold_of(it, c0) and dep == d and nd == NT, f'I text eval {task} {d} {i}: {v} {dep} {nd}')
            real = txt.rsplit('Now the real question.', 1)[1]
            nlines = sum(1 for ln in real.split('\n') if ln.startswith(('def ', '- ', '# note')))
            check(nlines == C.M_FUN + C.N_TOOLS_EXTRA, f'constant registry length {task} {d}: {nlines}')
            check(pre == 'Answer:', 'prefill')
            lens.setdefault(d, []).append(len(re.findall(r"\w+|[^\w\s]", txt)))   # crude token count
            txtE, _ = render(task, it, c0, 'E')
            check(txtE.count(' -> ') >= 4 * (2 - 1) and ('[base]' in txtE.rsplit('Now the real question.', 1)[1]), 'E chain line')
            # split into calls for every budget D
            for D in range(1, C.M_FUN + 1):
                calls = build_calls(it, D)
                n_exp = 1 if d <= D else math.ceil(d / D)
                check(len(calls) == n_exp, f'n calls {task} d={d} D={D}: {len(calls)} != {n_exp}')
                check(calls[-1]['main'] and calls[-1]['q'] == 0 and sum(c['main'] for c in calls) == 1, 'main call')
                check(all(1 <= c['nest'] <= D for c in calls) and sum(c['nest'] for c in calls) == d, 'nest <= D, covers d')
                ans = {}
                for c in calls:
                    t, _ = render(task, it, c, 'I', ans)
                    v, dep, _ = text_eval(task, t)
                    check(str(v) == gold_of(it, c) and dep == c['nest'], f'ctrl text eval {task} d={d} D={D} call {c["id"]}')
                    check(len(t) == len(txt) + (len(t) - len(txt)) and t.count('Now the real question.') == 1, 'ctrl prompt')
                    ans[c['id']] = gold_of(it, c)
                check(ans[calls[-1]['id']] == str(it['val'][0]), 'ctrl final gold')
    m = [sum(v) / len(v) for v in lens.values()]
    check(max(m) / min(m) < 1.06, f'prompt length not constant over depth ({task}): {min(m):.0f}..{max(m):.0f}')
# demos: correct and outside the pool
for dm, kn in demos():
    check(set(dm['names']) <= set(DEMO_NAMES), 'demo names')
# splits never collide
fp = {}
for split in ('cal', 'test', 'pilot2', 'smokecal', 'smoketest'):
    for task in C.TASKS:
        for d in (1, 2, 3):
            for i in range(50):
                f = (task, fingerprint(make_item(task, split, d, i, M=C.M_FUN, order=C.ORDER, dis_mode=C.DIS_MODE)))
                check(f not in fp or fp[f] == split, f'split collision {split} {fp.get(f)}')
                fp[f] = split
# extractor
for s, e in [(' 12', '12'), (' 12.', '12'), ('12\n', '12'), (' The answer is 7', '7'), (' x', None), (' -3', '-3')]:
    check(extract(s) == e, f'extract {s!r}')
print(f'{len(fails)} failures' + (': ' + '; '.join(fails[:20]) if fails else ''))
sys.exit(1 if fails else 0)
