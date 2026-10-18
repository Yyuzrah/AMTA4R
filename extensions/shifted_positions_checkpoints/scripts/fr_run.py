"""Free-running conditions from the model's own prefix (f1, f2). Stages: base (unedited run) and cond (edited
continuations; site rule strict or prefix). Resumes a partial output file (rows keyed by problem and condition) and
skips a stage whose <out>.done file exists."""
import json, os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, OUT, DATA, greedy, bos_id, DEV
from tk2 import find_sites, unit_text, metric_fn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--stage', required=True, choices=['base', 'cond'])
ap.add_argument('--rule', default='strict', choices=['strict', 'prefix'])
ap.add_argument('--n_per_set', type=int, default=1000)
ap.add_argument('--max_bs', type=int, default=0, help='0 = by GPU memory')
ap.add_argument('--chunk', type=int, default=2048)
ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=int, default=512)
ap.add_argument('--conds', default='all')
ap.add_argument('--shard', default='')
ap.add_argument('--wait_base', type=int, default=0, help='seconds to wait for the base stage')
a = ap.parse_args()
RES = f'{OUT}/results' if not a.tag else f'{OUT}/smoke'
pref = f'{RES}/fr_{a.model}{a.tag}'
CONDS = ['base_resume', 'short_kc', 'A16', 'A64', 'Akc', 'B16', 'B64', 'Bkc', 'C', 'Dpre', 'Dpost']
MIN_G = 64


def problems():
    P = []
    for vs in ['bfs', 'dfs', 'mixed']:
        d = json.load(open(f'{DATA}/val_{vs}.json'))
        for i, x in enumerate(d[:a.n_per_set]):
            P.append(dict(pid=f'{vs}:{i}', target=x['target'], nums=x['nums']))
    return P


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


tok, model, pe, fam = load(a.model)
if a.max_bs == 0:
    mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else 0
    a.max_bs = (192 if mem > 70 else 96) if a.model == 'f2' else 256
print('max_bs', a.max_bs, flush=True)
EOS = bos_id(tok); BOS = EOS
t0 = time.time()
if a.stage == 'base':
    out = pref + '_base.jsonl'
    if os.path.exists(out + '.done'):
        print('done already'); sys.exit(0)
    have = {r['pid'] for r in read_jsonl(out)}
    P = [p for p in problems() if p['pid'] not in have]
    print(a.model, 'base todo', len(P), 'have', len(have), flush=True)
    f = open(out, 'a')
    for s in range(0, len(P), a.chunk):
        ch = P[s:s + a.chunk]
        rows = []
        for p in ch:
            line = f"Current State: {p['target']}:{p['nums']}, Operations: []\n"
            ids = [BOS] + tok(line, add_special_tokens=False)['input_ids']
            rows.append(dict(ids=ids, pos=list(range(len(ids))), budget=a.budget))
        G = greedy(model, rows, EOS, max_bs=a.max_bs)
        for p, r, g in zip(ch, rows, G):
            line = f"Current State: {p['target']}:{p['nums']}, Operations: []\n"
            sc, st = metric_fn(line + tok.decode(g, skip_special_tokens=True))
            f.write(json.dumps(dict(pid=p['pid'], prompt=r['ids'], gen=g, ok=int(sc > 0), status=st)) + '\n')
        f.flush()
        print('base', s + len(ch), f'{time.time() - t0:.0f}s', flush=True)
    f.close(); open(out + '.done', 'w').write(time.strftime('%F %T'))
    sys.exit(0)

# ---- cond stage ----
w0 = time.time()
while not os.path.exists(pref + '_base.jsonl.done') and time.time() - w0 < a.wait_base:
    time.sleep(20)
base = {r['pid']: r for r in read_jsonl(pref + '_base.jsonl')}
assert os.path.exists(pref + '_base.jsonl.done'), 'base stage not finished'
out = pref + f'_{a.rule}{a.shard}.jsonl'
if a.conds != 'all':
    CONDS = a.conds.split(',')
if os.path.exists(out + '.done'):
    print('done already'); sys.exit(0)
sites_p = pref + f'_{a.rule}_sites.json'
if os.path.exists(sites_p):
    SITES = json.load(open(sites_p))
else:
    SITES = {}
    for pid, b in base.items():
        g = b['gen']
        if EOS in g:
            g = g[:g.index(EOS)]
        own = b['prompt'][1:] + g
        pieces = [tok.decode([t]) for t in own]
        starts, c = [], 0
        for pc in pieces:
            starts.append(c); c += len(pc)
        text = ''.join(pieces)
        npr = len(b['prompt'])
        line_len = len(''.join(pieces[:npr - 1]))
        start2j = {st: jj + 1 for jj, st in enumerate(starts)}  # index in full ids (BOS = 0)
        cand = find_sites(text, scope='full' if a.rule == 'strict' else 'prefix')
        for st in cand:
            j = start2j.get(st['char'])
            if j is None or st['char'] < line_len:
                continue
            ngb = j - npr
            G = a.budget - ngb
            if G < MIN_G:
                continue
            op, r = st['elig'][0]
            U = unit_text(st, op, r)
            SITES[pid] = dict(j=j, char=st['char'], G=G, type=st['type'], unit=U, op=op,
                              u_ids=tok(U, add_special_tokens=False)['input_ids'], prefix_text=text[:st['char']])
            break
    json.dump(SITES, open(sites_p + f'.tmp{os.getpid()}', 'w')); os.replace(sites_p + f'.tmp{os.getpid()}', sites_p)
print(a.model, a.rule, 'problems', len(base), 'with site', len(SITES), flush=True)
rows, keys = [], []
done = {(r['pid'], r['cond']) for r in read_jsonl(out)}
for pid, s in SITES.items():
    b = base[pid]
    full = b['prompt'] + b['gen']
    Pids = full[:s['j']]; j = s['j']; G = s['G']; u = s['u_ids']; kc = len(u)
    ar = list(range(j))
    spec = {
        'base_resume': (Pids, ar, G, j), 'short_kc': (Pids, ar, G - kc, j),
        'A16': (Pids, [p + 16 for p in ar], G, j + 16), 'A64': (Pids, [p + 64 for p in ar], G, j + 64),
        'Akc': (Pids, [p + kc for p in ar], G, j + kc),
        'C': (Pids + u, list(range(j + kc)), G, j + kc), 'Dpre': (u + Pids, list(range(j + kc)), G, j + kc),
        'Dpost': (Pids[:1] + u + Pids[1:], list(range(j + kc)), G, j + kc)}
    for c in CONDS:
        if (pid, c) in done:
            continue
        if c[0] == 'B':
            k = {'B16': 16, 'B64': 64, 'Bkc': kc}[c]
            # gap: prefix keeps its positions, omega (the continuation) starts at j + k
            rows.append(dict(ids=Pids, pos=ar, budget=G, gap=k)); keys.append((pid, c))
            continue
        ids, pos, bud, _ = spec[c]
        rows.append(dict(ids=ids, pos=pos, budget=bud, gap=0)); keys.append((pid, c))
print('rows todo', len(rows), flush=True)


@torch.no_grad()
def greedy_gap(rows_):
    """Rows with a position gap: the first generated token gets position (last prefix pos + 1 + gap)."""
    out = [None] * len(rows_)
    by = {}
    for i, r in enumerate(rows_):
        by.setdefault(r['gap'], []).append(i)
    for gap, idx in by.items():
        if gap == 0:
            G_ = greedy(model, [rows_[i] for i in idx], EOS, max_bs=a.max_bs)
        else:
            G_ = greedy_with_gap(model, [rows_[i] for i in idx], gap)
        for i, g in zip(idx, G_):
            out[i] = g
    return out


@torch.no_grad()
def greedy_with_gap(model, rr, gap, device=None):
    device = device or DEV
    # same as common.greedy but the decoded tokens' positions are shifted by +gap
    order = sorted(range(len(rr)), key=lambda i: (len(rr[i]['ids']), rr[i]['budget']))
    res = [None] * len(rr)
    for s0 in range(0, len(order), a.max_bs):
        idx = order[s0:s0 + a.max_bs]
        B = len(idx); P = max(len(rr[i]['ids']) for i in idx)
        ids = torch.full((B, P), EOS, dtype=torch.long); att = torch.zeros((B, P), dtype=torch.long)
        pos = torch.zeros((B, P), dtype=torch.long)
        for r_, i in enumerate(idx):
            n = len(rr[i]['ids']); off = P - n
            ids[r_, off:] = torch.tensor(rr[i]['ids']); att[r_, off:] = 1; pos[r_, off:] = torch.tensor(rr[i]['pos'])
        ids, att, pos = ids.to(device), att.to(device), pos.to(device)
        budget = torch.tensor([rr[i]['budget'] for i in idx], device=device)
        # prefill the whole prompt at its own positions; the generated tokens get their positions shifted by +gap
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
        past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
        last = pos[:, -1].clone() + gap
        gen = []; done_ = budget <= 0
        for step in range(int(budget.max().item())):
            gen.append(torch.where(done_, torch.full_like(nxt, -1), nxt))
            done_ = done_ | (nxt == EOS) | (step + 1 >= budget)
            if bool(done_.all()):
                break
            att = torch.cat([att, torch.ones((B, 1), dtype=torch.long, device=device)], 1)
            last = last + 1
            out = model(input_ids=nxt[:, None], attention_mask=att, position_ids=last[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
        Gm = torch.stack(gen, 1).tolist() if gen else [[] for _ in idx]
        for r_, i in enumerate(idx):
            res[i] = [t for t in Gm[r_] if t >= 0]
        del past, out
    return res


f = open(out, 'a')
for s0 in range(0, len(rows), a.chunk):
    rr = rows[s0:s0 + a.chunk]; kk = keys[s0:s0 + a.chunk]
    Gs = greedy_gap(rr)
    for (pid, c), g in zip(kk, Gs):
        s = SITES[pid]; b = base[pid]
        cont = tok.decode(g, skip_special_tokens=True)
        sc, st = metric_fn(s['prefix_text'] + cont)
        rec = dict(pid=pid, cond=c, type=s['type'], kc=len(s['u_ids']), G=s['G'], ok=int(sc > 0), status=st, n_gen=len(g))
        if c == 'base_resume':
            ref = b['gen'][s['j'] - len(b['prompt']):]
            rec['identical'] = int(g == ref[:len(g)] and len(g) == min(len(ref), s['G']))
        f.write(json.dumps(rec) + '\n')
    f.flush()
    print(a.rule, s0 + len(rr), '/', len(rows), f'{time.time() - t0:.0f}s', flush=True)
f.close()
open(out + '.done', 'w').write(time.strftime('%F %T'))
print('finished', f'{time.time() - t0:.0f}s')
