"""Teacher-forced conditions on reference traces. One JSONL record per item; resumes a partial output file and skips
a model whose <out>.done file exists."""
import json, os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, OUT, MODELS, route_logits, install_capture, CAPTURE
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--limit_per_type', type=int, default=None)
ap.add_argument('--n_l0', type=int, default=100, help='items per type for the layer-0 check')
ap.add_argument('--out', default=None)
a = ap.parse_args()
fam = 'neox' if a.model.startswith('py') else 'gpt2bpe'
out = a.out or f'{OUT}/results/tf_{a.model}.jsonl'
done_flag = out + '.done'
if os.path.exists(done_flag):
    print('done already', out); sys.exit(0)
D = json.load(open(f'{OUT}/results/items_{fam}.json'))
items = D['items']
if a.limit_per_type is not None:
    cnt, sel = {}, []
    for z in items:
        if cnt.get(z['type'], 0) < a.limit_per_type:
            sel.append(z); cnt[z['type']] = cnt.get(z['type'], 0) + 1
    items = sel
done = set()
if os.path.exists(out):
    for ln in open(out):
        try:
            r = json.loads(ln); done.add((r['set'], r['idx']))
        except Exception:
            pass
tok, model, pe, _ = load(a.model)
install_capture()
from common import DEV
dev = DEV
print(a.model, pe, 'items', len(items), 'done', len(done), flush=True)
l0_cnt = {}
t0 = time.time()
f = open(out, 'a')
for n, z in enumerate(items):
    key = (z['set'], z['idx'])
    if key in done:
        continue
    ids = z['ids']; T = len(ids); j = z['site']; u = z['u_ids']; kc = len(u); rt = z['route']; gold = torch.tensor(z['gold'])
    base_pos = list(range(T))
    same = {'base': base_pos}
    for tag, k in [('16', 16), ('64', 64), ('kc', kc)]:
        same[f'A{tag}'] = [p + k for p in base_pos]
        same[f'B{tag}'] = [p + (k if p >= j else 0) for p in base_pos]
    names = list(same)
    ids_t = torch.tensor([ids] * len(names), device=dev)
    pos_t = torch.tensor([same[c] for c in names], device=dev)
    g = [t - 1 for t in rt]
    L = route_logits(model, ids_t, pos_t, [g] * len(names))
    res = {}
    base_L = L[0].cpu()
    btop = base_L.argmax(-1)
    s2 = base_L.topk(2, -1).values
    margin = (s2[:, 0] - s2[:, 1]).tolist()
    for c, lg in zip(names, L):
        if c == 'base':
            continue
        lg = lg.cpu(); top = lg.argmax(-1)
        res[c] = dict(flip=(top != btop).int().tolist(), ok=(top == gold).int().tolist())
        if c.startswith('A'):
            res[c]['maxdlogit'] = float((lg - base_L).abs().max())
    # insertion conditions (length T + kc)
    ins = {'C': ids[:j] + u + ids[j:], 'Dpre': u + ids, 'Dpost': ids[:1] + u + ids[1:]}
    gi = {'C': [t + kc - 1 for t in rt], 'Dpre': [t + kc - 1 for t in rt], 'Dpost': [t + kc - 1 for t in rt]}
    names2 = list(ins)
    ids2 = torch.tensor([ins[c] for c in names2], device=dev)
    pos2 = torch.tensor([list(range(T + kc))] * len(names2), device=dev)
    L2 = route_logits(model, ids2, pos2, [gi[c] for c in names2])
    for c, lg in zip(names2, L2):
        lg = lg.cpu(); top = lg.argmax(-1)
        res[c] = dict(flip=(top != btop).int().tolist(), ok=(top == gold).int().tolist())
    rec = dict(set=z['set'], idx=z['idx'], type=z['type'], search_type=z['search_type'], T=T, site=j, kc=kc,
               dist=[t - j for t in rt], base_ok=(btop == gold).int().tolist(), margin=margin, cond=res)
    # literal Lemma 2: layer-0 pre-softmax scores
    if l0_cnt.get(z['type'], 0) < a.n_l0:
        l0_cnt[z['type']] = l0_cnt.get(z['type'], 0) + 1
        S = {}
        for c, pos in [('base', base_pos), ('B16', same['B16']), ('B64', same['B64']), ('A16', same['A16'])]:
            CAPTURE['on'] = True
            with torch.no_grad():
                model.base_model(input_ids=torch.tensor([ids], device=dev), position_ids=torch.tensor([pos], device=dev), use_cache=False)
            CAPTURE['on'] = False
            S[c] = CAPTURE['scores'][0]  # [H, T, T]
        q = torch.arange(T)[:, None]; kk = torch.arange(T)[None, :]
        causal = kk <= q
        in_w = causal & (q >= j) & (kk >= j)
        cross = causal & (q >= j) & (kk < j)
        scale = S['base'].abs()[:, in_w | cross].mean().item()
        l0 = dict(scale=scale)
        for c in ['B16', 'B64', 'A16']:
            d = (S[c] - S['base']).abs()
            l0[c] = dict(in_mean=d[:, in_w].mean().item(), in_max=d[:, in_w].max().item(),
                         cross_mean=d[:, cross].mean().item(), cross_max=d[:, cross].max().item())
        rec['l0'] = l0
        CAPTURE['scores'] = None
    f.write(json.dumps(rec) + '\n'); f.flush()
    if n % 100 == 0:
        print(a.model, n, f'{time.time() - t0:.0f}s', flush=True)
f.close()
open(done_flag, 'w').write(time.strftime('%F %T'))
print('finished', a.model, f'{time.time() - t0:.0f}s', flush=True)
