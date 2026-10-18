"""Second check of finished runs (part of the checks G_b and (c) of analyze.py).
For each run: reload final.pt; regenerate the test plans with their trees directly from the generator (gen_plan with
the same rng stream as make_example); build the split calls with a second, tree-based implementation (new calls start
at the nodes with children at absolute depth D + k(D-1); each call covers depth(root)..depth(root)+D-1); score the
targets with nestplan.oracle_targets (recomputed from the tokens); run fresh forward passes with a different batch
size and no length sorting, one depth at a time. Compares per-depth single-call and always-split success with
results.json.
Free running: on a subset (20 plans per depth), KV-cached decoding in which the model's own pop predictions are fed
back must give the same per-plan sequence success as teacher forcing.
Usage: python verify.py RUN_DIR [RUN_DIR ...]   (writes RUN_DIR/verify.json; skips runs that already have one)"""
import json, os, sys, random, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nestplan import Vocab, gen_plan, oracle_targets, make_example, BOS, EOS, OPEN, CLOSE
from handoff import split_calls, materialize
from train import build_model
import common as C

TOL = 0.01
FR_MIN = 0.99      # minimum free-run agreement (bf16 near-ties can change an argmax); disagreements are listed
N_CROSS = 25       # plans per depth on which the two splitting implementations are compared token by token
N_FREE = 20        # plans per depth for the free-run check


def plan_with_tree(seed, N, d, i, k, vocab):
    rng = random.Random(f'{seed}-{d}-{i}')
    parent, ndepth, children = gen_plan(rng, N, d)
    types = [rng.randrange(k) for _ in range(N)]
    rng2 = random.Random(f'{seed}-{d}-{i}')
    e = make_example(rng2, N, d, k, 'implicit', vocab)   # same stream -> same plan; used only for cross-checks
    return parent, ndepth, children, types, e['ids']


def tree_calls(ndepth, children, types, D, vocab):
    """Second implementation of the split. Returns a list of token arrays (targets filled with the true type)."""
    T0 = vocab.type0
    dmax = max(ndepth)
    if D is None or dmax <= D:
        cut = set()
    else:
        cut = {u for u in range(len(ndepth)) if children[u] and ndepth[u] >= D and (ndepth[u] - D) % (D - 1) == 0}
    calls = []

    def ser(u, out, is_root):
        out += [OPEN, T0 + types[u]]
        if u in cut and not is_root:
            roots.append(u)
        else:
            for c in children[u]:
                ser(c, out, False)
        out += [CLOSE, T0 + types[u]]

    roots = []
    out = [BOS]
    for u in children[-1]:
        ser(u, out, False)
    calls.append(out + [EOS])
    while roots:
        r = roots.pop(0)
        out = [BOS]
        ser(r, out, True)
        calls.append(out + [EOS])
    return [np.asarray(c, dtype=np.int64) for c in calls]


@torch.no_grad()
def predict(model, seqs, bs, dev):
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        T = max(len(s) for s in chunk)
        ids = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.from_numpy(s)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(input_ids=ids.to(dev), use_cache=False).logits
        pr = lg.float().argmax(-1).cpu().numpy()
        for j, s in enumerate(chunk):
            p = np.full(len(s), -1, dtype=np.int64); p[1:] = pr[j, :len(s) - 1]; out.append(p)
    return out


def seq_ok(s, p, vocab):
    orc = oracle_targets(s, 'implicit', vocab)
    return all(int(p[q]) == v for q, v in orc.items())


@torch.no_grad()
def free_run_ok(model, seqs, dev, vocab, bs=320):
    """Batched KV-cached free running (all single-call sequences have the same length): at every scored position
    the model's own argmax is fed back; all other tokens come from the plan."""
    res = []
    for b0 in range(0, len(seqs), bs):
        chunk = seqs[b0:b0 + bs]
        T = len(chunk[0]); assert all(len(x) == T for x in chunk)
        ids = torch.from_numpy(np.stack(chunk)).to(dev)
        sc = torch.zeros_like(ids, dtype=torch.bool)
        for j, x in enumerate(chunk):
            for q in oracle_targets(x, 'implicit', vocab):
                sc[j, q] = True
        cur = ids.clone(); ok = torch.ones(len(chunk), dtype=torch.bool, device=dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(input_ids=cur[:, :1], use_cache=True)
        pkv, last = o.past_key_values, o.logits[:, -1].float()
        for t in range(1, T):
            p = last.argmax(-1)
            ok &= torch.where(sc[:, t], p == ids[:, t], torch.ones_like(ok))
            cur[:, t] = torch.where(sc[:, t], p, ids[:, t])
            if t == T - 1:
                break
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(input_ids=cur[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv, last = o.past_key_values, o.logits[:, -1].float()
        res += ok.cpu().tolist()
    return res


def verify_run(rdir, dev):
    t0 = time.time()
    res = json.load(open(os.path.join(rdir, 'results.json'))); c = res['config']
    vocab = Vocab(k=c['k'], n_nodes=c['N'])
    model = build_model(c['pe'], c['L'], c['d'], c['h'], vocab.size_padded, 256).to(dev)
    model.load_state_dict(torch.load(os.path.join(rdir, 'final.pt'), map_location=dev, weights_only=False)['model'])
    model.eval()
    depths = list(range(1, c['dtest'] + 1))
    sds = C.split_ds(c['dtrain'])
    arms = ['single'] + [f'split{D}' for D in sds]
    controller_ok = True
    controller_issues = []
    per = {a: {} for a in arms}
    fr_seqs, fr_tf = [], []
    for d in depths:
        seqs, meta = [], []
        for i in range(c['n_test']):
            parent, ndepth, children, types, ids = plan_with_tree(C.TEST_SEED, c['N'], d, i, c['k'], vocab)
            s0 = tree_calls(ndepth, children, types, None, vocab)[0]
            if not np.array_equal(s0, ids.astype(np.int64)):
                controller_ok = False; controller_issues.append(('single', d, i))
            seqs.append(s0); meta.append(('single', i))
            for D in sds:
                tc = tree_calls(ndepth, children, types, D, vocab)
                if max(max_depth(x) for x in tc) > D:
                    controller_ok = False; controller_issues.append((f'depth>{D}', d, i))
                if i < N_CROSS:   # compare the two splitting implementations directly
                    hc = [materialize(x, ids)['ids'].astype(np.int64) for x in split_calls(ids, D)]
                    if sorted(map(tuple, tc)) != sorted(map(tuple, hc)):
                        controller_ok = False; controller_issues.append((f'split{D}', d, i))
                for s in tc:
                    seqs.append(s); meta.append((f'split{D}', i))
        preds = predict(model, seqs, 333, dev)
        agg = {}
        for (arm, i), s, p in zip(meta, seqs, preds):
            ok = seq_ok(s, p, vocab)
            agg[(arm, i)] = agg.get((arm, i), True) and ok
        for arm in arms:
            per[arm][d] = float(np.mean([agg[(arm, i)] for i in range(c['n_test'])]))
        for i in range(min(N_FREE, c['n_test'])):
            fr_seqs.append(seqs[meta.index(('single', i))]); fr_tf.append(agg[('single', i)])
    diffs = {}
    for arm in arms:
        for d in depths:
            ref = (res['test_single']['per_depth'][str(d)]['seq'] if arm == 'single'
                   else res[f'test_{arm}']['per_depth'][str(d)]['seq_all'])
            diffs[f'{arm}@{d}'] = abs(per[arm][d] - ref)
    fr = free_run_ok(model, fr_seqs, dev, vocab)
    dis = [k for k, (a, b) in enumerate(zip(fr, fr_tf)) if a != b]
    fr_agree = 1.0 - len(dis) / max(1, len(fr))
    mx = float(max(diffs.values()))
    rec = dict(name=res['name'], pe=c['pe'], dtrain=c['dtrain'], dtest=c['dtest'], arms=arms,
               per_depth={a: {str(d): v for d, v in per[a].items()} for a in arms},
               max_abs_diff=mx, worst=max(diffs, key=diffs.get), controller_ok=bool(controller_ok),
               controller_issues=controller_issues[:20], free_run_agreement=fr_agree, n_free_run=len(fr),
               free_run_disagreements=[(k // N_FREE + 1, k % N_FREE) for k in dis][:50], tol=TOL, fr_min=FR_MIN,
               gpu=torch.cuda.get_device_name(), seconds=time.time() - t0,
               **{'pass': bool(controller_ok and mx <= TOL and fr_agree >= FR_MIN)})
    return rec


def max_depth(s):
    dd = mx = 0
    for t in s:
        t = int(t)
        if t == OPEN:
            dd += 1; mx = max(mx, dd)
        elif t == CLOSE:
            dd -= 1
    return mx


def main():
    dev = 'cuda'
    n_done = 0
    for rdir in sys.argv[1:]:
        outp = os.path.join(rdir, 'verify.json')
        if os.path.exists(outp):
            print('already verified:', rdir, flush=True); n_done += 1
            continue
        if not os.path.exists(os.path.join(rdir, 'results.json')):
            print('no results.json, skipped:', rdir, flush=True)
            continue
        rec = verify_run(rdir, dev)
        tmp = outp + '.tmp'
        json.dump(rec, open(tmp, 'w'), indent=1); os.replace(tmp, outp)
        n_done += 1
        print(rec['name'], 'max |diff|', round(rec['max_abs_diff'], 4), '(', rec['worst'], ') controller_ok',
              rec['controller_ok'], 'fr_agree', rec['free_run_agreement'], 'pass', rec['pass'],
              's', round(rec['seconds'], 1), flush=True)
    print('verified runs:', n_done)


if __name__ == '__main__':
    main()
