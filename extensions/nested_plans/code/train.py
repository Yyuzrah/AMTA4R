"""Train one model from scratch on NestPlan plans of depth d ~ U{1..d_train} (implicit format) and evaluate it on the
depths d = 1..4*d_train+8 at constant length 194: each plan in one call, and each plan split into calls with every
per-call budget D in 2..d_train (analyze.py assembles the calibrated split from these).

Restarts:
  <root>/<name>/results.json exists  -> exit immediately (run finished)
  <root>/<name>/final.pt exists      -> skip training, (re)run evaluation
  <root>/<name>/ckpt.pt exists       -> resume training from it
Once --time_budget_s seconds have passed, the run saves ckpt.pt at the next checkpoint and exits with code 3; the same
command then resumes it. A resumed run draws its training plans from a new stream keyed by the resume step
(common.stream_key).
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nestplan import Vocab, make_example, make_testset, recency_baseline, PAD
from handoff import split_calls, materialize, compose
import common as C


def build_model(pe, L, d, h, vocab_size, max_pos):
    # Same model families as the main experiments (src/train_flexible.py): RoPE -> GPT-NeoX, APE -> GPT-Neo.
    if pe == 'rope':
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        cfg = GPTNeoXConfig(vocab_size=vocab_size, hidden_size=d, num_hidden_layers=L,
                            num_attention_heads=h, intermediate_size=4 * d, hidden_act='gelu',
                            max_position_embeddings=max_pos, rotary_pct=1.0,
                            use_parallel_residual=False, attention_bias=True,
                            bos_token_id=1, eos_token_id=2, pad_token_id=PAD,
                            attn_implementation='sdpa', use_cache=True)
        return GPTNeoXForCausalLM(cfg)
    if pe == 'ape':
        # global attention in every layer and learned absolute positions, as in src/train_flexible.py
        from transformers import GPTNeoConfig, GPTNeoForCausalLM
        cfg = GPTNeoConfig(vocab_size=vocab_size, hidden_size=d, num_layers=L, num_heads=h,
                           intermediate_size=4 * d, hidden_act='gelu', max_position_embeddings=max_pos,
                           attention_types=[[['global'], L]], window_size=256,
                           bos_token_id=1, eos_token_id=2, pad_token_id=PAD,
                           attn_implementation='eager', use_cache=True)
        return GPTNeoForCausalLM(cfg)
    raise ValueError(pe)


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters()
              if any(s in n for s in ['embed_in', 'embed_out', 'wte', 'wpe', 'lm_head']))
    return tot, tot - emb


def collate(exs):
    T = max(len(e['ids']) for e in exs)
    B = len(exs)
    ids = np.zeros((B, T), dtype=np.int64)
    tgt = np.zeros((B, T), dtype=np.int64)
    for i, e in enumerate(exs):
        n = len(e['ids'])
        ids[i, :n] = e['ids']
        tgt[i, :n] = e['tgt']
    ids = torch.from_numpy(ids)
    tgt = torch.from_numpy(tgt)
    labels = torch.where(tgt > 0, ids, torch.full_like(ids, -100))
    return ids, labels


class Stream(torch.utils.data.IterableDataset):
    """Fresh i.i.d. plans; depth d ~ Uniform{1..dmax}; N fixed (constant length)."""

    def __init__(self, N, k, dmax, B, seed, start_step, vocab):
        self.N, self.k, self.dmax, self.B = N, k, dmax, B
        self.seed, self.start_step, self.vocab = seed, start_step, vocab

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        rng = random.Random(C.stream_key(self.dmax, self.seed, self.start_step, wid))
        while True:
            exs = [make_example(rng, self.N, rng.randint(1, self.dmax), self.k, 'implicit', self.vocab)
                   for _ in range(self.B)]
            yield collate(exs)


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def tf_predict(model, exs, bs, dev, sort=True):
    """Teacher-forced: pred[i][p] = argmax at position p-1 (prediction of token p). Right padding only
    (causal model, default position ids), so padding never affects real positions."""
    order = sorted(range(len(exs)), key=lambda i: len(exs[i]['ids'])) if sort else list(range(len(exs)))
    out = [None] * len(exs)
    model.eval()
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        chunk = [exs[j] for j in idx]
        ids, _ = collate(chunk)
        ids = ids.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids, use_cache=False).logits
        pred = logits.float().argmax(-1).cpu().numpy()
        for row, (j, e) in enumerate(zip(idx, chunk)):
            n = len(e['ids'])
            pr = np.full(n, -1, dtype=np.int64)
            pr[1:] = pred[row, :n - 1]
            out[j] = pr
    model.train()
    return out


def score(exs, preds, vocab, with_baseline=True):
    """Single-call metrics per plan depth d and per popped-node depth j."""
    per_d, per_j = {}, {}
    succ, ncorr = [], []
    for e, pr in zip(exs, preds):
        d = int(e['depth'])
        ids, tgt = e['ids'].astype(np.int64), e['tgt']
        corr = (pr == ids)
        popm = tgt == 1
        nt = (e['pop_nontriv'] == 1) & popm
        ok = bool(corr[popm].all())
        succ.append(ok); ncorr.append(int(corr[popm].sum()))
        r = per_d.setdefault(d, dict(n=0, seq=0, pop_n=0, pop_c=0, nt_n=0, nt_c=0, rec_seq=0, rec_pop_c=0))
        r['n'] += 1; r['seq'] += int(ok)
        r['pop_n'] += int(popm.sum()); r['pop_c'] += int(corr[popm].sum())
        r['nt_n'] += int(nt.sum()); r['nt_c'] += int(corr[nt].sum())
        if with_baseline:
            rb = recency_baseline(ids, 'implicit', vocab)
            rc = [rb[p] == ids[p] for p in np.nonzero(popm)[0]]
            r['rec_seq'] += int(all(rc)); r['rec_pop_c'] += int(sum(rc))
        for p in np.nonzero(popm)[0]:
            j = int(e['pop_depth'][p])
            q = per_j.setdefault(j, [0, 0, 0, 0])
            q[0] += 1; q[1] += int(corr[p])
            if e['pop_nontriv'][p] == 1:
                q[2] += 1; q[3] += int(corr[p])
    out_d = {}
    for d in sorted(per_d):
        r = per_d[d]; n = r['n']
        out_d[d] = dict(n=n, seq=r['seq'] / n, pop_acc=r['pop_c'] / max(1, r['pop_n']),
                        nontriv_pop_acc=(r['nt_c'] / r['nt_n']) if r['nt_n'] else None, nontriv_pop_n=r['nt_n'],
                        recency_seq=r['rec_seq'] / n if with_baseline else None,
                        recency_pop_acc=r['rec_pop_c'] / max(1, r['pop_n']) if with_baseline else None)
    out_j = {j: dict(n=q[0], acc=q[1] / q[0], nontriv_n=q[2], nontriv_acc=(q[3] / q[2]) if q[2] else None)
             for j, q in sorted(per_j.items())}
    return out_d, out_j, np.array(succ, dtype=bool), np.array(ncorr, dtype=np.int16)


def eval_split(model, exs, D, bs, dev):
    """Split every plan into calls with per-call budget D (a plan with depth <= D gives one call, identical to the
    plan). compose() asserts that every pop target is scored exactly once; oracle_ok is True if composing the true
    tokens of all calls solves every plan (part of check G_b of analyze.py)."""
    calls, owner = [], []
    for i, e in enumerate(exs):
        for c in split_calls(e['ids'], D):
            calls.append(materialize(c, e['ids'])); owner.append(i)
    preds = tf_predict(model, calls, bs, dev)
    by_plan = [[] for _ in exs]
    for k, i in enumerate(owner):
        by_plan[i].append(k)
    all_ok, own_ok, n_calls, n_tok, max_call_depth = [], [], [], [], []
    c_plan, c_depth, c_ok, c_len = [], [], [], []
    oracle_ok = True
    for i, e in enumerate(exs):
        ks = by_plan[i]
        r = compose(e['ids'], [calls[k] for k in ks], [preds[k] for k in ks])
        ro = compose(e['ids'], [calls[k] for k in ks], [calls[k]['ids'].astype(np.int64) for k in ks])
        oracle_ok &= bool(ro['all_ok'] and ro['own_ok'])
        all_ok.append(r['all_ok']); own_ok.append(r['own_ok'])
        n_calls.append(len(ks)); n_tok.append(sum(len(calls[k]['ids']) for k in ks))
        max_call_depth.append(max(calls[k]['depth'] for k in ks))
        for k, ok in zip(ks, r['call_ok']):
            c_plan.append(i); c_depth.append(calls[k]['depth']); c_ok.append(ok); c_len.append(len(calls[k]['ids']))
    return dict(all_ok=np.array(all_ok, bool), own_ok=np.array(own_ok, bool), n_calls=np.array(n_calls, np.int16),
                n_tok=np.array(n_tok, np.int32), max_call_depth=np.array(max_call_depth, np.int8),
                call_plan=np.array(c_plan, np.int32), call_depth=np.array(c_depth, np.int8),
                call_ok=np.array(c_ok, bool), call_len=np.array(c_len, np.int16)), oracle_ok


def save_atomic(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json_atomic(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--pe', choices=['rope', 'ape'], required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--h', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--N', type=int, default=C.N)
    ap.add_argument('--k', type=int, default=C.K)
    ap.add_argument('--dtrain', type=int, required=True)
    ap.add_argument('--dtest', type=int, default=None, help='default 4*dtrain+8')
    ap.add_argument('--n_test', type=int, default=C.N_TEST)
    ap.add_argument('--n_cal', type=int, default=C.N_CAL)
    ap.add_argument('--n_val', type=int, default=C.N_VAL)
    ap.add_argument('--val_every', type=int, default=2500)
    ap.add_argument('--ckpt_every', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--time_budget_s', type=float, default=5 * 3600)
    ap.add_argument('--eval_bs', type=int, default=1000)
    ap.add_argument('--grid_idx', type=int, default=-1)
    args = ap.parse_args()
    if args.dtest is None:
        args.dtest = C.dtest_of(args.dtrain)
    assert args.dtest <= args.N

    t_start = time.time()
    rdir = os.path.join(args.root, args.name)
    os.makedirs(rdir, exist_ok=True)
    res_path = os.path.join(rdir, 'results.json')
    if os.path.exists(res_path):
        print('results exist, nothing to do:', res_path)
        return 0
    cfg = vars(args).copy()
    cfg_path = os.path.join(rdir, 'config.json')
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for kk in ['pe', 'L', 'd', 'h', 'seed', 'steps', 'bs', 'lr', 'N', 'k', 'dtrain', 'dtest', 'n_test', 'n_cal']:
            assert old[kk] == cfg[kk], f'config mismatch on {kk}: {old[kk]} vs {cfg[kk]}'
    else:
        write_json_atomic(cfg, cfg_path)

    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    vocab = Vocab(k=args.k, n_nodes=args.N)
    max_pos = 256
    assert 4 * args.N + 2 <= max_pos
    torch.manual_seed(args.seed)
    model = build_model(args.pe, args.L, args.d, args.h, vocab.size_padded, max_pos).to(dev)
    n_tot, n_core = count_params(model)
    print(f'{args.name}: params total {n_tot/1e6:.3f}M, non-embedding {n_core/1e6:.3f}M, '
          f'gpu {torch.cuda.get_device_name()}, time budget {args.time_budget_s:.0f}s', flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and 'embed' not in n and 'wte' not in n and 'wpe' not in n else no_decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    final_path = os.path.join(rdir, 'final.pt')
    ckpt_path = os.path.join(rdir, 'ckpt.pt')
    log_path = os.path.join(rdir, 'train_log.jsonl')
    step = 0
    train_seconds = 0.0
    n_resumes = 0
    if os.path.exists(final_path):
        sd = torch.load(final_path, map_location=dev, weights_only=False)
        model.load_state_dict(sd['model'])
        step = args.steps
        train_seconds = sd.get('train_seconds', 0.0)
        n_resumes = sd.get('n_resumes', 0)
        print('final model found; evaluation only', flush=True)
    elif os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(sd['model'])
        opt.load_state_dict(sd['opt'])
        step = sd['step']
        train_seconds = sd.get('train_seconds', 0.0)
        n_resumes = sd.get('n_resumes', 0) + 1
        print(f'resumed from step {step}', flush=True)

    if step < args.steps:
        val_exs = make_testset(C.VAL_SEED, args.N, C.val_depths(args.dtrain), args.n_val, args.k, ['implicit'],
                               vocab)['implicit']
        ds = Stream(args.N, args.k, args.dtrain, args.bs, args.seed, step, vocab)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.workers,
                                         prefetch_factor=4 if args.workers > 0 else None,
                                         persistent_workers=args.workers > 0, pin_memory=True)
        it = iter(dl)
        model.train()
        t_last = time.time()
        tok_count = 0
        loss_acc, n_acc = 0.0, 0
        logf = open(log_path, 'a')
        while step < args.steps:
            ids, labels = next(it)
            ids = ids.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            for g in opt.param_groups:
                g['lr'] = lr_at(step, args.steps, args.lr, args.warmup)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tok_count += ids.numel()
            if step % 50 == 0:
                loss_acc += loss.item(); n_acc += 1
            if step % 250 == 0 or step == args.steps:
                now = time.time()
                dt = now - t_last
                train_seconds += dt
                rec = dict(step=step, loss=loss_acc / max(1, n_acc), lr=opt.param_groups[0]['lr'],
                           grad_norm=float(gn), tok_per_s=tok_count / dt, train_seconds=train_seconds)
                tok_count = 0
                loss_acc, n_acc = 0.0, 0
                if not math.isfinite(rec['loss']):
                    print('non-finite loss; aborting (exit 4)', flush=True)
                    return 4
                if step % args.val_every == 0 or step == args.steps:
                    vp = tf_predict(model, val_exs, args.eval_bs, dev)
                    vd, _, _, _ = score(val_exs, vp, vocab, with_baseline=False)
                    rec['val_seq'] = {d: round(v['seq'], 4) for d, v in vd.items()}
                    rec['val_nontriv_pop_acc'] = {d: (round(v['nontriv_pop_acc'], 4) if v['nontriv_pop_acc'] is not None else None) for d, v in vd.items()}
                logf.write(json.dumps(rec) + '\n'); logf.flush()
                print(json.dumps({k_: rec[k_] for k_ in ['step', 'loss', 'lr', 'tok_per_s', 'train_seconds']}), flush=True)
                if step % args.ckpt_every == 0 and step < args.steps:
                    save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                                     train_seconds=train_seconds, n_resumes=n_resumes), ckpt_path)
                    if time.time() - t_start > args.time_budget_s:
                        print('time budget reached; checkpoint saved; exiting with code 3', flush=True)
                        return 3
                t_last = time.time()
        logf.close()
        save_atomic(dict(model=model.state_dict(), step=step, train_seconds=train_seconds, n_resumes=n_resumes),
                    final_path)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # ---------------- final evaluation ----------------
    t0 = time.time()
    depths = list(range(1, args.dtest + 1))
    sds = C.split_ds(args.dtrain)
    res = dict(name=args.name, config=cfg, params_total=n_tot, params_nonembed=n_core,
               gpu=torch.cuda.get_device_name(), train_seconds=train_seconds, n_resumes=n_resumes,
               split_ds=sds, depths=depths)
    arrays = {}
    oracle_all = True
    for tag, seed, n in [('cal', C.CAL_SEED, args.n_cal), ('test', C.TEST_SEED, args.n_test)]:
        exs = make_testset(seed, args.N, depths, n, args.k, ['implicit'], vocab)['implicit']
        assert all(len(e['ids']) == 4 * args.N + 2 for e in exs), 'matched length violated'
        pr = tf_predict(model, exs, args.eval_bs, dev)
        pd_, pj_, succ, ncorr = score(exs, pr, vocab, with_baseline=(tag == 'test'))
        res[f'{tag}_single'] = dict(per_depth=pd_, per_pop_depth=pj_)
        arrays[f'{tag}_depth'] = np.array([e['depth'] for e in exs], np.int8)
        arrays[f'{tag}_single_ok'] = succ
        arrays[f'{tag}_single_ncorr'] = ncorr
        if tag == 'test':
            for D in sds:
                r, ook = eval_split(model, exs, D, args.eval_bs * 4, dev)
                oracle_all &= ook
                for kk, vv in r.items():
                    arrays[f'test_split{D}_{kk}'] = vv
                per_d = {}
                dep = arrays['test_depth']
                for d in depths:
                    m = dep == d
                    per_d[d] = dict(n=int(m.sum()), seq_all=float(r['all_ok'][m].mean()), seq_own=float(r['own_ok'][m].mean()),
                                    calls=float(r['n_calls'][m].mean()), tokens=float(r['n_tok'][m].mean()))
                cm = r['call_ok']
                res[f'test_split{D}'] = dict(per_depth=per_d, call_success=float(cm.mean()), n_calls_total=int(len(cm)),
                                             max_call_depth=int(r['max_call_depth'].max()),
                                             call_success_by_call_depth={int(c): float(cm[r['call_depth'] == c].mean())
                                                                         for c in np.unique(r['call_depth'])})
                print(f'split D={D}: calls {len(cm)}, call success {cm.mean():.4f}, max call depth '
                      f'{int(r["max_call_depth"].max())}', flush=True)
    res['oracle_composition_ok'] = bool(oracle_all)
    np.savez_compressed(os.path.join(rdir, 'per_plan.npz.tmp.npz'), **arrays)
    os.replace(os.path.join(rdir, 'per_plan.npz.tmp.npz'), os.path.join(rdir, 'per_plan.npz'))
    res['eval_seconds'] = time.time() - t0
    write_json_atomic(res, res_path)
    print('wrote', res_path, 'eval s', round(res['eval_seconds'], 1), flush=True)
    for d in depths:
        v = res['test_single']['per_depth'][d]
        print(d, 'single %.3f' % v['seq'], ' '.join('D%d %.3f' % (D, res[f'test_split{D}']['per_depth'][d]['seq_all']) for D in sds), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
