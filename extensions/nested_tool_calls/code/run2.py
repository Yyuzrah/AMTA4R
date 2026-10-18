"""Inference runner (vLLM, greedy, bf16). One process = one (model, task, unit):
  cal    I on the calibration split -> results/<m>/<task>_I_cal.jsonl and dhat_<task>.json (d_hat* per tau)
  I, E   single calls on the test split -> <task>_I_test.jsonl, <task>_E_test.jsonl
  A      split into calls of D = 1 level on the test split -> <task>_A_test.jsonl
  H      split into calls of D = max(1, d_hat*(tau)) levels for every tau in config.TAUS -> <task>_H_D<D>_test.jsonl
         and budgets_<task>.json (needs dhat_<task>.json of the same model, written by the cal unit)
  rerun  re-run check: RERUN_FRAC of the I_test items re-run in a fresh vLLM process in shuffled order with a
         different batch limit -> <task>_rerun.json (needs <task>_I_test.jsonl)
Usage: python run2.py --model <tag> --task T --unit <unit>. Can be stopped and restarted: each depth is checkpointed
in results/<m>/ckpt/<file>.d<d>.jsonl and skipped on restart; the unit file is written atomically at the end and the
unit is skipped if it exists."""
import argparse, glob, hashlib, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from tasks2 import make_item, build_calls, render, gold_of, extract, KNOWN_STYLE
from models import MODELS
from stats import dhat, dhat_boot


def items_for(task, split, n, depths=None):
    return [make_item(task, split, d, i, M=C.M_FUN, order=C.ORDER, dis_mode=C.DIS_MODE) for d in (depths or C.DEPTHS) for i in range(n)]


class Runner:
    def __init__(self, tag, max_num_seqs=None, prefix_caching=True):
        from vllm import LLM
        meta = json.load(open(f'{C.EXP}/meta/licenses.json'))[tag]
        self.tag, self.meta = tag, meta
        cfg = json.load(open(f"{meta['path']}/config.json"))
        tc = cfg.get('text_config', cfg)
        self.max_model_len = min(C.MAX_MODEL_LEN, int(tc.get('max_position_embeddings') or C.MAX_MODEL_LEN))
        kw = dict(model=meta['path'], dtype='bfloat16', max_model_len=self.max_model_len, seed=0,
                  gpu_memory_utilization=float(os.environ.get('NTC_GPU_UTIL', '0.90')),
                  enable_prefix_caching=prefix_caching, tensor_parallel_size=int(os.environ.get('NTC_TP', '1')))
        if max_num_seqs:
            kw['max_num_seqs'] = max_num_seqs
        self.llm = LLM(**kw)
        self.tok = make_tokenizer(meta['path'])
        self.chat_kwargs = MODELS[tag][2]
        self.ncalls = 0

    def ids(self, user, prefill=''):
        return chat_ids(self.tok, user, prefill, self.chat_kwargs)


    def gen(self, pairs, max_tokens):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        ids = [self.ids(u, p) for u, p in pairs]
        too_long = [len(x) for x in ids if len(x) + max_tokens >= self.max_model_len]
        assert not too_long, f'prompt too long for {self.tag}: {max(too_long)} tokens'
        sp = SamplingParams(temperature=0.0, seed=0, max_tokens=max_tokens)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp, use_tqdm=False)
        self.ncalls += len(pairs)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]


def make_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def chat_ids(tok, user, prefill, chat_kwargs):
    """Token ids of the chat-formatted user turn plus the assistant prefill. Token-level templating (identical to
    text-level templating for every model except Mistral, whose mistral-common tokenizer would otherwise encode the
    control tokens as plain text)."""
    t = tok.apply_chat_template([{'role': 'user', 'content': user}], add_generation_prompt=True, tokenize=True,
                                **chat_kwargs)
    if not isinstance(t, list):
        t = t['input_ids']
    return list(t) + list(tok.encode(prefill, add_special_tokens=False)) if prefill else list(t)


def run_items(R, task, items, cond, D):
    """cond I/E: one call per item. cond ctrl: split into calls of <= D levels (each call rendered as I, with the
    answer of the previous call in place of the definition of the tool it computed).
    Returns one record per item (same order)."""
    plans = [(it, build_calls(it, D if cond == 'ctrl' else None)) for it in items]
    pcond = 'E' if cond == 'E' else 'I'
    answers = [dict() for _ in plans]
    callrecs = [[] for _ in plans]
    maxlev = max(c['level'] for _, cs in plans for c in cs)
    for lev in range(maxlev + 1):
        batch = [(k, it, c) for k, (it, cs) in enumerate(plans) for c in cs if c['level'] == lev]
        prompts = [render(task, it, c, pcond, answers[k]) for k, it, c in batch]
        outs = R.gen(prompts, C.MAXTOK)
        for (k, it, c), (u, p), (o, pt, ot) in zip(batch, prompts, outs):
            ex = extract(o)
            g = gold_of(it, c)
            answers[k][c['id']] = ex if ex is not None else '?'
            callrecs[k].append({'id': c['id'], 'q': c['q'], 'known': c['known'], 'nest': c['nest'], 'main': c['main'],
                                'gold': g, 'ex': ex, 'ok': ex == g, 'strict': o.strip() == g, 'out': o,
                                'ptok': pt, 'otok': ot,
                                'phash': hashlib.md5((u + '\x00' + p).encode()).hexdigest()[:16]})
    recs = []
    for (it, cs), cr in zip(plans, callrecs):
        cr.sort(key=lambda x: x['id'])
        main = cr[-1]
        recs.append({'d': it['d'], 'idx': it['idx'], 'D': D, 'n_calls': len(cr), 'ptok': sum(c['ptok'] for c in cr),
                     'otok': sum(c['otok'] for c in cr), 'ok': main['ok'], 'ok_strict': main['strict'],
                     'all_ok': all(c['ok'] for c in cr), 'calls': cr})
    return recs


def save(path, recs, meta):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(json.dumps({'_meta': meta}) + '\n')
        for r in recs:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, path)


def load(path):
    with open(path) as f:
        L = [json.loads(l) for l in f]
    return L[0]['_meta'], L[1:]


def run_unit_file(get_runner, path, task, split, n, cond, D, meta):
    """Per-depth checkpointed unit; returns records."""
    if os.path.exists(path):
        return load(path)[1]
    ck = os.path.join(os.path.dirname(path), 'ckpt')
    os.makedirs(ck, exist_ok=True)
    allrecs = []
    for d in C.DEPTHS:
        cp = f'{ck}/{os.path.basename(path)}.d{d}'
        if os.path.exists(cp):
            allrecs += load(cp)[1]
            continue
        t0 = time.time()
        recs = run_items(get_runner(), task, items_for(task, split, n, [d]), cond, D)
        save(cp, recs, dict(meta, depth=d, wall_s=time.time() - t0))
        print(f'  {os.path.basename(path)} d={d}: acc={sum(r["ok"] for r in recs) / len(recs):.3f} '
              f'calls={sum(r["n_calls"] for r in recs)} {time.time() - t0:.0f}s', flush=True)
        allrecs += recs
    save(path, allrecs, meta)
    for cp in glob.glob(f'{ck}/{os.path.basename(path)}.d*'):
        os.remove(cp)
    return allrecs


def calibrate(recs):
    by = {d: [1.0 if r['ok'] else 0.0 for r in recs if r['d'] == d] for d in C.DEPTHS}
    acc = [sum(by[d]) / len(by[d]) for d in C.DEPTHS]
    n = [len(by[d]) for d in C.DEPTHS]
    out = {'depths': C.DEPTHS, 'acc': acc, 'n': n}
    for tau in C.TAUS:
        dh, fit = dhat(C.DEPTHS, acc, n, tau)
        lo, hi, _ = dhat_boot(C.DEPTHS, [by[d] for d in C.DEPTHS], tau, reps=C.BOOT)
        out[f'tau{tau}'] = {'dhat': int(dh), 'D': max(1, int(dh)), 'fit': [float(x) for x in fit], 'ci95': [lo, hi]}
    out['included'] = bool(all(acc[C.DEPTHS.index(d)] >= C.TAU for d in C.INCL_DEPTHS))
    return out


def rerun(model, task, out):
    """Re-run check: fresh vLLM process, no prefix caching, max_num_seqs 16, shuffled order."""
    p = f'{out}/{task}_rerun.json'
    if os.path.exists(p):
        return
    _, recs = load(f'{out}/{task}_I_test.jsonl')
    rng = random.Random(f'rerun|{model}|{task}')
    sub = rng.sample(recs, max(1, int(round(len(recs) * C.RERUN_FRAC))))
    rng.shuffle(sub)
    R = Runner(model, max_num_seqs=16, prefix_caching=False)
    items = {(it['d'], it['idx']): it for it in items_for(task, C.SPLIT_TEST, C.N_TEST)}
    pairs, metas = [], []
    for r in sub:
        it = items[(r['d'], r['idx'])]
        c = build_calls(it, None)[-1]
        pairs.append(render(task, it, c, 'I', {}))
        metas.append((it, c, r))
    outs = R.gen(pairs, C.MAXTOK)
    ok_new = ok_old = same = 0
    for (it, c, r), (o, _, _) in zip(metas, outs):
        ok_new += extract(o) == gold_of(it, c)
        ok_old += bool(r['ok'])
        same += o == r['calls'][-1]['out']
    n = len(sub)
    res = {'model': model, 'task': task, 'n': n, 'acc_stored': ok_old / n, 'acc_rerun': ok_new / n,
           'abs_diff': abs(ok_new - ok_old) / n, 'identical_outputs': same / n,
           'pass': abs(ok_new - ok_old) / n <= 0.03, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    json.dump(res, open(p + '.tmp', 'w'), indent=1)
    os.replace(p + '.tmp', p)
    print('rerun', res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', required=True, choices=C.TASKS)
    ap.add_argument('--unit', required=True, choices=['cal', 'I', 'E', 'A', 'H', 'rerun'])
    a = ap.parse_args()
    out = f'{C.RES}/{a.model}'
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    R = [None]

    def runner():
        if R[0] is None:
            R[0] = Runner(a.model)
        return R[0]
    meta = lambda **kw: dict(model=a.model, task=a.task, time=time.strftime('%Y-%m-%dT%H:%M:%S'), M=C.M_FUN,
                             order=C.ORDER, layout=C.DIS_MODE, known_style=KNOWN_STYLE, depths=C.DEPTHS,
                             smoke=C.SMOKE, **kw)
    sc, st = C.SPLIT_CAL, C.SPLIT_TEST
    if a.unit == 'cal':
        recs = run_unit_file(runner, f'{out}/{a.task}_I_cal.jsonl', a.task, sc, C.N_CAL, 'I', None,
                             meta(cond='I', split=sc, n=C.N_CAL))
        pc = f'{out}/dhat_{a.task}.json'
        if not os.path.exists(pc):
            json.dump(calibrate(recs), open(pc + '.tmp', 'w'), indent=1)
            os.replace(pc + '.tmp', pc)
    elif a.unit in ('I', 'E'):
        run_unit_file(runner, f'{out}/{a.task}_{a.unit}_test.jsonl', a.task, st, C.N_TEST, a.unit, None,
                      meta(cond=a.unit, split=st, n=C.N_TEST))
    elif a.unit == 'A':
        run_unit_file(runner, f'{out}/{a.task}_A_test.jsonl', a.task, st, C.N_TEST, 'ctrl', 1,
                      meta(cond='A', split=st, n=C.N_TEST, D=1))
    elif a.unit == 'H':
        cal = json.load(open(f'{out}/dhat_{a.task}.json'))
        bud = {}
        for tau in C.TAUS:
            Dv = cal[f'tau{tau}']['D']
            p = f'{out}/{a.task}_H_D{Dv}_test.jsonl'
            run_unit_file(runner, p, a.task, st, C.N_TEST, 'ctrl', Dv, meta(cond='H', split=st, n=C.N_TEST, D=Dv))
            bud[str(tau)] = {'dhat': cal[f'tau{tau}']['dhat'], 'D': Dv, 'file': os.path.basename(p)}
        json.dump(bud, open(f'{out}/budgets_{a.task}.json', 'w'), indent=1)
    elif a.unit == 'rerun':
        rerun(a.model, a.task, out)
    info = {'model': a.model, 'task': a.task, 'unit': a.unit, 'calls': R[0].ncalls if R[0] else 0,
            'wall_s': time.time() - t0, 'max_model_len': R[0].max_model_len if R[0] else None,
            'sha': json.load(open(f'{C.EXP}/meta/licenses.json'))[a.model]['sha'], 'smoke': C.SMOKE,
            'host': os.uname().nodename, 'job': os.environ.get('SLURM_JOB_ID')}
    os.makedirs(f'{out}/info', exist_ok=True)
    json.dump(info, open(f'{out}/info/{a.task}_{a.unit}_{int(time.time())}.json', 'w'), indent=1)
    print('UNIT DONE', info, flush=True)


if __name__ == '__main__':
    main()
