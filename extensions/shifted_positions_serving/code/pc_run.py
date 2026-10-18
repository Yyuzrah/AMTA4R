"""Run one work unit = (model, scenario) of the global unit list (registry.units()).
Writes <base>/results/<outroot>/<model>/<scenario>/{checks.json, <task>.jsonl, DONE.json}. Restart-safe: finished
tasks are skipped, an unfinished task resumes from <task>.jsonl.partial (one line per item, flushed). Exit code 3: the
--deadline (unix time; default 5 h after the start) was reached, and the same command continues the unit. Exit code 4:
the model is not fetched or not in the item plan. --strict: every row in its own forward pass, written to
results/rescue/ (for the units that gatecheck.py lists). --split pilot / --cap n: reduced-scale run (items of the small test
split, first n items per task)."""
import argparse, json, os, sys, time, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from registry import MODELS, CHECK_N, WINDOW, units, scen_tasks, scen_precision
import pc_engine as PE


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_items(base, split, plan_m, cap):
    items = json.load(open(f'{base}/data/items_{split}.json'))['items']
    out = {}
    for it in items:
        if it['id'] not in plan_m['items']:
            continue
        out.setdefault(it['task'], []).append(it)
    if cap:
        out = {t: v[:cap] for t, v in out.items()}
    return out


class Runner:
    def __init__(self, a):
        self.a = a
        uid, self.tag, self.scen = units()[a.unit]
        hub, rev, self.cls, self.family, self.ctx, lic, size = MODELS[self.tag]
        self.meta = json.load(open(f'{a.base}/meta/fetch/{self.tag}.json'))
        plan = json.load(open(f'{a.base}/data/plan_{a.split}.json'))
        self.plan = plan['models'][self.tag]
        self.items = load_items(a.base, a.split, self.plan, a.cap)
        self.out = f'{a.base}/results/{a.outroot}/{self.tag}/{self.scen}'
        os.makedirs(self.out, exist_ok=True)
        self.prec = scen_precision(self.scen)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stats = {'chunks': {}, 'oom_events': 0}

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        dt = {'fp32': torch.float32, 'bf16': torch.bfloat16}[self.prec]
        p = self.meta['path']
        self.tok = AutoTokenizer.from_pretrained(p)
        t0 = time.time()
        kw = dict(device_map={'': 0}) if self.dev == 'cuda' else {}       # load straight to the GPU (low host RAM)
        self.model = AutoModelForCausalLM.from_pretrained(p, dtype=dt, attn_implementation='eager', **kw).eval()
        assert self.model.config._attn_implementation == 'eager', self.model.config._attn_implementation
        assert next(self.model.parameters()).dtype == dt
        log('loaded', self.tag, self.prec, f'{time.time() - t0:.0f}s')
        self.lm = PE.PosLM(self.model, self.family)
        V = self.lm.head.weight.shape[0]
        self.nl = PE.stop_mask(self.tok, V, self.dev)          # newline tokens + EOS
        self.nl_cpu = self.nl.cpu().tolist()
        self.ntok = len(self.tok)
        tk = self.tok
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else (tk.eos_token_id if tk.eos_token_id is not None else 0)
        self.elt = torch.finfo(dt).bits // 8
        if self.dev == 'cuda':
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            self.budget = 0.7 * free
            self.gpu = torch.cuda.get_device_name(0)
        else:
            self.budget, self.gpu = 8e9, 'cpu'

    def encode(self, it):
        ids = self.tok(it['prompt'])['input_ids']
        T, m = self.plan['items'][it['id']]
        assert len(ids) == T, (self.tag, it['id'], len(ids), T)
        return ids, m

    def chunk_rows(self, phases, max_new, R):
        """(decode-group size, prefill chunk size) from the memory budget: a decode group holds the KV caches of its
        rows (+ the prefill of one chunk); the prefill chunk is limited by the eager attention scores (q x kv)."""
        if self.a.strict:
            return 1, 1
        q = max(p[0].shape[1] for p in phases)
        kv = sum(p[0].shape[1] for p in phases) + max_new
        pre = self.lm.row_bytes(q, kv, self.elt)
        cache = self.lm.cache_bytes(kv, self.elt)
        # largest decode group whose caches fit next to the prefill of one row, then the largest prefill chunk
        c_dec = int(max(1, min(R, (self.budget - pre) // cache))) if self.budget > pre else 1
        c_pre = int(max(1, min(c_dec, (self.budget - c_dec * cache) // pre))) if self.budget > c_dec * cache else 1
        return c_dec, c_pre

    def run_item_rows(self, phases, max_new, win, R):
        """generate for all rows: decode groups of c_dec rows, each prefilled in chunks of c_pre rows (OOM -> halve
        both and shrink the budget). Returns list of gen lists, first logits."""
        c_dec, c_pre = self.chunk_rows(phases, max_new, R)
        while True:
            try:
                gens, firsts = [], []
                for s in range(0, R, c_dec):
                    rows = list(range(s, min(R, s + c_dec)))
                    g, f = PE.generate(self.lm, PE.slice_phases(phases, rows), max_new, self.nl, win, pre_chunk=c_pre)
                    gens += g.tolist(); firsts.append(f)
                key = f'{c_dec}/{c_pre}'
                self.stats['chunks'][key] = self.stats['chunks'].get(key, 0) + 1
                return gens, torch.cat(firsts, 0)
            except torch.OutOfMemoryError:
                self.stats['oom_events'] += 1
                gens = firsts = None
                torch.cuda.empty_cache()
                if c_dec == 1 and c_pre == 1:
                    raise
                c_dec, c_pre = max(1, c_dec // 2), max(1, c_pre // 2)
                self.budget *= 0.7
                log('OOM -> decode group', c_dec, 'prefill chunk', c_pre)

    # ------------------------------------------------------------------ checks
    def check_items(self):
        out = []
        for t, v in self.items.items():
            if self.scen == 'window' and t not in scen_tasks('window'):
                continue
            out += v[:CHECK_N]
        return out

    @torch.no_grad()
    def check_item(self, it):
        dev, lm = self.dev, self.lm
        out = {'finite': True}
        ids, m = self.encode(it)
        T = len(ids)
        x = torch.tensor(ids, device=dev)[None]
        ar = torch.arange(T, device=dev)[None]
        one = torch.ones(1, T, dtype=torch.bool, device=dev)
        plain = PE.full_logits(lm, [(x, ar, one)], None)            # explicit 0..T-1, 2-D mask of ones
        out['finite'] &= bool(torch.isfinite(plain).all())
        sc = float(plain.abs().max())
        out['scale'] = sc
        if self.scen in ('offset', 'offset_bf16'):
            d = self.model(input_ids=x).logits[0, -1].float()          # default HF forward, no position ids
            v = float((d - plain[0]).abs().max())
            out['G1_default_vs_explicit'], out['G1_rel'] = v, v / sc
            del d
            gap = ar + 64 * (ar >= T // 2)
            g = PE.full_logits(lm, [(x, gap, one)], None)
            v = float((g - plain).abs().max())
            out['G2_gap_maxdiff'], out['G2_rel'] = v, v / sc
        if self.scen == 'window':
            wide = PE.full_logits(lm, [(x, ar, one)], None, window=10 ** 6)
            v = float((wide - plain).abs().max())
            out['W1_wide_band_vs_full'], out['W1_rel'] = v, v / sc
            if T > WINDOW + 16:
                band = PE.full_logits(lm, [(x, ar, one)], None, window=WINDOW)
                v = float((band - plain).abs().max())
                out['W2_band_vs_full'], out['W2_rel'] = v, v / sc
        K = self.plan['tasks'][it['task']]['K']
        labels, phases, win, _ = PE.build_rows(self.scen, ids, m, K, self.pad_id, dev, WINDOW)
        R = len(labels)
        c, c_pre = self.chunk_rows(phases, 3, R)
        agree = []
        for s in range(0, R, c):
            rows = list(range(s, min(R, s + c)))
            ph = PE.slice_phases(phases, rows)
            g, f = PE.generate(lm, ph, 3, torch.zeros_like(self.nl), win, pre_chunk=c_pre)
            if s == 0 and self.scen in ('leftpad', 'cache'):
                v = float((f[0] - plain[0]).abs().max())
                out['path_ref_vs_plain_maxdiff'], out['path_rel'] = v, v / sc
            out['finite'] &= bool(torch.isfinite(f).all())
            if g.shape[1] >= 3:
                for s2 in range(0, len(rows), c_pre):            # full recomputation, prefill-sized chunks
                    sub = list(range(s2, min(len(rows), s2 + c_pre)))
                    lf = PE.full_logits(lm, PE.slice_phases(ph, sub), g[sub, :2], win)
                    agree += (lf.argmax(-1) == g[sub, 2]).float().tolist()
            del g, f
        out['V4_cached_vs_full_agree'] = sum(agree) / max(1, len(agree))
        return out

    @torch.no_grad()
    def checks(self):
        """Per-unit checks on the first CHECK_N items of each task. Every logit difference is
        stored absolute and relative to the item's logit scale max|plain logits| (plain = explicit 0..T-1, one pass)."""
        p = f'{self.out}/checks.json'
        if os.path.exists(p):
            return json.load(open(p))
        dev, lm = self.dev, self.lm
        keys = ('scale', 'G1_default_vs_explicit', 'G1_rel', 'G2_gap_maxdiff', 'G2_rel', 'V4_cached_vs_full_agree',
                'path_ref_vs_plain_maxdiff', 'path_rel', 'W1_wide_band_vs_full', 'W1_rel', 'W2_band_vs_full', 'W2_rel')
        ck = {k: [] for k in keys}
        ck['finite'] = True
        for it in self.check_items():
            for attempt in range(6):
                try:
                    vals = self.check_item(it)
                    break
                except torch.OutOfMemoryError:
                    self.stats['oom_events'] += 1
                    torch.cuda.empty_cache()
                    self.budget *= 0.7
                    log('OOM in checks -> budget', round(self.budget / 2 ** 30, 2), 'GiB')
            else:
                raise RuntimeError('checks: repeated OOM')
            for k, v in vals.items():
                if k == 'finite':
                    ck['finite'] &= v
                else:
                    ck[k].append(v)
        mx = ('G1_default_vs_explicit', 'G1_rel', 'W1_wide_band_vs_full', 'W1_rel', 'path_ref_vs_plain_maxdiff',
              'path_rel')
        mn = ('G2_gap_maxdiff', 'G2_rel', 'W2_band_vs_full', 'W2_rel', 'scale')
        summ = {}
        for k, v in ck.items():
            if isinstance(v, list):
                summ[k] = (max(v) if k in mx else min(v) if k in mn else sum(v) / len(v)) if v else None
            else:
                summ[k] = v
        res = {'summary': summ, 'values': ck, 'n_check_items': len(self.check_items())}
        json.dump(res, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
        log('checks', json.dumps(summ))
        return res

    # ------------------------------------------------------------------ main loop
    def run_task(self, task):
        final = f'{self.out}/{task}.jsonl'
        if os.path.exists(final):
            return 'skip'
        part = final + '.partial'
        done = {}
        if os.path.exists(part):
            for line in open(part):
                try:
                    r = json.loads(line); done[r['id']] = r
                except Exception:
                    pass
        K = self.plan['tasks'][task]['K']
        todo = [it for it in self.items[task] if it['id'] not in done]
        log(task, 'K', K, 'todo', len(todo), 'resumed', len(done))
        t0 = time.time()
        with open(part, 'a') as f:
            for it in todo:
                if time.time() > self.a.deadline:
                    log('time guard: stopping before', it['id']); return 'timeout'
                ids, m = self.encode(it)
                labels, phases, win, posmeta = PE.build_rows(self.scen, ids, m, K, self.pad_id, self.dev, WINDOW)
                gens, first = self.run_item_rows(phases, it['max_new_tokens'], win, len(labels))
                fs = PE.first_stats(first)
                trunc, ans = [], []
                for g in gens:
                    cut = next((i + 1 for i, t in enumerate(g) if self.nl_cpu[t]), len(g))
                    trunc.append(g[:cut]); ans.append(PE.answer(self.tok, g, self.ntok))
                rec = {'id': it['id'], 'task': task, 'T': len(ids), 'm': m, 'K': K, 'labels': labels, 'pos': posmeta,
                       'target': it['target'], 'ans': ans, 'correct': [int(s == it['target']) for s in ans],
                       'gen': trunc, 'first': fs, 'finite': bool(torch.isfinite(first).all()),
                       'steps': max(len(g) for g in gens)}
                f.write(json.dumps(rec) + '\n'); f.flush()
                done[it['id']] = rec
        order = [it['id'] for it in self.items[task]]
        with open(final + '.tmp', 'w') as f:
            for i in order:
                f.write(json.dumps(done[i]) + '\n')
        os.replace(final + '.tmp', final)
        os.remove(part)
        acc = sum(done[i]['correct'][0] for i in order) / len(order)
        self.stats.setdefault('task_sec', {})[task] = round(time.time() - t0, 1)
        log(task, 'done n', len(order), 'acc_ref', round(acc, 3), f'{time.time() - t0:.0f}s')
        return 'done'

    def main(self):
        if os.path.exists(f'{self.out}/DONE.json'):
            log('unit already done', self.out); return 0
        self.load()
        self.checks()
        for task in scen_tasks(self.scen):
            if task not in self.items:
                log('task not in plan for this model:', task); continue
            if self.run_task(task) == 'timeout':
                json.dump(self.stats, open(f'{self.out}/stats_partial.json', 'w'))
                return 3
        import transformers
        n = {t: sum(1 for _ in open(f'{self.out}/{t}.jsonl')) for t in scen_tasks(self.scen) if t in self.items}
        if self.dev == 'cuda':
            self.stats['max_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        done = {'unit': self.a.unit, 'model': self.tag, 'scenario': self.scen, 'precision': self.prec,
                'strict': bool(self.a.strict), 'split': self.a.split, 'cap': self.a.cap, 'n_items': n,
                'K': {t: self.plan['tasks'][t]['K'] for t in n}, 'gpu': self.gpu, 'host': platform.node(),
                'torch': torch.__version__, 'transformers': transformers.__version__, 'sha': self.meta['sha'],
                'stats': self.stats, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'job': os.environ.get('SLURM_JOB_ID'), 'window': WINDOW if self.scen == 'window' else None}
        json.dump(done, open(f'{self.out}/DONE.json.tmp', 'w'), indent=1)
        os.replace(f'{self.out}/DONE.json.tmp', f'{self.out}/DONE.json')
        log('UNIT DONE', self.tag, self.scen, n)
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--split', default='main')
    ap.add_argument('--cap', type=int, default=0)
    ap.add_argument('--unit', type=int, required=True)
    ap.add_argument('--deadline', type=float, default=time.time() + 5 * 3600)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--outroot', default='raw')
    a = ap.parse_args()
    if a.strict and a.outroot == 'raw':
        a.outroot = 'rescue'
    uid, tag, scen = units()[a.unit]
    if not os.path.exists(f'{a.base}/meta/fetch/{tag}.json'):
        log('UNIT NOT RUNNABLE: model not fetched', uid); return 4
    plan = json.load(open(f'{a.base}/data/plan_{a.split}.json'))
    if tag not in plan['models']:
        log('UNIT NOT RUNNABLE: model not in the item plan', uid); return 4
    return Runner(a).main()


if __name__ == '__main__':
    sys.exit(main())
