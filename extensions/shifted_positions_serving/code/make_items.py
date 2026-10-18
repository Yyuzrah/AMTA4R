"""Items and per-model plan (CPU, deterministic).
(1) Copies the five few-shot tasks from <FEWSHOT_DIR>/data/items_<split>.json (written by make_fewshot_items.py; same
prompts, targets and seeds: seeds 1000+ in the `main` split, 900000+ in the small test split `pilot`) and adds the character
position at which the cached prefix ends. (2) Generates the long-context retrieval task (key-value recall) per context
tier: r1k (ctx 1024; prompt <= 700 tokens), r2k (ctx 2048; <= 1700), r4k (ctx >= 4096; <= 3000), 500 items per tier
in `main` (seeds 2000000+), 40 per tier in `pilot` (seeds 2900000+).
(3) Writes the per-model plan: prompt length T and cache split m per item under the model's own tokenizer (default
special tokens), the generation budget per item (max over the distinct tokenizers of the fetched models of the target
length + 8; retrieval 16), and the allowed shifts K per task: {k in KS : max_T + max_new + k <= ctx}.
Usage: make_items.py <base dir> <split: main|pilot> [n_per_task_cap]
Outputs: <base>/data/items_<split>.json, <base>/data/plan_<split>.json and the empty file <base>/data/ITEMS_<split>.done;
exits at once if that file exists."""
import json, os, sys, random, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import MODELS, ORDER, FEWSHOT_DIR, TASKS, FEWSHOT_TASKS, KS, RET_MAX_NEW, ret_tier

WORDS = ('apple bird chair door egg fish goat hat ink jar key lamp moon nest owl pen queen rope sun tree '
         'van wall box cup dog car bed bag bell boat book cake coat corn desk drum duck fan farm fire flag fork '
         'frog gate gift girl hill horse house kite leaf lion map milk mouse nail nose orange pig pipe plate pot '
         'rain ring road rock salt seed sheep ship shoe sock star stone table tent toy train wheel window wing '
         'wolf yard zoo cloud river glass bread cheese paper sand snow wind king piano robot tiger lemon onion '
         'candle bottle button carpet garden ladder mirror pencil pillow rabbit spoon tomato violin wagon').split()
RET_HEAD = 'Each line pairs a code with a word. Complete the last line with the word that belongs to its code.\n\n'


def split_char(it):
    p = it['prompt']
    if it['task'] in ('copy', 'reverse', 'brackets'):
        c = p.rfind('Input: ')
    else:
        c = p.rfind('\n') + 1
    assert 0 < c < len(p), it['id']
    return c


def ret_prompt(recs, q):
    return RET_HEAD + ''.join(f'code {c}: {w}\n' for c, w in recs) + f'code {q}:'


def gen_retrieval(seed, budget, toks):
    rng = random.Random(seed)
    codes = rng.sample(range(10000, 100000), 600)
    recs = [(c, rng.choice(WORDS)) for c in codes]
    depth = rng.random()

    def build(n):
        qi = min(int(depth * n), n - 1)
        return ret_prompt(recs[:n], recs[qi][0]), recs[qi][1], qi

    def ntok(n):
        p = build(n)[0]
        return max(len(t(p)['input_ids']) for t in toks)
    lo, hi = 2, 600
    assert ntok(lo) <= budget
    while lo < hi:                       # largest n with max tokens <= budget
        mid = (lo + hi + 1) // 2
        if ntok(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    p, t, qi = build(lo)
    return {'task': 'retrieval', 'id': f'retrieval_{seed}', 'seed': seed, 'prompt': p, 'target': t, 'slot': None,
            'n_records': lo, 'query_index': qi, 'depth': qi / lo}


def main():
    base, split = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(f'{base}/data', exist_ok=True)
    mark = f'{base}/data/ITEMS_{split}.done'
    if os.path.exists(mark):
        print('done marker exists', mark); return 0
    from transformers import AutoTokenizer
    fetched = {}
    for tag in ORDER:
        p = f'{base}/meta/fetch/{tag}.json'
        if os.path.exists(p):
            fetched[tag] = json.load(open(p))
    missing = [t for t in ORDER if t not in fetched]
    if missing and split == 'main':
        print('WARNING: models not fetched (excluded from the plan):', missing, flush=True)
    toks = {t: AutoTokenizer.from_pretrained(fetched[t]['path']) for t in fetched}
    # tokenizer fingerprints (identical tokenizers counted once for the budget searches)
    probe = 'code 48213: violin\nExploring Operation: 85-69=16, Resulting Numbers: [24, 7]'
    fpr = {t: (type(tk).__name__, len(tk), tuple(tk(probe)['input_ids'])) for t, tk in toks.items()}

    def dedup(tags):
        seen = {}
        for t in sorted(tags):
            seen.setdefault(fpr[t], t)
        return sorted(seen.values())
    uniq = dedup(list(toks))
    print('distinct tokenizers:', uniq, flush=True)

    fewshot = json.load(open(f'{FEWSHOT_DIR}/data/items_{split}.json'))['items']
    items = []
    for task in FEWSHOT_TASKS:
        sel = [dict(x) for x in fewshot if x['task'] == task]
        if cap:
            sel = sel[:cap]
        for x in sel:
            x.pop('prompt_tokens', None); x.pop('max_new_tokens', None)
            x['tier'] = None
            items.append(x)
    n_ret = 500 if split == 'main' else 40
    if cap:
        n_ret = min(n_ret, cap)
    seed0 = 2000000 if split == 'main' else 2900000
    tiers = {}
    for t in fetched:
        tiers.setdefault(ret_tier(fetched[t]['ctx_registry']), []).append(t)
    for ti, ((tier, budget), members) in enumerate(sorted(tiers.items())):
        tk = [toks[t] for t in dedup(members)]          # every distinct tokenizer of the tier's models
        tier_idx = {'r1k': 0, 'r2k': 1, 'r4k': 2}[tier]
        for i in range(n_ret):
            it = gen_retrieval(seed0 + tier_idx * 100000 + i, budget, tk)
            it['tier'] = tier
            it['id'] = f'retrieval_{tier}_{it["seed"]}'
            items.append(it)
        print('retrieval tier', tier, 'budget', budget, 'members', members, flush=True)
    for it in items:
        it['split_char'] = split_char(it)
    # generation budget per item: max over all fetched tokenizers of the target length + 8 (retrieval: 16)
    for it in items:
        if it['task'] == 'retrieval':
            it['max_new_tokens'] = RET_MAX_NEW
        else:
            it['max_new_tokens'] = max(len(toks[t](' ' + it['target'], add_special_tokens=False)['input_ids'])
                                       for t in uniq) + 8
    ids = [it['id'] for it in items]
    assert len(ids) == len(set(ids))
    plan = {'split': split, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'KS': list(KS), 'models': {},
            'missing_models': missing}
    for tag in fetched:
        tk, ctx = toks[tag], fetched[tag]['ctx_registry']
        tier = ret_tier(ctx)[0]
        per_item, per_task = {}, {}
        for it in items:
            if it['task'] == 'retrieval' and it['tier'] != tier:
                continue
            enc = tk(it['prompt'], return_offsets_mapping=True)
            T = len(enc['input_ids'])
            sc = it['split_char']
            m = next(i for i, (s, e) in enumerate(enc['offset_mapping']) if s >= sc and e > s)
            assert 0 < m < T, (tag, it['id'], m, T)
            per_item[it['id']] = [T, m]
            d = per_task.setdefault(it['task'], {'n': 0, 'maxT': 0, 'max_new': 0})
            d['n'] += 1; d['maxT'] = max(d['maxT'], T); d['max_new'] = max(d['max_new'], it['max_new_tokens'])
        for task, d in per_task.items():
            d['K'] = [k for k in KS if d['maxT'] + d['max_new'] + k <= ctx]
            assert d['maxT'] + d['max_new'] <= ctx, (tag, task, d)       # every prompt + generation fits at k = 0
            assert 16 in d['K'], (tag, task, d)                          # the smallest shift (16) fits every task
        plan['models'][tag] = {'ctx': ctx, 'tier': tier, 'tasks': per_task, 'items': per_item,
                               'special_tokens_added': fetched[tag]['special_tokens_added']}
        print(tag, {t: (d['n'], d['maxT'], d['max_new'], d['K']) for t, d in per_task.items()}, flush=True)
    h = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    json.dump({'split': split, 'sha256': h, 'n': len(items), 'items': items},
              open(f'{base}/data/items_{split}.json.tmp', 'w'))
    os.replace(f'{base}/data/items_{split}.json.tmp', f'{base}/data/items_{split}.json')
    plan['items_sha256'] = h
    json.dump(plan, open(f'{base}/data/plan_{split}.json.tmp', 'w'))
    os.replace(f'{base}/data/plan_{split}.json.tmp', f'{base}/data/plan_{split}.json')
    open(mark, 'w').write(f'{h} {len(items)} items, {len(plan["models"])} models {time.strftime("%Y-%m-%dT%H:%M:%S")}\n')
    print('items', len(items), 'sha256', h, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
