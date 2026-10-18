"""Input sequences (CPU): fixed-length token sequences (T tokens) for four input families and three tokenizers.
Selection is deterministic (file order / fixed seeds) and identical across tokenizers: an item is used only if it has
>= T tokens under every tokenizer; the first N such items are kept.
Families: countdown_dfs, countdown_bfs (search_path of data/experiment_500k/val_{dfs,bfs}.json from README step 1),
dyck (synthetic Dyck-4, space-separated brackets), code (the *.py files of the Python 3.10 standard library of the
running interpreter, sorted by name).
Output: results/inputs.json  {family: {"ids": [item ids], "tok": {tokenizer: [[T ids], ...]}}}"""
import json, glob, os, sys, random
from transformers import AutoTokenizer

T = 512
N = 128
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/lemma2_pretrained
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root
OUT = f'{HERE}/results/inputs.json'
TOKS = {'gpt2bpe': 'gpt2', 'neox': 'EleutherAI/pythia-160m', 'smollm2': 'HuggingFaceTB/SmolLM2-360M'}


def dyck(seed, n_sym=1600, max_depth=12):
    rng = random.Random(seed)
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    st, out = [], []
    for _ in range(n_sym):
        if not st or (len(st) < max_depth and rng.random() < 0.5):
            p = rng.randrange(4); st.append(p); out.append(pairs[p][0])
        else:
            out.append(pairs[st.pop()][1])
    while st:
        out.append(pairs[st.pop()][1])
    return ' '.join(out)


def sources():
    d = f'{ROOT}/data/experiment_500k'
    for fam, f in [('countdown_dfs', 'val_dfs.json'), ('countdown_bfs', 'val_bfs.json')]:
        data = json.load(open(f'{d}/{f}'))
        yield fam, [(f'{f}#{i}', e['search_path']) for i, e in enumerate(data)]
    yield 'dyck', [(f'dyck_seed{s}', dyck(s)) for s in range(1000)]
    std = sorted(glob.glob(sys.base_prefix + '/lib/python3.10/*.py'))
    yield 'code', [(os.path.basename(p), open(p, encoding='utf-8').read()) for p in std]


def main():
    if os.path.exists(OUT):
        print('exists', OUT); return
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKS.items()}
    res = {}
    for fam, items in sources():
        ids, tk = [], {k: [] for k in toks}
        for iid, text in items:
            enc = {k: t(text, add_special_tokens=False)['input_ids'] for k, t in toks.items()}
            if all(len(v) >= T for v in enc.values()):
                ids.append(iid)
                for k in toks:
                    tk[k].append(enc[k][:T])
            if len(ids) == N:
                break
        print(fam, 'kept', len(ids), 'of scanned', items.index(next(x for x in items if x[0] == ids[-1])) + 1, 'available', len(items), flush=True)
        assert len(ids) == N, fam
        res[fam] = {'ids': ids, 'tok': tk}
    res['_meta'] = {'T': T, 'N_per_family': N, 'tokenizers': TOKS, 'add_special_tokens': False}
    json.dump(res, open(OUT, 'w'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
