"""Build teacher-forced items (CPU): for each reference trace of the three evaluation sets, the first site (scope='full')
within the window, its inert unit, token-level indices and routing-token positions in omega. Deterministic."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tk2 import DATA, find_sites, unit_text, route_spans

W = 896          # window (tokens incl. BOS); W + max(k, k_c) stays < 1024 for the paper models
MIN_SITE_TOK = 16


def build_items(tok, sets=('bfs', 'dfs', 'mixed'), limit_per_type=None, bos=None):
    bos = tok.bos_token if bos is None else bos
    bos_id = tok.convert_tokens_to_ids(bos)
    items, stats = [], dict(n=0, no_site=0, site_out_window=0, tok_mismatch=0, no_route=0)
    cnt = {}
    for vs in sets:
        data = json.load(open(f'{DATA}/val_{vs}.json'))
        for idx, x in enumerate(data):
            stats['n'] += 1
            text = x['search_path'].strip() + '\n'
            sites = find_sites(text, scope='full')
            if not sites:
                stats['no_site'] += 1; continue
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = [bos_id] + enc['input_ids']; offs = [(-1, -1)] + [tuple(o) for o in enc['offset_mapping']]
            ids = ids[:W]; offs = offs[:W]
            start2tok = {o[0]: j for j, o in enumerate(offs) if j > 0}
            chosen = None
            for st in sites:
                j = start2tok.get(st['char'])
                if j is None or j < MIN_SITE_TOK or j > W - 16:
                    continue
                chosen = (st, j); break
            if chosen is None:
                stats['site_out_window'] += 1; continue
            st, j = chosen
            if limit_per_type is not None and cnt.get(st['type'], 0) >= limit_per_type:
                continue
            op, r = st['elig'][0]
            U = unit_text(st, op, r)
            u_ids = tok(U, add_special_tokens=False)['input_ids']
            # tokenization consistency: char-level edit must tokenize to ids[:j] + u_ids + ids[j:]
            text_c = text[:st['char']] + U + text[st['char']:]
            ids_c = [bos_id] + tok(text_c, add_special_tokens=False)['input_ids']
            if ids_c[:len(ids) + len(u_ids)] != ids[:j] + u_ids + ids[j:]:
                stats['tok_mismatch'] += 1; continue
            spans = route_spans(text)
            rt = [t for t in range(j, len(ids)) if any(a <= offs[t][0] < b for a, b in spans)]
            if not rt:
                stats['no_route'] += 1; continue
            items.append(dict(set=vs, idx=idx, type=st['type'], search_type=x['search_type'], ids=ids, site=j,
                              u_ids=u_ids, unit=U, route=rt, gold=[ids[t] for t in rt]))
            cnt[st['type']] = cnt.get(st['type'], 0) + 1
    return items, stats


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from common import MODELS
    for name, path in [('gpt2bpe', MODELS['f1'][0]),
                       ('neox', MODELS['py160m'][0])]:
        tok = AutoTokenizer.from_pretrained(path)
        items, stats = build_items(tok)
        print(name, 'bos', tok.bos_token, stats)
        for ty in ['bfs', 'dfs']:
            it = [z for z in items if z['type'] == ty]
            if not it: continue
            kc = [len(z['u_ids']) for z in it]; nr = [len(z['route']) for z in it]; site = [z['site'] for z in it]
            print(f'  {ty}: n={len(it)} by set', {s: sum(z["set"] == s for z in it) for s in ['bfs', 'dfs', 'mixed']},
                  'k_c pct', np.percentile(kc, [0, 25, 50, 75, 100]).tolist(), 'route/trace med', np.median(nr), 'sum', sum(nr),
                  'site tok pct', np.percentile(site, [10, 50, 90]).tolist(), 'len med', np.median([len(z['ids']) for z in it]))
