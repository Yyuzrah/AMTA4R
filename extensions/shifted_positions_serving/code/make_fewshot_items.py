"""Items of the five few-shot exact-match tasks (CPU, deterministic); the same prompt text is used for every model
(tokenized by each model's own tokenizer, no special tokens).
  copy      : repeat a list of 6..24 common words (6 demos)
  reverse   : reverse a list of 2..4 common words (6 demos)
  brackets  : output the closing sequence of a Dyck-4 prefix (open depth 1..5, 0..3 closed pairs) (6 demos)
  arith     : a + b, each operand 1-digit w.p. 0.75 else 2-digit (8 demos)
  countdown : complete the deterministic remainder of a line of a Countdown search trace (val_dfs.json and
              val_bfs.json of data/experiment_500k from step 1 of the repository README, or of $DATA_DIR); slot
              types: (A) "Resulting Numbers: [" -> list, (B) "Generated Node #id:" -> state + operation,
              (C) "Current State:" after "Moving to Node" -> state + operations. The prompt is the trace prefix (a line
              of the same slot type occurs earlier in the prefix).
Answer = generated text up to the first newline, stripped; exact match with the stripped target.
Splits: main (500 items per task) and the small test split (split label pilot, 40 items per task). Item seed = 1000
(main) or 900000 (pilot) + 100000 x task index + a counter that also counts the prompts rejected for length.
Prompts are kept <= MAXTOK tokens under all three tokenizers so that prompt + generation + 256 < 1024 (GPT-2 context).
Output: <FEWSHOT_DIR>/data/items_<split>.json, with FEWSHOT_DIR = $FEWSHOT_DIR or fewshot/ in this experiment's folder
(registry.py)."""
import json, os, random, sys
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import FEWSHOT_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
TOKS = {'gpt2bpe': 'gpt2', 'neox': 'EleutherAI/pythia-160m', 'smollm2': 'HuggingFaceTB/SmolLM2-360M'}
MAXTOK = 640
WORDS = ('apple bird chair door egg fish goat hat ink jar key lamp moon nest owl pen queen rope sun tree '
         'van wall box cup dog car bed bag bell boat book cake coat corn desk drum duck fan farm fire flag fork '
         'frog gate gift girl hill horse house kite leaf lion map milk mouse nail nose orange pig pipe plate pot '
         'rain ring road rock salt seed sheep ship shoe sock star stone table tent toy train wheel window wing '
         'wolf yard zoo cloud river glass bread cheese paper sand snow wind king piano robot tiger lemon onion '
         'candle bottle button carpet garden ladder mirror pencil pillow rabbit spoon tomato violin wagon').split()
PAIRS = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]


def fewshot(rng, gen, n, head):
    demos = [gen(rng) for _ in range(n)]
    q, t = gen(rng)
    body = ''.join(f'Input: {a}\nOutput: {b}\n\n' for a, b in demos)
    return f'{head}\n\n{body}Input: {q}\nOutput:', t


def g_copy(rng):
    w = [rng.choice(WORDS) for _ in range(rng.randint(6, 24))]
    return ' '.join(w), ' '.join(w)


def g_rev(rng):
    w = [rng.choice(WORDS) for _ in range(rng.randint(2, 4))]
    return ' '.join(w), ' '.join(w[::-1])


def g_br(rng):
    d, b = rng.randint(1, 5), rng.randint(0, 3)
    while True:
        ops = ['o'] * (d + b) + ['c'] * b
        rng.shuffle(ops)
        h, ok = 0, True
        for o in ops:
            h += 1 if o == 'o' else -1
            ok &= h >= 0
        if ok:
            break
    st, out = [], []
    for o in ops:
        if o == 'o':
            p = rng.randrange(4); st.append(p); out.append(PAIRS[p][0])
        else:
            out.append(PAIRS[st.pop()][1])
    return ' '.join(out), ' '.join(PAIRS[p][1] for p in reversed(st))


def num(rng):
    return rng.randint(0, 9) if rng.random() < 0.75 else rng.randint(10, 99)


def t_arith(rng):
    def g(r):
        a, b = num(r), num(r)
        return f'{a} + {b} =', str(a + b)
    lines = [g(rng) for _ in range(8)]
    q, t = g(rng)
    return ''.join(f'{a} {b}\n' for a, b in lines) + q, t


def load_traces():
    d = os.environ.get('DATA_DIR', f'{ROOT}/data/experiment_500k')
    tr = []
    for f in ('val_dfs.json', 'val_bfs.json'):
        tr += [e['search_path'] for e in json.load(open(f'{d}/{f}'))]
    return tr


def cd_slots(lines):
    """Candidate (line index, prefix, remainder, type)."""
    out = []
    for j, ln in enumerate(lines):
        if ln.startswith('Exploring Operation:') and ', Resulting Numbers: [' in ln:
            i = ln.index('Resulting Numbers: [') + len('Resulting Numbers: [')
            out.append((j, ln[:i], ln[i:], 'A'))
        elif ln.startswith('Generated Node #') and ': ' in ln:
            i = ln.index(':') + 1
            out.append((j, ln[:i], ln[i:], 'B'))
        elif ln.startswith('Current State:') and j > 0 and lines[j - 1].startswith('Moving to Node'):
            i = len('Current State:')
            out.append((j, ln[:i], ln[i:], 'C'))
    return out


def t_countdown(rng, traces):
    while True:
        lines = rng.choice(traces).rstrip('\n').split('\n')
        sl = cd_slots(lines)
        types = {}
        for s in sl:
            if any(p[3] == s[3] and p[0] < s[0] for p in sl):   # same slot type seen earlier in the prefix
                types.setdefault(s[3], []).append(s)
        if not types:
            continue
        ty = rng.choice(sorted(types))
        j, pre, rem, _ = rng.choice(types[ty])
        return '\n'.join(lines[:j]) + '\n' + pre, rem, ty


def build(split, n, base_seed):
    traces = load_traces()
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKS.items()}
    items = []
    gens = {'copy': lambda r: fewshot(r, g_copy, 6, 'Repeat the words.'),
            'reverse': lambda r: fewshot(r, g_rev, 6, 'Reverse the order of the words.'),
            'brackets': lambda r: fewshot(r, g_br, 6, 'Close all open brackets.'),
            'arith': t_arith}
    for ti, task in enumerate(['copy', 'reverse', 'brackets', 'arith', 'countdown']):
        made, s = 0, 0
        while made < n:
            seed = base_seed + ti * 100000 + s
            s += 1
            rng = random.Random(seed)
            slot = None
            if task == 'countdown':
                prompt, target, slot = t_countdown(rng, traces)
            else:
                prompt, target = gens[task](rng)
            ntok = {k: len(t(prompt, add_special_tokens=False)['input_ids']) for k, t in toks.items()}
            ttok = {k: len(t(' ' + target.strip(), add_special_tokens=False)['input_ids']) for k, t in toks.items()}
            if max(ntok.values()) > MAXTOK:
                continue
            items.append({'task': task, 'id': f'{task}_{seed}', 'seed': seed, 'prompt': prompt,
                          'target': target.strip(), 'slot': slot, 'prompt_tokens': ntok,
                          'max_new_tokens': max(ttok.values()) + 8})
            made += 1
    os.makedirs(f'{FEWSHOT_DIR}/data', exist_ok=True)
    json.dump({'split': split, 'n_per_task': n, 'MAXTOK': MAXTOK, 'items': items},
              open(f'{FEWSHOT_DIR}/data/items_{split}.json', 'w'))
    for task in ['copy', 'reverse', 'brackets', 'arith', 'countdown']:
        it = [x for x in items if x['task'] == task]
        mx = max(max(x['prompt_tokens'].values()) for x in it)
        mn = sum(x['prompt_tokens']['gpt2bpe'] for x in it) / len(it)
        print(split, task, len(it), 'max prompt tok', mx, 'mean gpt2', round(mn), 'max_new max',
              max(x['max_new_tokens'] for x in it), flush=True)


if __name__ == '__main__':
    build('pilot', 40, 900000)
    build('main', 500, 1000)
