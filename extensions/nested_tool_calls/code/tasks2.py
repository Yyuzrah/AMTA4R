"""Nested tool calls: item generator, prompt rendering, split into calls and answer extraction (pure Python, no GPU).

Two surfaces over the same generator (value composition along a chain of nested subgoals):
  T  nested tool-call returns (run): a registry of one-line Python tools
                           `def f(x): return g(x) + c` (calls one tool, adds c) or `def f(x): return x + c` (base).
                           Query: what does f1(x0) return?  Gold = x0 + sum of the constants along f1 -> ... -> fd.
  S  nested to-do subgoals in an agent's notes (not run): `- f: takes c steps.` (base) or
                           `- f: needs g first, then c more steps.`  Query: the agent starts f1 at step x0; at which
                           step is f1 finished?  Gold = x0 + sum of the constants (same algebra as T).
An item has a queried chain of nesting depth d (f1 calls f2 ... calls fd, fd is a base). With the layout of the runs
(config.DIS_MODE = 'k2') it also has 2 distractor tools, which call each other or nothing but never a chain tool, and
24 - d note lines, so the registry has 26 lines at every d. Registry order 'callee' (default): every tool is listed
after the tool it calls (bottom-up, as helpers are defined before use in code); distractors and note lines are
interleaved at random positions, and a distractor calls an earlier distractor with probability 0.75 (otherwise it is
a base).
Conditions: I (one call, answer only), E (one call; the prompt also lists the call chain of the question = the stack
of pending subgoals), A and H (split: the chain is cut bottom-up into segments of <= D levels; each segment is a fresh
call in which the tool just below the segment is replaced by the value that the previous call returned; A: D = 1;
H: D = max(1, d_hat*) from the calibration split).
The split reads only the structure (which tool calls which), never a constant or a value.
"""
import random
import re
from names import NAMES

M_FUN = 24
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
TASKS = ('T', 'S')
CONST = (1, 2, 3)
DEMO_NAMES = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliet',
              'kilo', 'lima', 'mike', 'november', 'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango']
assert not set(DEMO_NAMES) & set(NAMES)
PREFILL = 'Answer:'


def item_rng(task, split, d, idx, tag='v2'):
    return random.Random(f'LLMH2|{task}|{split}|{d}|{idx}|{tag}')  # fixed seed string: it determines every item


def make_item(task, split, d, idx, M=M_FUN, order='callee', names_pool=None, rng=None, dis_mode='linked'):
    """Chain 0 -> 1 -> ... -> d-1 (0 = queried tool, d-1 = base of the chain); distractors d..M-1.
    dis_mode 'linked': a distractor may call any earlier tool (also chain tools); 'free': only earlier distractors;
    'filler': no distractor tools, M - d neutral note lines instead; 'k<K>': see below (the runs use 'k2')."""
    assert 1 <= d <= M
    rng = rng or item_rng(task, split, d, idx)
    K = None
    if dis_mode.startswith('k'):            # 'k<K>': d chain tools + K free distractor tools + (M - d) note lines
        K = int(dis_mode[1:])
        names = rng.sample(names_pool or NAMES, d + K) + [None] * (M - d)
        M = M + K
    else:
        names = rng.sample(names_pool or NAMES, M)
    chain = list(range(d))
    dis = list(range(d, M))
    callee = {i: (i + 1 if i + 1 < d else None) for i in chain}
    if order in ('callee', 'caller'):
        shown = list(reversed(chain)) if order == 'callee' else list(chain)
        for u in dis:
            shown.insert(rng.randrange(len(shown) + 1), u)
        pos = {u: i for i, u in enumerate(shown)}
        for u in dis:
            cand = shown[:pos[u]] if order == 'callee' else shown[pos[u] + 1:]
            if dis_mode == 'free':
                cand = [v for v in cand if v >= d]
            if K is not None:
                cand = [v for v in cand if d <= v < d + K]
                if u >= d + K:
                    callee[u] = None
                    continue
            callee[u] = rng.choice(cand) if cand and rng.random() < 0.75 else None
    elif order == 'shuffled':          # registry lines in random order (not used in the runs)
        topo = list(chain)
        for u in dis:
            topo.insert(rng.randrange(len(topo) + 1), u)
        pos = {u: i for i, u in enumerate(topo)}
        for u in dis:
            later = topo[pos[u] + 1:]
            callee[u] = rng.choice(later) if later and rng.random() < 0.75 else None
        shown = list(range(M))
        rng.shuffle(shown)
    else:
        raise ValueError(order)
    const = {u: rng.choice(CONST) for u in range(M)}
    x0 = rng.randrange(10)
    val = {}

    def ev(u):
        if u not in val:
            val[u] = (ev(callee[u]) if callee[u] is not None else x0) + const[u]
        return val[u]
    for u in range(M):
        if names[u] is not None:
            ev(u)
    notes = {}
    if dis_mode == 'filler':
        for u in dis:
            notes[u] = ' '.join(rng.sample(NOTE_WORDS, 3))
    if K is not None:            # note length chosen so that the registry length is about constant over d
        for u in range(d + K, M):
            notes[u] = ' '.join(rng.sample(NOTE_WORDS, NOTE_N.get(task, 5)))
            const.pop(u, None)
    return {'task': task, 'split': split, 'd': d, 'idx': idx, 'M': M, 'order': order, 'names': names,
            'callee': callee, 'const': const, 'x0': x0, 'shown': shown, 'val': val, 'notes': notes,
            'dis_mode': dis_mode}


NOTE_N = {'T': 9, 'S': 6}            # T: 9 words keeps the token length within ~1% over d
NOTE_WORDS = ('check', 'cache', 'later', 'logs', 'retry', 'review', 'update', 'docs', 'style', 'tests', 'format',
              'rename', 'cleanup', 'version', 'config', 'timeout', 'backup', 'archive', 'monitor', 'refactor')


def build_calls(it, D):
    """Split into calls. D None -> one call. Otherwise the chain is cut bottom-up into segments of <= D levels.
    Call k queries chain[q]; if `known` is set, the value of chain[known] is supplied from call k-1.
    Returns calls in execution order (the last one is the main call)."""
    d = it['d']
    if D is None or d <= D:
        segs = [(0, None)]
    else:
        segs = []
        top = d
        while top > 0:
            lo = max(0, top - D)
            segs.append((lo, top if top < d else None))
            top = lo
    return [{'id': k, 'q': q, 'known': kn, 'known_call': (k - 1 if kn is not None else None),
             'main': k == len(segs) - 1, 'level': k, 'nest': ((kn if kn is not None else d) - q)}
            for k, (q, kn) in enumerate(segs)]


# ----------------------------------------------------------------------------------------------- rendering
INTRO = {
    'T': ('You will answer questions about small registries of Python tools. Every tool takes an integer x. A tool '
          'either returns x plus a constant, or calls exactly one other tool with the same x and returns that '
          "tool's result plus a constant. Reply with the integer only."),
    'S': ('You will answer questions about the to-do notes of an agent. Each task either takes a fixed number of '
          'steps, or first needs one other task (its subtask) and then takes a fixed number of more steps. The agent '
          'always finishes a needed subtask completely before it continues with the task that needs it, and it starts '
          'the subtask at the step at which it starts the task. Reply with the step number only.'),
}
HEAD = {'T': 'Registry:', 'S': 'To-do notes:'}


def _line(task, nm, cal, c):
    if nm is None:              # note line
        return f'# note: {cal}' if task == 'T' else f'- (note) {cal}'
    if task == 'T':
        return f'def {nm}(x): return {cal}(x) + {c}' if cal is not None else f'def {nm}(x): return x + {c}'
    st = 'step' if c == 1 else 'steps'
    return f'- {nm}: needs {cal} first, then {c} more {st}.' if cal is not None else f'- {nm}: takes {c} {st}.'


def _question(task, q, x0):
    if task == 'T':
        return f'Question: What does {q}({x0}) return?'
    return f'Question: The agent starts {q} at step {x0}. At which step is {q} finished?'


def _known(task, g, x0, v):
    if task == 'T':
        return f'Known result (already computed by a helper): {g}({x0}) = {v}. Use it instead of recomputing {g}.'
    return (f'Known result (reported by a helper agent): started at step {x0}, {g} is finished at step {v}. '
            f'Use it instead of working out {g} again.')


KNOWN_STYLE = 'subst'   # 'subst' (used in the runs): the answer of the previous call replaces the definition of the
                        # tool it computed; 'line' (not used): the full registry plus a separate 'Known result' line


def _subst_line(task, g, x0, v):
    if task == 'T':
        return f'def {g}(x): return {v}  # result for x = {x0}, returned by a helper call'
    return f'- {g}: already done by a helper agent; started at step {x0}, it was finished at step {v}.'


def _chain_line(task, path, x0, known):
    if task == 'T':
        s = ' -> '.join(f'{p}({x0})' for p in path)
        return 'Call chain of the question (each call waits for the next one): ' + s + (
            ' [known]' if known else ' [base]') + '.'
    s = ' -> '.join(path)
    return 'Open to-do stack of the question (each task waits for the next one): ' + s + (
        ' [known]' if known else ' [base]') + '.'


def _block(task, rows, q, x0, cond, known=None, path=None, style=None):
    """rows: list of (name, callee name or None, const). known: (name, value) or None. path: chain names (E)."""
    style = style or KNOWN_STYLE
    lines = [(_subst_line(task, r[0], x0, known[1]) if (known is not None and style == 'subst' and r[0] == known[0])
              else _line(task, *r)) for r in rows]
    out = HEAD[task] + '\n' + '\n'.join(lines) + '\n'
    if known is not None and style == 'line':
        out += _known(task, known[0], x0, known[1]) + '\n'
    if cond == 'E':
        out += _chain_line(task, path, x0, known is not None) + '\n'
    out += _question(task, q, x0)
    return out


def _demo_items(dis_mode='linked', M=8):
    """Four fixed demonstrations (names outside the task pool): depth 1, 2, 3 and a depth-2 segment with a known
    result (the format of the calls of a split). Registries with M = 8 in 'callee' order (with dis_mode 'k2': d + 2
    tools and 8 - d note lines)."""
    out = []
    # fixed seed string: it determines the demonstrations
    rng = random.Random('LLMH2|demos|v2' + ('' if (dis_mode, M) == ('linked', 8) else f'|{dis_mode}|{M}'))
    for k, (d, kn) in enumerate([(1, False), (2, False), (3, False), (3, True)]):
        it = make_item('demo', 'demo', d, k, M=M, order='callee', names_pool=DEMO_NAMES, rng=rng, dis_mode=dis_mode)
        out.append((it, kn))
    return out


_DEMOS = {}


def demos(dis_mode='linked', M=8):
    if (dis_mode, M) not in _DEMOS:
        _DEMOS[(dis_mode, M)] = _demo_items(dis_mode, M)
    return _DEMOS[(dis_mode, M)]


def render(task, it, call, cond, answers=None, shots=True):
    """Returns (user message, assistant prefill). cond in I/E (A/H calls are rendered with cond 'I' plus a known line)."""
    assert cond in ('I', 'E')
    txt = INTRO[task] + '\n'
    if shots:
        for j, (dm, kn) in enumerate(demos(it.get('dis_mode', 'linked'), it.get('demo_M', 8))):
            rows = [((None, dm['notes'][u], None) if u in dm['notes'] else
                     (dm['names'][u], dm['names'][dm['callee'][u]] if dm['callee'][u] is not None else None,
                      dm['const'][u])) for u in dm['shown']]
            known = (dm['names'][dm['d'] - 1], dm['val'][dm['d'] - 1]) if kn else None
            path = [dm['names'][u] for u in range(dm['d'])]   # E: the chain down to the base (or the known tool)
            txt += f'\nExample {j + 1}.\n' + _block(task, rows, dm['names'][0], dm['x0'], cond, known, path,
                                                     it.get('known_style')) + \
                f'\n{PREFILL} {dm["val"][0]}\n'
        txt += '\nNow the real question.\n'
    names, callee, const, x0 = it['names'], it['callee'], it['const'], it['x0']
    rows = [((None, it['notes'][u], None) if u in it['notes'] else
             (names[u], names[callee[u]] if callee[u] is not None else None, const[u])) for u in it['shown']]
    known = None
    if call['known'] is not None:
        known = (names[call['known']], answers[call['known_call']])
    top = call['known'] if call['known'] is not None else it['d'] - 1
    path = [names[u] for u in range(call['q'], top + 1)]
    txt += '\n' + _block(task, rows, names[call['q']], x0, cond, known, path, it.get('known_style'))
    return txt, PREFILL


def gold_of(it, call):
    return str(it['val'][call['q']])


def extract(text):
    """First integer of the generated text (after the 'Answer:' prefill)."""
    m = re.search(r'-?\d+', text)
    return m.group(0) if m else None


def fingerprint(it):
    """Structure + constants + x0 of an item (for split-leakage checks); note lines count by position only."""
    rows = tuple(('#',) if it['names'][u] is None else
                 (it['names'][u], it['callee'][u] is not None and it['names'][it['callee'][u]], it['const'][u])
                 for u in it['shown'])
    return (rows, it['x0'])
