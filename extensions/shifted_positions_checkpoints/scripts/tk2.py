"""Trace parsing, insertion sites and inert units (CPU only).
Site = start of a leaf-level 'Exploring Operation' line (current node has 2 numbers, result has 1 number), before the
first terminal line, whose previous line is either
  'bfs' type: a 'No Solution' line of a sibling leaf at the same node            -> unit U = EX NS
  'dfs' type: a restated 'Current State' line (identical text seen earlier) that follows 'Moving to Node #X'
                                                                                  -> unit U = EX NS MV(X) CS(S)
Eligible inert op: valid combine_nums op on the node's two numbers, result != target, whose (operator, result) is not
among the ops explored at that node (same Current State text) within `scope`:
  scope='full'   : anywhere in the text (so the continuation stays a valid trace after insertion; used for reference
                   traces and for the 'strict' free-running rule on the model's own unedited run)
  scope='prefix' : only in lines before the site (free-running 'prefix' rule)
The op chosen is the first eligible op in combine_nums order (deterministic)."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
DATA = os.environ.get('SHIFT_DATA', f'{ROOT}/data/experiment_500k')
sys.path.insert(0, f'{ROOT}/src')
from countdown_utils import combine_nums, metric_fn  # noqa: E402,F401

RE_CS = re.compile(r'^Current State: (\d+):\[([0-9, ]*)\], Operations: \[(.*)\]$')
RE_EX = re.compile(r'^Exploring Operation: (\d+)([-+*/])(\d+)=(\d+), Resulting Numbers: \[([0-9, ]*)\]$')
RE_MV = re.compile(r'^Moving to Node #([0-9,]+)$')
RE_NS = re.compile(r'^(\d+),(\d+) unequal: No Solution$')
RE_GR = re.compile(r'^(\d+),(\d+) equal: Goal Reached$')
ROUTE_MV = re.compile(r'^Moving to Node #([0-9,]+)$')
ROUTE_EX = re.compile(r'^Exploring Operation: (\d+[-+*/]\d+)=')


def nums_of(s):
    """Numbers of a well-formed list body 'a, b, c'; None if malformed (e.g. model output '48,48')."""
    s = s.strip()
    if not s:
        return []
    parts = s.split(', ')
    if not all(p.isdigit() for p in parts):
        return None
    return [int(x) for x in parts]


def split_lines(text):
    """[(start_char, line)] for every complete line (terminated by newline)."""
    out, pos = [], 0
    while True:
        e = text.find('\n', pos)
        if e < 0:
            break
        out.append((pos, text[pos:e]))
        pos = e + 1
    return out


def route_spans(text):
    """Char spans of routing tokens: node id after 'Moving to Node #', and 'a op b' in 'Exploring Operation: a op b='."""
    spans = []
    for s, ln in split_lines(text):
        m = ROUTE_MV.match(ln)
        if m:
            spans.append((s + m.start(1), s + m.end(1)))
        m = ROUTE_EX.match(ln)
        if m:
            spans.append((s + m.start(1), s + m.end(1)))
    return spans


def terminal_char(text):
    for s, ln in split_lines(text):
        if RE_GR.match(ln) or ln.startswith('No solution found'):
            return s
    return len(text)


def _opch(op):
    return re.match(r'\d+([-+*/])\d+=', op).group(1)


def find_sites(text, scope='full', strict_text=None):
    """All candidate sites in order. strict_text: if given (free-running), each site also gets 'strict_ops' = eligible
    ops w.r.t. explored ops at that node anywhere in strict_text (the model's own full unedited run)."""
    L = split_lines(text)
    term = terminal_char(text)

    def explored_map(lines):
        ex, cur = {}, None
        for _, ln in lines:
            if RE_CS.match(ln):
                cur = ln
            m = RE_EX.match(ln)
            if m and cur is not None:
                ex.setdefault(cur, set()).add((m.group(2), int(m.group(4))))
        return ex
    full_ex = explored_map(L)
    strict_ex = explored_map(split_lines(strict_text)) if strict_text is not None else None
    sites = []
    cur, cur_i = None, None
    seen_cs = set()
    pre_ex = {}  # explored ops before current line
    pcur = None
    for i, (s, ln) in enumerate(L):
        if s >= term:
            break
        mcs = RE_CS.match(ln)
        if mcs:
            restated = ln in seen_cs
            seen_cs.add(ln)
            cur, cur_i, cur_restated = ln, i, restated
            continue
        m = RE_EX.match(ln)
        if m and cur is not None and i >= 2:
            mc = RE_CS.match(cur)
            t = int(mc.group(1)); nn = nums_of(mc.group(2)); rn = nums_of(m.group(5))
            prev = L[i - 1][1]
            typ, node_id = None, None
            if nn is not None and rn is not None and len(nn) == 2 and len(rn) == 1:
                if RE_NS.match(prev) and cur_i < i - 1:
                    typ = 'bfs'
                elif cur_i == i - 1 and cur_restated:
                    mm = RE_MV.match(L[i - 2][1])
                    if mm:
                        typ, node_id = 'dfs', mm.group(1)
            if typ is not None:
                ex = full_ex if scope == 'full' else pre_ex
                seen = ex.get(cur, set())
                elig, strict = [], []
                for r, op in combine_nums(nn[0], nn[1]):
                    if r == t or (_opch(op), r) in seen:
                        continue
                    elig.append((op, r))
                    if strict_ex is not None and (_opch(op), r) not in strict_ex.get(cur, set()):
                        strict.append((op, r))
                if elig:
                    sites.append(dict(char=s, line_idx=i, type=typ, state_line=cur, target=t, nums=nn,
                                      node_id=node_id, elig=elig, strict_ops=strict))
        if m and cur is not None:
            pre_ex.setdefault(cur, set()).add((m.group(2), int(m.group(4))))
    return sites


def unit_text(site, op, r):
    t = site['target']
    u = f"Exploring Operation: {op}, Resulting Numbers: [{r}]\n{r},{t} unequal: No Solution\n"
    if site['type'] == 'dfs':
        u += f"Moving to Node #{site['node_id']}\n{site['state_line']}\n"
    return u
