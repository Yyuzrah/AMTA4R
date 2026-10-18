"""NestPlan: nested subgoal plans with nesting depth controlled independently of length.

A plan is a forest of N subgoal nodes whose maximum nesting depth is exactly d.
It is built from a spine (a chain of d nested subgoals) plus N-d lateral
"distractor" subgoals, each attached uniformly at random to an existing node of
depth < d (or to the top level). Children are ordered at random; the plan is
serialised by DFS as push/pop events (a Dyck-k word with k = number of subgoal
types). Total length is the same for every d (fixed N).

Formats (same plan, different serialisation):
  implicit : BOS  ( g  ...  ) g  ... EOS        -- pop target = type of the most
             recent unmatched push (the open stack is never restated)
  explicit : BOS  ( g | s1..sk  ) g | s1..sk-1 ... EOS
             -- after every event the open stack (bottom->top) is restated,
             so the pop target is the last restated item (a lookup)
  flat     : BOS  (_t g  ...  )_t g ... EOS     -- every subgoal carries a unique
             tag t shared by its push and pop token; the pop target is found by
             tag lookup (no nesting has to be resolved). Same length and same
             push-pop distances as implicit.

Deterministic targets (scored): the subgoal-type token after every pop
(all formats) and, in explicit, the restated stack tokens.
"""
import random
import numpy as np

PAD, BOS, EOS, OPEN, CLOSE, SEP = 0, 1, 2, 3, 4, 5
N_SPECIAL = 6


class Vocab:
    def __init__(self, k=8, n_nodes=24):
        self.k = k
        self.n = n_nodes
        self.type0 = N_SPECIAL
        self.otag0 = N_SPECIAL + k
        self.ctag0 = N_SPECIAL + k + n_nodes
        self.size = N_SPECIAL + k + 2 * n_nodes
        self.size_padded = int(np.ceil(self.size / 64) * 64)


def gen_plan(rng, n_nodes, depth):
    """Return (parent, node_depth, children_order) for a forest with exactly
    n_nodes nodes and max depth exactly `depth` (top-level nodes have depth 1)."""
    assert 1 <= depth <= n_nodes
    parent = [-1]
    ndepth = [1]
    for i in range(1, depth):
        parent.append(i - 1)
        ndepth.append(i + 1)
    eligible = [-1] + [i for i in range(depth) if ndepth[i] < depth]
    for j in range(depth, n_nodes):
        p = eligible[rng.randrange(len(eligible))]
        dep = 1 if p == -1 else ndepth[p] + 1
        parent.append(p)
        ndepth.append(dep)
        if dep < depth:
            eligible.append(j)
    children = {-1: []}
    for i in range(n_nodes):
        children[i] = []
    for i in range(n_nodes):
        children[parent[i]].append(i)
    for key in children:
        if len(children[key]) > 1:
            rng.shuffle(children[key])
    return parent, ndepth, children


def dfs_events(children):
    """Iterative DFS -> list of (is_open, node)."""
    events = []
    stack = [(-1, 0)]
    while stack:
        node, ci = stack.pop()
        ch = children[node]
        if ci < len(ch):
            stack.append((node, ci + 1))
            c = ch[ci]
            events.append((True, c))
            stack.append((c, 0))
        else:
            if node != -1:
                events.append((False, node))
    return events


def make_example(rng, n_nodes, depth, k, fmt, vocab):
    """Build one sequence (rng: a random.Random instance). Returns dict with
    ids: np.int16 token ids
    tgt: np.int8 target kind per position p (the token at p is a scored target):
         0 = not scored, 1 = pop target (subgoal type after a pop), 2 = restated stack token
    pop_depth: depth of the node popped at each pop target (0 elsewhere)
    pop_nontriv: 1 if the popped node has children (the most recent push is not its match)
    """
    parent, ndepth, children = gen_plan(rng, n_nodes, depth)
    types = [rng.randrange(k) for _ in range(n_nodes)]
    tags = list(range(n_nodes))
    rng.shuffle(tags)
    events = dfs_events(children)
    assert len(events) == 2 * n_nodes
    ids = [BOS]
    tgt = [0]
    pdep = [0]
    pnt = [0]
    stack = []
    for is_open, node in events:
        if fmt == 'flat':
            ids.append((vocab.otag0 if is_open else vocab.ctag0) + int(tags[node]))
        else:
            ids.append(OPEN if is_open else CLOSE)
        tgt.append(0); pdep.append(0); pnt.append(0)
        ids.append(vocab.type0 + int(types[node]))
        if is_open:
            tgt.append(0); pdep.append(0); pnt.append(0)
            stack.append(node)
        else:
            assert stack[-1] == node
            stack.pop()
            tgt.append(1); pdep.append(ndepth[node]); pnt.append(1 if len(children[node]) > 0 else 0)
        if fmt == 'explicit':
            ids.append(SEP); tgt.append(0); pdep.append(0); pnt.append(0)
            for s in stack:
                ids.append(vocab.type0 + int(types[s])); tgt.append(2); pdep.append(0); pnt.append(0)
    ids.append(EOS); tgt.append(0); pdep.append(0); pnt.append(0)
    return {
        'ids': np.asarray(ids, dtype=np.int16),
        'tgt': np.asarray(tgt, dtype=np.int8),
        'pop_depth': np.asarray(pdep, dtype=np.int8),
        'pop_nontriv': np.asarray(pnt, dtype=np.int8),
        'depth': depth,
    }


def oracle_targets(ids, fmt, vocab):
    """Independent reference solver: recompute every scored target from the
    token sequence alone (no access to the tree). Returns dict position->token."""
    out = {}
    stack = []
    tagmap = {}
    p = 1
    T = len(ids)
    while p < T and ids[p] != EOS:
        tok = int(ids[p])
        if fmt == 'flat':
            if vocab.otag0 <= tok < vocab.ctag0:
                tagmap[tok - vocab.otag0] = int(ids[p + 1])
                p += 2
            else:
                out[p + 1] = tagmap[tok - vocab.ctag0]
                p += 2
            continue
        if tok == OPEN:
            stack.append(int(ids[p + 1]))
        elif tok == CLOSE:
            out[p + 1] = stack.pop()
        else:
            raise ValueError(f'unexpected token {tok} at {p}')
        p += 2
        if fmt == 'explicit':
            assert ids[p] == SEP
            p += 1
            for j, s in enumerate(stack):
                out[p + j] = s
            p += len(stack)
    return out


def recency_baseline(ids, fmt, vocab):
    """Heuristic: predict the type of the most recent push (ignores matching)."""
    out = {}
    last = None
    for p in range(1, len(ids) - 1):
        tok = int(ids[p])
        is_open = (tok == OPEN) or (fmt == 'flat' and vocab.otag0 <= tok < vocab.ctag0)
        is_close = (tok == CLOSE) or (fmt == 'flat' and vocab.ctag0 <= tok < vocab.ctag0 + vocab.n)
        if is_open:
            last = int(ids[p + 1])
        elif is_close:
            out[p + 1] = last
    return out


def make_testset(seed, n_nodes, depths, n_per_depth, k, fmts, vocab):
    """Same plans for every format (paired): one rng stream per (depth, i)."""
    sets = {f: [] for f in fmts}
    for d in depths:
        for i in range(n_per_depth):
            for f in fmts:
                rng = random.Random(f'{seed}-{d}-{i}')
                sets[f].append(make_example(rng, n_nodes, d, k, f, vocab))
    return sets
