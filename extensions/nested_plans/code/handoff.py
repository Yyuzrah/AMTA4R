"""Splitting a NestPlan plan into separate model calls (implicit format only).

The splitting code reads only the skeleton of a plan: the event kinds (push / pop) and the pushed subgoal types.
It never reads the token that follows a pop (the scored target), so no answer enters a call; those tokens are
filled in from the plan only for teacher-forced scoring (`materialize`).

Splitting rule with per-call nesting budget D >= 2 (relative depth inside a call):
  emit(u, rd):  if rd == D and u has children  -> stub   "( g_u ) g_u"   (the subtree of u becomes a new call)
                else                           -> "( g_u"  emit(children, rd+1)  ") g_u"
  top call   : emit(u, 1) for every top-level subgoal u (in order)
  sub-call(v): emit(v, 1)  (a fresh context that contains only the subtree of v, split again by the same rule)
Every call therefore has nesting depth <= D. The pop target of a subgoal v whose subtree became a new call is scored
in that call ("own" target); the parent call contains only the stub (a leaf; its pop target is the most recent push).
A plan with depth <= D yields exactly one call, identical to the original sequence.
"""
import numpy as np
from nestplan import PAD, BOS, EOS, OPEN, CLOSE


def parse_skeleton(ids):
    """Skeleton of a plan: kinds (True = push) and pushed types (None for pops)."""
    kinds, ptypes = [], []
    p = 1
    while int(ids[p]) != EOS:
        k = int(ids[p])
        if k == OPEN:
            kinds.append(True); ptypes.append(int(ids[p + 1]))
        elif k == CLOSE:
            kinds.append(False); ptypes.append(None)   # ids[p+1] (the target) is deliberately not read
        else:
            raise ValueError(f'unexpected token {k} at {p}')
        p += 2
    return kinds, ptypes


def build_tree(kinds):
    """Bracket matching on the skeleton. nodes[i] = dict(open_ev, close_ev, depth, parent, children)."""
    nodes, top, stack = [], [], []
    for i, is_open in enumerate(kinds):
        if is_open:
            nid = len(nodes)
            par = stack[-1] if stack else -1
            nodes.append(dict(open_ev=i, close_ev=None, depth=len(stack) + 1, parent=par, children=[]))
            (nodes[par]['children'] if par >= 0 else top).append(nid)
            stack.append(nid)
        else:
            nodes[stack.pop()]['close_ev'] = i
    assert not stack, 'unbalanced plan'
    return nodes, top


def plan_depth(kinds):
    d = mx = 0
    for k in kinds:
        d += 1 if k else -1
        mx = max(mx, d)
    return mx


def split_calls(ids, D):
    """Split one plan into calls of nesting depth <= D (D >= 2; D=None -> one call).
    Each call is a dict with
      tok   : list of tokens, where a scored target is the placeholder -1 (filled by `materialize`)
      ptype : pushed type for each emitted pop target (only used to fill a stub's returned type)
      orig  : list (per position) of the original position of the scored target, -1 if not scored
      stub  : list (per position) 1 if the scored target is a returned stub, else 0
      depth : nesting depth of the call
    """
    kinds, ptypes = parse_skeleton(ids)
    if D is None or plan_depth(kinds) <= D:
        D_eff = None
    else:
        assert D >= 2, 'per-call budget must be >= 2 for the stub scheme to make progress'
        D_eff = D
    nodes, top = build_tree(kinds)
    calls = []
    queue = [None]  # None = top call
    while queue:
        root = queue.pop(0)
        tok, orig, stub = [BOS], [-1], [0]
        maxd = [0]

        def emit(u, rd):
            nd = nodes[u]
            maxd[0] = max(maxd[0], rd)
            pop_pos = 2 + 2 * nd['close_ev']   # original position of u's pop target
            tok.append(OPEN); orig.append(-1); stub.append(0)
            tok.append(ptypes[nd['open_ev']]); orig.append(-1); stub.append(0)
            if D_eff is not None and rd == D_eff and nd['children']:
                queue.append(u)
                tok.append(CLOSE); orig.append(-1); stub.append(0)
                tok.append(-1); orig.append(pop_pos); stub.append(1)
                return
            for c in nd['children']:
                emit(c, rd + 1)
            tok.append(CLOSE); orig.append(-1); stub.append(0)
            tok.append(-1); orig.append(pop_pos); stub.append(0)

        if root is None:
            for u in top:
                emit(u, 1)
        else:
            emit(root, 1)
        tok.append(EOS); orig.append(-1); stub.append(0)
        calls.append(dict(tok=tok, orig=orig, stub=stub, depth=maxd[0], root=root))
    return calls


def materialize(call, ids):
    """Fill the scored targets of a call with the true tokens of the plan (teacher forcing and scoring)."""
    t = np.asarray(call['tok'], dtype=np.int64).copy()
    o = np.asarray(call['orig'], dtype=np.int64)
    m = o >= 0
    t[m] = np.asarray(ids, dtype=np.int64)[o[m]]
    tgt = m.astype(np.int8)
    return dict(ids=t.astype(np.int16), tgt=tgt, orig=o.astype(np.int32),
                stub=np.asarray(call['stub'], dtype=np.int8), depth=call['depth'])


def compose(plan_ids, mcalls, preds):
    """Score one plan that was split into calls. preds[i][p] = predicted token at position p of call i.
    Returns dict(all_ok, own_ok, call_ok list, own_pred{orig_pos: pred})."""
    call_ok, own = [], {}
    for mc, pr in zip(mcalls, preds):
        m = mc['tgt'] > 0
        ok = bool((pr[m] == mc['ids'][m].astype(np.int64)).all())
        call_ok.append(ok)
        for p in np.nonzero(m & (mc['stub'] == 0))[0]:
            op = int(mc['orig'][p])
            assert op not in own, 'a pop target resolved twice'
            own[op] = int(pr[p])
    pops = [p for p in range(2, len(plan_ids) - 1, 2) if int(plan_ids[p - 1]) == CLOSE]
    assert set(own) == set(pops), 'every pop target must be resolved exactly once'
    own_ok = all(own[p] == int(plan_ids[p]) for p in pops)
    return dict(all_ok=all(call_ok), own_ok=own_ok, call_ok=call_ok, own_pred=own)
