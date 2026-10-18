"""Unit tests of the plan generator (N = 48, k = 8) and of the splitting code in handoff.py, for every per-call
budget D in 2..8 and plan depths 1..40 (part of check G_b of analyze.py).
Usage: python test_handoff.py [plans per (D, d), default 12]"""
import sys, os, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nestplan import Vocab, make_example, oracle_targets, OPEN, CLOSE, EOS, BOS
from handoff import split_calls, materialize, compose, parse_skeleton, plan_depth

N, K = 48, 8
V = Vocab(k=K, n_nodes=N)


def check_plan(d, seed, D):
    rng = random.Random(f'unit-{seed}')
    e = make_example(rng, N, d, K, 'implicit', V)
    ids = e['ids']
    assert len(ids) == 4 * N + 2
    kinds, _ = parse_skeleton(ids)
    assert plan_depth(kinds) == d
    calls = [materialize(c, ids) for c in split_calls(ids, D)]
    for c in calls:
        # depth of every call <= D (recomputed from tokens)
        dd = mx = 0
        for t in c['ids']:
            t = int(t)
            if t == OPEN: dd += 1; mx = max(mx, dd)
            if t == CLOSE: dd -= 1
        assert dd == 0 and mx == c['depth'] and mx <= max(D, 1), (d, D, mx)
        # within a call, the scored targets are exactly the oracle's (the call is a valid implicit plan)
        orc = oracle_targets(c['ids'], 'implicit', V)
        assert set(orc) == set(np.nonzero(c['tgt'])[0].tolist())
        for p, v in orc.items():
            assert int(c['ids'][p]) == v
    if d <= D:
        assert len(calls) == 1 and np.array_equal(calls[0]['ids'], ids), 'no split expected'
    # oracle predictions -> success; one wrong own target -> failure
    preds = [c['ids'].astype(np.int64).copy() for c in calls]
    r = compose(ids, calls, preds)
    assert r['all_ok'] and r['own_ok']
    n_nodes = sum(int((c['ids'] == OPEN).sum()) for c in calls)
    n_stub = sum(int(c['stub'].sum()) for c in calls)
    assert n_nodes == N + n_stub, 'each subgoal appears once, plus one stub per new call'
    assert len(calls) == 1 + n_stub
    i = random.Random(seed).randrange(len(calls))
    pos = np.nonzero(calls[i]['tgt'])[0]
    p = int(pos[random.Random(seed + 1).randrange(len(pos))])
    preds[i][p] = (preds[i][p] + 1)
    r2 = compose(ids, calls, preds)
    assert not r2['all_ok']
    assert r2['own_ok'] == bool(calls[i]['stub'][p] == 1)
    return len(calls)


if __name__ == '__main__':
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    stats = {}
    for D in range(2, 9):
        for d in range(1, 41):
            for s in range(n_rep):
                nc = check_plan(d, 1000 * d + s, D)
                stats.setdefault((D, d), []).append(nc)
    for D in range(2, 9):
        print('D', D, 'mean calls at d=8,16,24,32,40:', [round(float(np.mean(stats[(D, d)])), 1) for d in (8, 16, 24, 32, 40)])
    print('all splitting unit tests passed (D 2..8, d 1..40, %d plans each)' % n_rep)
