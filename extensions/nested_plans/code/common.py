"""Constants and helpers shared by train.py, verify.py, gatecheck.py and analyze.py (no torch import)."""
import os
import random

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(CODE)

N, K = 48, 8                      # subgoals per plan, subgoal types
SEQ_LEN = 4 * N + 2               # 194 tokens for every plan (matched length)
TEST_SEED, CAL_SEED, VAL_SEED = 'NPv2-test-v1', 'NPv2-cal-v1', 'NPv2-val-v1'   # fixed seed strings of the plan sets
N_TEST, N_CAL, N_VAL = 500, 200, 64
STEPS, STEPS_EXT = 30000, 60000   # training steps; STEPS_EXT for the cells re-trained after gatecheck.py
D_TRAINS = [2, 4, 6, 8]
PES = ['rope', 'ape']
MODELS = ['L2', 'L4', 'L8', 'L12', 'W624']
SEEDS = [0, 1, 2]
MODEL_SPEC = {'L2': (2, 256, 4), 'L4': (4, 256, 4), 'L8': (8, 256, 4), 'L12': (12, 256, 4), 'W624': (2, 624, 12)}
L_OF = {m: MODEL_SPEC[m][0] for m in MODELS}
GRID_COLS = ['idx', 'name', 'pe', 'dtrain', 'model', 'L', 'd', 'h', 'seed', 'steps']
VERIFY_GROUP = 5                  # grid rows checked together by one verify.py job (rows idx // VERIFY_GROUP)


def dtest_of(dtrain):
    return 4 * dtrain + 8


def split_ds(dtrain):
    """Per-call depth budgets D for which every test plan is also evaluated split into calls (D = 2..d_train)."""
    return list(range(2, dtrain + 1))


def val_depths(dtrain):
    return sorted({1, 2, max(1, dtrain // 2), dtrain, dtrain + 1, dtrain + 2, 2 * dtrain, 3 * dtrain,
                   dtest_of(dtrain)})


def stream_key(dtrain, seed, start_step, wid):
    """rng key of the training stream (fresh i.i.d. plans; d ~ U{1..dtrain}); start_step is the step at which
    training started or resumed, wid the data-loader worker."""
    return f'NPv2-train-dt{dtrain}-s{seed}-{start_step}-{wid}'   # fixed seed string


def run_name(pe, dtrain, model, seed):
    return f'{pe}_dt{dtrain}_{model}_s{seed}'


def cell_of(name):
    """'rope_dt4_L12_s0' -> ('rope', 4, 'L12')"""
    pe, dt, m, _ = name.split('_')
    return pe, int(dt[2:]), m


def make_grid():
    """Rows in seed-major order, larger models first within a (d_train, PE) block."""
    rows = []
    for s in SEEDS:
        for dt in D_TRAINS:
            for pe in PES:
                for m in ['L12', 'W624', 'L8', 'L4', 'L2']:
                    L, d, h = MODEL_SPEC[m]
                    rows.append(dict(idx=len(rows), name=run_name(pe, dt, m, s), pe=pe, dtrain=dt, model=m,
                                     L=L, d=d, h=h, seed=s, steps=STEPS))
    return rows


def load_grid(path):
    rows = []
    with open(path) as f:
        head = f.readline().split()
        assert head == GRID_COLS, head
        for line in f:
            if not line.strip():
                continue
            v = line.split()
            r = dict(zip(head, v))
            for k in ['idx', 'dtrain', 'L', 'd', 'h', 'seed', 'steps']:
                r[k] = int(r[k])
            rows.append(r)
    assert [r['idx'] for r in rows] == list(range(len(rows)))
    return rows


def write_grid(rows, path):
    with open(path, 'w') as f:
        f.write('\t'.join(GRID_COLS) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[c]) for c in GRID_COLS) + '\n')


if __name__ == '__main__':
    g = make_grid()
    write_grid(g, os.path.join(CODE, 'grid.tsv'))
    # grid_smoke.tsv: quick test grid of 16 runs of 600 steps
    sm = []
    for pe in PES:
        for dt in (2, 8):
            for s in (0, 1, 2):
                L, d, h = MODEL_SPEC['L2']
                sm.append(dict(idx=len(sm), name=run_name(pe, dt, 'L2', s), pe=pe, dtrain=dt, model='L2', L=L, d=d,
                               h=h, seed=s, steps=600))
    for pe in PES:
        for m in ('L12', 'W624'):
            L, d, h = MODEL_SPEC[m]
            sm.append(dict(idx=len(sm), name=run_name(pe, 8, m, 0), pe=pe, dtrain=8, model=m, L=L, d=d, h=h, seed=0,
                           steps=600))
    write_grid(sm, os.path.join(CODE, 'grid_smoke.tsv'))
    print(len(g), 'grid rows;', len(sm), 'smoke rows')


# ---------------------------------------------------------------- check G_a, re-trained cells and analysed runs
GATE_A_MIN = 0.9


def gate_a(res):
    """G_a: test A_single(d) >= 0.9 for every d in 1..d_train (in distribution). res = results.json dict."""
    dt = res['config']['dtrain']
    pdp = res['test_single']['per_depth']
    return all(pdp[str(d)]['seq'] >= GATE_A_MIN - 1e-12 for d in range(1, dt + 1))


def cell_key(row):
    return (row['pe'], row['dtrain'], row['model'])


def read_json(path, default=None):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def rescue_names(base):
    """Names of the runs re-trained with STEPS_EXT steps in runs_ext/ (from runs/gatecheck.json); analyze.py uses
    those runs for these names."""
    g = read_json(os.path.join(base, 'runs', 'gatecheck.json'), {}) or {}
    return set(g.get('rescue_names', []))


def effective_dir(base, name, rescue):
    return os.path.join(base, 'runs_ext' if name in rescue else 'runs', name)


def touch(path, text=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
