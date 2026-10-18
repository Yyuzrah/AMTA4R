"""Analysis of the nested-plan runs: d_ext, the calibrated split, the splitting rule R per run, the checks G_a/G_b/G_c,
the summary of R over the (d_train, model, PE) cells (3 seeds each; the runs of the cells listed in runs/gatecheck.json
are read from runs_ext/), checks (recomputation from the per-plan arrays, sanity checks, disjointness of training,
validation, calibration and test plans, a second derivation of the status), tables and figures.
Usage: python analyze.py BASE GRID_TSV [OUT_DIR]
Writes OUT/summary.json (key "status"), OUT/RESULTS.md, OUT/verify.json, OUT/d_star.json, OUT/curves.png,
OUT/d_ext.png, OUT/table_cells.tex."""
import hashlib, json, math, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from nestplan import Vocab, make_example, make_testset

BASE = sys.argv[1]
GRID_PATH = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, 'analysis')
NBOOT = int(os.environ.get('NPV2_NBOOT', 2000))
TAUS = (0.5, 0.9)
L0 = C.SEQ_LEN


# ---------------------------------------------------------------- d_ext
def pava_nonincreasing(y, w=None):
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(-y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            ww = wts[-2] + wts[-1]; c = cnt[-2] + cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
            vals[-1], wts[-1], cnt[-1] = v, ww, c
    out = []
    for v, c in zip(vals, cnt):
        out += [-v] * c
    return np.array(out)


def d_ext_iso(A, tau):
    """Largest d at which the fit that may not increase with depth (pava_nonincreasing) is >= tau; 0 if none.
    Returns (value, censored at d_test)."""
    fit = pava_nonincreasing(A)
    ok = [d for d, f in enumerate(fit, start=1) if f >= tau - 1e-12]
    if not ok:
        return 0, False
    return int(max(ok)), bool(max(ok) == len(A))


def d_ext_raw(A, tau):
    v = 0
    for d, a in enumerate(A, start=1):
        if a >= tau - 1e-12:
            v = d
        else:
            break
    return v, bool(v == len(A))


def per_depth(ok, depth, depths):
    return np.array([ok[depth == d].mean() for d in depths])


def boot_ci(ok, depth, depths, tau, seed):
    rng = np.random.default_rng(seed)
    n = np.array([(depth == d).sum() for d in depths])
    p = per_depth(ok.astype(float), depth, depths)
    Ab = rng.binomial(n, p, size=(NBOOT, len(depths))) / n
    vals = [d_ext_iso(a, tau)[0] for a in Ab]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def policy(z, dcal, dtrain, dtest):
    """Calibrated split: a plan deeper than d_ext_cal is split into calls with budget D_call = min(d_ext_cal, d_train),
    every other plan is solved in one call; not constructible if D_call < 2 (then every plan is solved in one call)."""
    dep = z['test_depth'].astype(int)
    single = z['test_single_ok']
    Dc = min(dcal, dtrain)
    if dcal >= dtest or Dc < 2:
        return dict(D_call=Dc, constructible=bool(Dc >= 2), all_ok=single.copy(), own_ok=single.copy(),
                    calls=np.ones(len(dep)), tokens=np.full(len(dep), float(L0)), trig=np.zeros(len(dep), bool))
    trig = dep > dcal
    sa, so = z[f'test_split{Dc}_all_ok'], z[f'test_split{Dc}_own_ok']
    nc, nt = z[f'test_split{Dc}_n_calls'].astype(float), z[f'test_split{Dc}_n_tok'].astype(float)
    return dict(D_call=Dc, constructible=True, all_ok=np.where(trig, sa, single), own_ok=np.where(trig, so, single),
                calls=np.where(trig, nc, 1.0), tokens=np.where(trig, nt, float(L0)), trig=trig)


def rule_R(A_hand, A_single, dcal, constructible, depths):
    Bset = [d for d in depths if d > dcal]
    if not Bset:
        return dict(B=[], holds=True, vacuous=True)
    idx = [d - 1 for d in Bset]
    mn, mean = float(A_hand[idx].min()), float(A_hand[idx].mean())
    gain = float((A_hand[idx] - A_single[idx]).mean())
    holds = bool(constructible and mn >= 0.8 and mean >= 0.9 and gain >= 0.3)
    return dict(B=[Bset[0], Bset[-1]], holds=holds, vacuous=False, min_hand=mn, mean_hand=mean, mean_gain=gain,
                mean_single=float(A_single[idx].mean()), constructible=bool(constructible))


def spearman(x, y):
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort(kind='mergesort')
        r = np.empty(len(a)); r[order] = np.arange(len(a), dtype=float)
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def sign_test_p(npos, nneg):
    n = npos + nneg
    if n == 0:
        return None
    k = min(npos, nneg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def fmt_t(v):
    d, c = v
    return f'{d}+' if c else f'{d}'


def r4(x):
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------- per-run analysis
def analyse_run(row, rdir, res, z):
    dt, dtest = row['dtrain'], C.dtest_of(row['dtrain'])
    depths = list(range(1, dtest + 1))
    tdep, cdep = z['test_depth'].astype(int), z['cal_depth'].astype(int)
    As = per_depth(z['test_single_ok'].astype(float), tdep, depths)
    Ac = per_depth(z['cal_single_ok'].astype(float), cdep, depths)
    out = dict(idx=row['idx'], pe=row['pe'], dtrain=dt, dtest=dtest, model=row['model'], seed=row['seed'],
               L=row['L'], width=row['d'], dir=os.path.relpath(rdir, BASE), steps=res['config']['steps'],
               params_nonembed=res['params_nonembed'], params_total=res['params_total'], gpu=res['gpu'],
               train_h=res['train_seconds'] / 3600, eval_s=res.get('eval_seconds'), n_resumes=res.get('n_resumes'))
    out['A_single'] = [r4(a) for a in As]
    out['A_single_cal'] = [r4(a) for a in Ac]
    pdp = res['test_single']['per_depth']
    out['pop_acc'] = [r4(pdp[str(d)]['pop_acc']) for d in depths]
    out['nontriv_pop_acc'] = [r4(pdp[str(d)]['nontriv_pop_acc']) for d in depths]
    out['recency_seq'] = [pdp[str(d)]['recency_seq'] for d in depths]
    out['recency_pop_acc'] = [r4(pdp[str(d)]['recency_pop_acc']) for d in depths]
    out['acc_by_popped_depth'] = {j: r4(v['acc']) for j, v in res['test_single']['per_pop_depth'].items()}
    de = {}
    for tau in TAUS:
        de[f'cal_{tau}'] = d_ext_iso(Ac, tau)
        de[f'test_{tau}'] = d_ext_iso(As, tau)
        de[f'cal_raw_{tau}'] = d_ext_raw(Ac, tau)
        de[f'test_raw_{tau}'] = d_ext_raw(As, tau)
        de[f'cal_{tau}_ci'] = boot_ci(z['cal_single_ok'], cdep, depths, tau, 1000 + row['idx'])
        de[f'test_{tau}_ci'] = boot_ci(z['test_single_ok'], tdep, depths, tau, 5000 + row['idx'])
    out['d_ext'] = de
    out['gate_a'] = bool(C.gate_a(res))
    for tau in TAUS:
        dcal = de[f'cal_{tau}'][0]
        P = policy(z, dcal, dt, dtest)
        Ah = per_depth(P['all_ok'].astype(float), tdep, depths)
        Ao = per_depth(P['own_ok'].astype(float), tdep, depths)
        Rr = rule_R(Ah, As, dcal, P['constructible'], depths)
        Bm = tdep > dcal
        out[f'policy_{tau}'] = dict(
            d_ext_cal=dcal, D_call=P['D_call'], constructible=P['constructible'],
            A_hand=[r4(a) for a in Ah], A_hand_own=[r4(a) for a in Ao],
            calls=[round(float(P['calls'][tdep == d].mean()), 2) for d in depths],
            tokens=[round(float(P['tokens'][tdep == d].mean()), 1) for d in depths],
            mean_calls_B=float(P['calls'][Bm].mean()) if Bm.any() else None,
            mean_tokens_B=float(P['tokens'][Bm].mean()) if Bm.any() else None,
            R=Rr, no_harm=bool(all(Ah[i] >= As[i] - 0.05 - 1e-12 for i in range(dtest))),
            min_hand_minus_single=float((Ah - As).min()))
    for D in C.split_ds(dt):
        out[f'split{D}'] = dict(
            A=[r4(a) for a in per_depth(z[f'test_split{D}_all_ok'].astype(float), tdep, depths)],
            A_own=[r4(a) for a in per_depth(z[f'test_split{D}_own_ok'].astype(float), tdep, depths)],
            calls=[round(float(z[f'test_split{D}_n_calls'][tdep == d].mean()), 2) for d in depths],
            tokens=[round(float(z[f'test_split{D}_n_tok'][tdep == d].mean()), 1) for d in depths],
            call_success=float(z[f'test_split{D}_call_ok'].mean()),
            call_success_by_call_depth=res[f'test_split{D}']['call_success_by_call_depth'])
    lp = os.path.join(rdir, 'train_log.jsonl')
    if os.path.exists(lp):
        recs = [json.loads(x) for x in open(lp) if x.strip()]
        if recs:
            out['final_loss'] = recs[-1]['loss']
            out['curves'] = [dict(step=x['step'], val_seq=x.get('val_seq')) for x in recs if 'val_seq' in x]
    return out


# ---------------------------------------------------------------- self-check helpers
def selfcheck_run(row, res, z, vrec):
    """Re-computation from raw per-plan outputs and sanity checks for one run. Returns (ok, issues)."""
    iss = []
    c = res['config']
    dt, dtest = row['dtrain'], C.dtest_of(row['dtrain'])
    depths = list(range(1, dtest + 1))
    exp_steps = C.STEPS_EXT if os.sep + 'runs_ext' + os.sep in row['_dir'] + os.sep else row['steps']
    for k, v in dict(pe=row['pe'], L=row['L'], d=row['d'], h=row['h'], seed=row['seed'], dtrain=dt, dtest=dtest,
                     N=C.N, k=C.K, steps=exp_steps).items():
        if c.get(k) != v:
            iss.append(f'config {k}={c.get(k)} expected {v}')
    tdep, cdep = z['test_depth'].astype(int), z['cal_depth'].astype(int)
    for tag, dep, n in (('test', tdep, c['n_test']), ('cal', cdep, c['n_cal'])):
        cnt = [int((dep == d).sum()) for d in depths]
        if cnt != [n] * dtest or len(dep) != n * dtest:
            iss.append(f'{tag} per-depth counts {sorted(set(cnt))} expected {n}')
        ok = z[f'{tag}_single_ok']; nc = z[f'{tag}_single_ncorr']
        if not np.array_equal(ok, nc == C.N):
            iss.append(f'{tag}: sequence success != (all {C.N} pops correct)')
    for d in depths:
        a = float(z['test_single_ok'][tdep == d].mean())
        if abs(a - res['test_single']['per_depth'][str(d)]['seq']) > 1e-9:
            iss.append(f'single A({d}) recomputed {a} != results.json'); break
    if abs(res['test_single']['per_depth']['1']['recency_seq'] - 1.0) > 1e-12:
        iss.append('recency baseline at d=1 is not 1.0')
    if not res.get('oracle_composition_ok', False):
        iss.append('oracle composition failed in evaluation')
    if res.get('split_ds') != C.split_ds(dt):
        iss.append(f"split budgets {res.get('split_ds')} != {C.split_ds(dt)}")
    for D in C.split_ds(dt):
        sa, so = z[f'test_split{D}_all_ok'], z[f'test_split{D}_own_ok']
        nc, md = z[f'test_split{D}_n_calls'], z[f'test_split{D}_max_call_depth']
        for d in depths:
            m = tdep == d
            if abs(float(sa[m].mean()) - res[f'test_split{D}']['per_depth'][str(d)]['seq_all']) > 1e-9:
                iss.append(f'split{D} A({d}) recomputed != results.json'); break
        shallow = tdep <= D
        if not (np.all(nc[shallow] == 1) and np.array_equal(sa[shallow], z['test_single_ok'][shallow])):
            iss.append(f'split{D}: plans with depth <= D are not identical to the single call')
        if np.any(nc[~shallow] < 2):
            iss.append(f'split{D}: a plan deeper than D was not split')
        if np.any(md > D):
            iss.append(f'split{D}: a call deeper than D')
        if np.any(sa & ~so):
            iss.append(f'split{D}: all-calls success without own-target success')
        if len(z[f'test_split{D}_call_ok']) != int(nc.sum()):
            iss.append(f'split{D}: call count mismatch')
    if vrec is None:
        iss.append('verify.json missing (GPU verification not run)')
    return (len(iss) == 0), iss


def plan_hash(ids):
    return hashlib.sha1(np.asarray(ids, np.int16).tobytes()).hexdigest()


def leakage_check(grid, smoke):
    """Calibration, test, validation and training-stream plans must be disjoint (hash of the token sequence)."""
    vocab = Vocab(k=C.K, n_nodes=C.N)
    dmax = max(C.dtest_of(r['dtrain']) for r in grid)
    n_test = max(int(r.get('_n_test', C.N_TEST)) for r in grid)
    n_cal = max(int(r.get('_n_cal', C.N_CAL)) for r in grid)
    depths = list(range(1, dmax + 1))
    H = {}
    for tag, seed, n in (('test', C.TEST_SEED, n_test), ('cal', C.CAL_SEED, n_cal)):
        H[tag] = {plan_hash(e['ids']) for e in make_testset(seed, C.N, depths, n, C.K, ['implicit'], vocab)['implicit']}
    vd = sorted({d for r in grid for d in C.val_depths(r['dtrain'])})
    H['val'] = {plan_hash(e['ids']) for e in make_testset(C.VAL_SEED, C.N, vd, C.N_VAL, C.K, ['implicit'], vocab)['implicit']}
    per_train = 300 if smoke else 2000
    tr = set()
    for dt in sorted({r['dtrain'] for r in grid}):
        for s in sorted({r['seed'] for r in grid}):
            for wid in range(5):
                rng = random.Random(C.stream_key(dt, s, 0, wid))
                for _ in range(per_train):
                    tr.add(plan_hash(make_example(rng, C.N, rng.randint(1, dt), C.K, 'implicit', vocab)['ids']))
    H['train_sample'] = tr
    out = dict(sizes={k: len(v) for k, v in H.items()},
               expected_sizes=dict(test=n_test * dmax, cal=n_cal * dmax),
               overlaps={f'{a}&{b}': len(H[a] & H[b]) for a, b in
                         [('test', 'cal'), ('test', 'val'), ('cal', 'val'), ('test', 'train_sample'),
                          ('cal', 'train_sample'), ('val', 'train_sample')]},
               seed_strings_distinct=len({C.TEST_SEED, C.CAL_SEED, C.VAL_SEED, 'NPv2-train'}) == 4,
               train_sample_per_stream=per_train)
    out['pass'] = bool(all(v == 0 for v in out['overlaps'].values()) and out['seed_strings_distinct']
                       and out['sizes']['test'] == n_test * dmax and out['sizes']['cal'] == n_cal * dmax)
    return out


def iso_minmax(y):
    """Second implementation of the non-increasing fit (equal weights): f(k) = min_{i<=k} max_{j>=k} mean(y[i..j])."""
    y = np.asarray(y, float); n = len(y)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    f = np.empty(n)
    for k in range(n):
        f[k] = min(max((cs[j + 1] - cs[i]) / (j + 1 - i) for j in range(k, n)) for i in range(k + 1))
    return f


def rederive_status(grid, Z, eff_ok, rescue, gate_b):
    """Second derivation of the status, directly from the per-plan arrays."""
    Rres, ga = {}, {}
    for r in grid:
        n = r['name']
        if n not in Z:
            continue
        z = Z[n]; dt = r['dtrain']; dtest = C.dtest_of(dt)
        tdep, cdep = z['test_depth'].astype(int), z['cal_depth'].astype(int)
        Acal = np.array([z['cal_single_ok'][cdep == d].mean() for d in range(1, dtest + 1)])
        Ate = np.array([z['test_single_ok'][tdep == d].mean() for d in range(1, dtest + 1)])
        ga[n] = bool(all(Ate[:dt] >= 0.9 - 1e-12))
        f = iso_minmax(Acal)
        above = [d for d in range(1, dtest + 1) if f[d - 1] >= 0.5 - 1e-12]
        dcal = max(above) if above else 0
        B = list(range(dcal + 1, dtest + 1))
        if not B:
            Rres[n] = (True, True); continue
        Dc = min(dcal, dt)
        if Dc < 2:
            Rres[n] = (False, False); continue
        hand = []
        for d in B:
            m = tdep == d
            hand.append(z[f'test_split{Dc}_all_ok'][m].mean())
        hand = np.array(hand); single = Ate[dcal:]
        Rres[n] = (bool(hand.min() >= 0.8 and hand.mean() >= 0.9 and (hand - single).mean() >= 0.3), False)
    cells = {}
    for r in grid:
        cells.setdefault(C.cell_key(r), []).append(r['name'])
    missing = [n for r in grid for n in [r['name']] if n not in Z]
    if missing:
        return 'not_run'
    if not all(ga.values()) or not gate_b:
        return 'not_all_checks_hold'
    holds = {k: all(Rres[n][0] for n in v) for k, v in cells.items()}
    testable = {k: sum(not Rres[n][1] for n in v) >= 2 for k, v in cells.items()}
    n_test = sum(testable.values())
    per_pe = {pe: sum(1 for k, t in testable.items() if t and k[0] == pe) for pe in {k[0] for k in cells}}
    if not (n_test >= len(cells) / 2 and all(v >= min(5, sum(1 for k in cells if k[0] == pe)) for pe, v in per_pe.items())):
        return 'not_all_checks_hold'
    if all(holds.values()):
        return 'R_holds_in_all_cells'
    fails = [k for k in cells if testable[k] and not holds[k]]
    return 'R_fails_in_half_of_testable_cells' if len(fails) >= n_test / 2 else 'not_all_checks_hold'


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    grid = C.load_grid(GRID_PATH)
    smoke = os.path.basename(GRID_PATH) != 'grid.tsv'
    gate = C.read_json(os.path.join(BASE, 'runs', 'gatecheck.json'), None)
    rescue = set(gate['rescue_names']) if gate else set()
    unit = C.read_json(os.path.join(BASE, 'runs', 'markers', 'unittest.json'), None)
    R, Z, V, dirs = {}, {}, {}, {}
    for r in grid:
        rd = C.effective_dir(BASE, r['name'], rescue)
        r['_dir'] = rd
        p, q = os.path.join(rd, 'results.json'), os.path.join(rd, 'per_plan.npz')
        if os.path.exists(p) and os.path.exists(q):
            R[r['name']] = json.load(open(p)); Z[r['name']] = dict(np.load(q)); dirs[r['name']] = rd
            r['_n_test'], r['_n_cal'] = R[r['name']]['config']['n_test'], R[r['name']]['config']['n_cal']
            v = C.read_json(os.path.join(rd, 'verify.json'), None)
            if v is not None:
                V[r['name']] = v
    missing = [r['name'] for r in grid if r['name'] not in R]
    cells = {}
    for r in grid:
        cells.setdefault(C.cell_key(r), []).append(r)
    S = dict(experiment='nested_plans', smoke=smoke, generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             n_runs_expected=len(grid), n_runs_found=len(R), missing=missing, rescue_cells=gate['rescue_cells'] if gate else None,
             rescue_names=sorted(rescue), gatecheck_found=gate is not None, runs={})
    for r in grid:
        if r['name'] in R:
            S['runs'][r['name']] = analyse_run(r, dirs[r['name']], R[r['name']], Z[r['name']])
    runs = S['runs']

    # ---------- gate ----------
    G_a = bool(all(runs[n]['gate_a'] for n in runs) and not missing)
    unit_ok = bool(unit and unit.get('pass'))
    oracle_ok = all(R[n].get('oracle_composition_ok', False) for n in R)
    verify_ctrl = all(V[n].get('controller_ok', False) for n in V) if V else False
    G_b = bool(unit_ok and oracle_ok and verify_ctrl and len(V) == len(R) and not missing)
    G_c = not missing
    S['gate'] = dict(G_a=G_a, G_a_failures=[n for n in runs if not runs[n]['gate_a']], G_b=G_b,
                     G_b_parts=dict(unit_tests=unit_ok, oracle_composition_all_runs=oracle_ok,
                                    verify_controller_ok=verify_ctrl, verified_runs=len(V)),
                     G_c=G_c, rescue_applied=bool(rescue))

    # ---------- cells and rule R ----------
    C_out = {}
    for k, rows in cells.items():
        names = [x['name'] for x in rows]
        have = [n for n in names if n in runs]
        Rs = {n: runs[n]['policy_0.5']['R'] for n in have}
        nonvac = [n for n in have if not Rs[n]['vacuous']]
        C_out['%s_dt%d_%s' % k] = dict(
            pe=k[0], dtrain=k[1], model=k[2], runs=names, found=len(have), complete=len(have) == len(names),
            holds=bool(len(have) == len(names) and all(Rs[n]['holds'] for n in have)),
            testable=bool(len(nonvac) >= 2), n_nonvacuous=len(nonvac),
            R={n: Rs[n]['holds'] for n in have},
            d_ext_cal=[runs[n]['d_ext']['cal_0.5'][0] for n in have],
            d_ext_test=[runs[n]['d_ext']['test_0.5'][0] for n in have],
            d_ext_test_censored=[runs[n]['d_ext']['test_0.5'][1] for n in have],
            D_call=[runs[n]['policy_0.5']['D_call'] for n in have],
            min_hand_B=[Rs[n].get('min_hand') for n in have], mean_hand_B=[Rs[n].get('mean_hand') for n in have],
            mean_single_B=[Rs[n].get('mean_single') for n in have], mean_gain_B=[Rs[n].get('mean_gain') for n in have],
            mean_calls_B=[runs[n]['policy_0.5']['mean_calls_B'] for n in have],
            mean_tokens_B=[runs[n]['policy_0.5']['mean_tokens_B'] for n in have],
            calls_at_dtest=[runs[n]['policy_0.5']['calls'][-1] for n in have],
            no_harm=[runs[n]['policy_0.5']['no_harm'] for n in have],
            R_tau09=[runs[n]['policy_0.9']['R']['holds'] for n in have],
            gate_a=[runs[n]['gate_a'] for n in have])
    S['cells'] = C_out
    cell_keys = list(C_out)
    testable_cells = [c for c in cell_keys if C_out[c]['testable']]
    pes = sorted({C_out[c]['pe'] for c in cell_keys})
    per_pe_total = {pe: sum(1 for c in cell_keys if C_out[c]['pe'] == pe) for pe in pes}
    per_pe_testable = {pe: sum(1 for c in testable_cells if C_out[c]['pe'] == pe) for pe in pes}
    testable = bool(len(testable_cells) >= len(cell_keys) / 2 and
                    all(per_pe_testable[pe] >= min(5, per_pe_total[pe]) for pe in pes))
    fails = [c for c in cell_keys if not C_out[c]['holds']]
    fails_testable = [c for c in testable_cells if not C_out[c]['holds']]
    r_all = bool(G_a and G_b and testable and not fails and not missing)
    if missing:
        status = 'not_run'
    elif not (G_a and G_b):
        status = 'not_all_checks_hold'
    elif not testable:
        status = 'not_all_checks_hold'
    elif r_all:
        status = 'R_holds_in_all_cells'
    elif len(fails_testable) >= len(testable_cells) / 2:
        status = 'R_fails_in_half_of_testable_cells'
    else:
        status = 'not_all_checks_hold'
    rob_a = bool(runs) and all(runs[n]['policy_0.9']['R']['holds'] for n in runs)
    rob_b = bool(runs) and all(runs[n]['policy_0.5']['no_harm'] for n in runs)
    rob_c = bool(runs) and all(n in V and V[n].get('pass', False) for n in runs)

    # ---------- self-check (d) ----------
    sc_runs = {}
    for r in grid:
        n = r['name']
        if n in R:
            ok, iss = selfcheck_run(r, R[n], Z[n], V.get(n))
            sc_runs[n] = dict(ok=ok, issues=iss)
    leak = leakage_check(grid, smoke)
    re_status = rederive_status(grid, Z, None, rescue, G_b)
    primary_for_compare = status if status != 'R_holds_in_all_cells' else 'R_holds_in_all_cells'
    sc = dict(per_run=sc_runs, all_runs_ok=all(v['ok'] for v in sc_runs.values()) if sc_runs else False,
              leakage=leak, rederived_status=re_status, rederived_agrees=re_status == primary_for_compare,
              params_matched=None)
    # width control matched to the deepest model (within 2%) per PE
    pm = {}
    for pe in pes:
        a = [runs[n]['params_nonembed'] for n in runs if runs[n]['pe'] == pe and runs[n]['model'] == 'W624']
        b = [runs[n]['params_nonembed'] for n in runs if runs[n]['pe'] == pe and runs[n]['model'] == 'L12']
        if a and b:
            pm[pe] = dict(W624=a[0], L12=b[0], rel_diff=abs(a[0] - b[0]) / b[0])
    sc['params_matched'] = pm
    sc['params_ok'] = all(v['rel_diff'] <= 0.02 for v in pm.values())
    rob_d = bool(sc['all_runs_ok'] and leak['pass'] and sc['rederived_agrees'] and sc['params_ok'])
    sc['pass'] = rob_d
    if status == 'R_holds_in_all_cells':
        status = 'all_checks_hold' if (rob_a and rob_b and rob_c and rob_d) else 'not_all_checks_hold'
    S['rule_R'] = dict(status=status, testable=testable, n_cells=len(cell_keys), n_testable_cells=len(testable_cells),
                         testable_per_pe=per_pe_testable, cells_failing=fails, testable_cells_failing=fails_testable,
                         R_holds_all_cells=r_all, check_a_tau09=rob_a, check_b_no_harm=rob_b,
                         check_c_verified=rob_c, check_d_selfcheck=rob_d,
                         runs_R_fail=[n for n in runs if not runs[n]['policy_0.5']['R']['holds']],
                         runs_tau09_fail=[n for n in runs if not runs[n]['policy_0.9']['R']['holds']],
                         runs_harm=[n for n in runs if not runs[n]['policy_0.5']['no_harm']],
                         runs_verify_fail=[n for n in runs if not (n in V and V[n].get('pass', False))])
    S['status'] = status

    # ---------- descriptive d_ext (test, tau 0.5) ----------
    def dx(n):
        return runs[n]['d_ext']['test_0.5'][0]
    desc = {}
    for k, rows in cells.items():
        v = [dx(x['name']) for x in rows if x['name'] in runs]
        desc['%s_dt%d_%s' % k] = dict(seeds=v, mean=float(np.mean(v)) if v else None,
                                      censored=[runs[x['name']]['d_ext']['test_0.5'][1] for x in rows if x['name'] in runs])
    vs_L = {}
    for pe in pes:
        for dt in sorted({r['dtrain'] for r in grid}):
            ms = [desc.get(f'{pe}_dt{dt}_{m}', {}).get('mean') for m in ['L2', 'L4', 'L8', 'L12']]
            xs, ys = [], []
            for m in ['L2', 'L4', 'L8', 'L12']:
                for x in cells.get((pe, dt, m), []):
                    if x['name'] in runs:
                        xs.append(C.L_OF[m]); ys.append(dx(x['name']))
            reading = None
            if all(v is not None for v in ms):
                if all(ms[i] <= ms[i + 1] for i in range(3)) and ms[3] >= ms[0] + 2:
                    reading = 'grows with L'
                elif abs(ms[3] - ms[0]) <= 1:
                    reading = 'flat'
                else:
                    reading = 'irregular'
            w = desc.get(f'{pe}_dt{dt}_W624', {}).get('mean')
            vs_L[f'{pe}_dt{dt}'] = dict(means_L2_L4_L8_L12=ms, spearman=spearman(xs, ys), reading=reading,
                                        W624=w, W624_minus_L12=(w - ms[3]) if (w is not None and ms[3] is not None) else None,
                                        W624_minus_L2=(w - ms[0]) if (w is not None and ms[0] is not None) else None)
    vs_dt = {}
    for pe in pes:
        for m in C.MODELS:
            dts = sorted({r['dtrain'] for r in grid if r['model'] == m and r['pe'] == pe})
            ex = [desc.get(f'{pe}_dt{dt}_{m}', {}).get('mean') for dt in dts]
            ex = [None if v is None else v - dt for v, dt in zip(ex, dts)]
            reading = None
            if len(dts) >= 2 and all(v is not None for v in ex):
                if max(ex) - min(ex) <= 1:
                    reading = 'constant margin'
                elif all(ex[i] <= ex[i + 1] for i in range(len(ex) - 1)) and ex[-1] >= ex[0] + 2:
                    reading = 'margin grows with d_train'
                elif all(ex[i] >= ex[i + 1] for i in range(len(ex) - 1)) and ex[-1] <= ex[0] - 2:
                    reading = 'margin shrinks with d_train'
                else:
                    reading = 'irregular'
            xs = [r['dtrain'] for r in grid if r['model'] == m and r['pe'] == pe and r['name'] in runs]
            ys = [dx(r['name']) for r in grid if r['model'] == m and r['pe'] == pe and r['name'] in runs]
            slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) >= 2 else None
            vs_dt[f'{pe}_{m}'] = dict(d_train=dts, excess_mean=ex, ols_slope=slope, reading=reading)
    pairs = []
    for r in grid:
        if r['pe'] == 'rope':
            a = r['name']; b = 'ape' + a[4:]
            if a in runs and b in runs:
                pairs.append(dx(a) - dx(b))
    npos, nneg, nzero = sum(p > 0 for p in pairs), sum(p < 0 for p in pairs), sum(p == 0 for p in pairs)
    S['descriptive_d_ext'] = dict(per_cell=desc, vs_L=vs_L, vs_d_train=vs_dt,
                                  rope_minus_ape=dict(n_pairs=len(pairs), mean=float(np.mean(pairs)) if pairs else None,
                                                      n_pos=npos, n_neg=nneg, n_zero=nzero,
                                                      sign_test_p=sign_test_p(npos, nneg)))
    # compute summary
    S['compute'] = dict(train_gpu_h=float(sum(runs[n]['train_h'] for n in runs)),
                        eval_gpu_h=float(sum((runs[n]['eval_s'] or 0) for n in runs) / 3600),
                        verify_gpu_h=float(sum(V[n].get('seconds', 0) for n in V) / 3600),
                        gpus=sorted({runs[n]['gpu'] for n in runs}))
    S['analysis_seconds'] = time.time() - t0

    # ---------- verify.json (self-check + GPU verification) ----------
    VJ = dict(status=status, selfcheck=sc, gpu_verification={n: {k: V[n].get(k) for k in
                                                                   ['pass', 'max_abs_diff', 'worst', 'controller_ok',
                                                                    'free_run_agreement', 'n_free_run', 'gpu', 'seconds']}
                                                               for n in V},
              gate=S['gate'], checks=dict(a=rob_a, b=rob_b, c=rob_c, d=rob_d))
    json.dump(VJ, open(os.path.join(OUT, 'verify.json.tmp'), 'w'), indent=1)
    os.replace(os.path.join(OUT, 'verify.json.tmp'), os.path.join(OUT, 'verify.json'))
    S['selfcheck_pass'] = rob_d

    # ---------- d_star.json ----------
    dstar = dict(definition='d_ext = largest depth at which the fit of sequence success that may not increase with '
                            'depth is >= tau (tau 0.5 primary); '
                            'NestPlan N=48 implicit format, trained on d ~ U{1..d_train}; 95%% CI: plan bootstrap (%d)' % NBOOT,
                 per_run={n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in runs[n]['d_ext'].items()} for n in runs},
                 per_cell=desc)
    json.dump(dstar, open(os.path.join(OUT, 'd_star.json'), 'w'), indent=1)

    write_results_md(S, VJ, grid, cells)
    write_table_tex(S, grid)
    try:
        make_figures(S, grid, cells)
    except Exception as ex:
        print('figure failed:', ex)
        S['figure_error'] = str(ex)
    # summary.json is written last
    tmp = os.path.join(OUT, 'summary.json.tmp')
    json.dump(S, open(tmp, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(tmp, os.path.join(OUT, 'summary.json'))
    print('status:', status, '| self-check', rob_d, '| rederived', re_status, '| seconds', round(time.time() - t0, 1))


def write_results_md(S, VJ, grid, cells):
    runs, d = S['runs'], S['rule_R']
    L = []
    L.append('# Nested plans: results (generated by code/analyze.py)\n')
    if S['smoke']:
        L.append('**Quick test mode: tiny training budgets and evaluation sets.**\n')
    L.append(f"Generated {S['generated']}. Runs found: {S['n_runs_found']}/{S['n_runs_expected']}; missing: {S['missing']}\n")
    L.append(f"## Status of the checks: **{S['status']}**\n")
    g = S['gate']
    L.append(f"- Gate: G_a (in-distribution A_single(d) >= 0.9 for d <= d_train, every run) = {g['G_a']}; "
             f"failures {g['G_a_failures']}; G_b (splitting code) = {g['G_b']} {g['G_b_parts']}; G_c (all runs) = {g['G_c']}")
    L.append(f"- Cells re-trained with 2x steps in runs_ext/: {S['rescue_cells']}")
    L.append(f"- Testable: {d['testable']} ({d['n_testable_cells']}/{d['n_cells']} cells with non-empty B_r for >= 2 seeds; "
             f"per PE {d['testable_per_pe']})")
    L.append(f"- Cells where the splitting rule R fails for some seed: {d['cells_failing']}")
    L.append(f"- Additional checks: (a) tau 0.9 calibration = {d['check_a_tau09']} (fails {d['runs_tau09_fail']}); "
             f"(b) no harm = {d['check_b_no_harm']} (fails {d['runs_harm']}); (c) GPU verification = "
             f"{d['check_c_verified']} (fails {d['runs_verify_fail']}); (d) self-check = {d['check_d_selfcheck']}")
    sc = VJ['selfcheck']
    L.append(f"- Self-check: per-run recomputation ok = {sc['all_runs_ok']}; split leakage pass = {sc['leakage']['pass']} "
             f"(overlaps {sc['leakage']['overlaps']}); re-derived status = {sc['rederived_status']} "
             f"(agrees {sc['rederived_agrees']}); W624 vs L12 params {sc['params_matched']}")
    bad = {n: v['issues'] for n, v in sc['per_run'].items() if not v['ok']}
    if bad:
        L.append(f"- Self-check issues: {json.dumps(bad)[:3000]}")
    cp = S['compute']
    L.append(f"- Compute: train {cp['train_gpu_h']:.1f} GPU-h, eval {cp['eval_gpu_h']:.1f} GPU-h, verify {cp['verify_gpu_h']:.1f} GPU-h; GPUs {cp['gpus']}\n")

    L.append('## Cells (tau 0.5 calibrated policy; one value per seed)\n')
    L.append('| cell | runs | gate_a | d_ext_cal | d_ext_test | D_call | R per seed | min A_hand on B | mean A_single on B | mean A_hand on B | mean calls on B | calls at d_test | no harm | R (tau 0.9) | cell holds |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')

    def f3(v):
        return '-' if v is None else f'{v:.3f}'

    def f1(v):
        return '-' if v is None else f'{v:.1f}'
    for c, x in S['cells'].items():
        L.append(f"| {c} | {x['found']}/{len(x['runs'])} | {x['gate_a']} | {x['d_ext_cal']} | "
                 f"{['%d%s' % (a, '+' if b else '') for a, b in zip(x['d_ext_test'], x['d_ext_test_censored'])]} | {x['D_call']} | "
                 f"{list(x['R'].values())} | {' '.join(f3(v) for v in x['min_hand_B'])} | {' '.join(f3(v) for v in x['mean_single_B'])} | "
                 f"{' '.join(f3(v) for v in x['mean_hand_B'])} | {' '.join(f1(v) for v in x['mean_calls_B'])} | "
                 f"{' '.join(f1(v) for v in x['calls_at_dtest'])} | {x['no_harm']} | {x['R_tau09']} | **{x['holds']}** |")
    dd = S['descriptive_d_ext']
    L.append('\n## Descriptive: d_ext (test, tau 0.5) versus d_train, L and PE\n')
    L.append('| PE | d_train | L2 | L4 | L8 | L12 | W624 | reading over L | Spearman(L, d_ext) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for k, v in dd['vs_L'].items():
        pe, dt = k.split('_dt')
        vals = [dd['per_cell'].get(f'{pe}_dt{dt}_{m}', {}).get('seeds') for m in C.MODELS]
        L.append(f"| {pe} | {dt} | " + ' | '.join(str(v_) for v_ in vals) + f" | {v['reading']} | {v['spearman']} |")
    L.append('\n| PE | model | d_train | excess d_ext - d_train (seed mean) | OLS slope of d_ext on d_train | reading |')
    L.append('|---|---|---|---|---|---|')
    for k, v in dd['vs_d_train'].items():
        pe, m = k.split('_')
        L.append(f"| {pe} | {m} | {v['d_train']} | {[None if e is None else round(e, 2) for e in v['excess_mean']]} | "
                 f"{None if v['ols_slope'] is None else round(v['ols_slope'], 3)} | {v['reading']} |")
    rp = dd['rope_minus_ape']
    L.append(f"\nRoPE - APE (paired over d_train x model x seed): n={rp['n_pairs']}, mean={rp['mean']}, "
             f"+{rp['n_pos']} / ={rp['n_zero']} / -{rp['n_neg']}, two-sided sign test p={rp['sign_test_p']}\n")

    L.append('## Per run\n')
    L.append('| run | dir | steps | params | GPU | train h | gate_a | d_ext_cal 0.5 [CI] | d_ext_test 0.5 [CI] | d_ext_test 0.9 | raw 0.5 | D_call | R | min/mean A_hand on B | gain | no harm | verify |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n, x in runs.items():
        de = x['d_ext']; P = x['policy_0.5']; Rr = P['R']
        mm = 'vacuous' if Rr['vacuous'] else f"{Rr['min_hand']:.3f}/{Rr['mean_hand']:.3f}"
        gn = '' if Rr['vacuous'] else f"{Rr['mean_gain']:.3f}"
        vv = VJ['gpu_verification'].get(n, {})
        L.append(f"| {n} | {x['dir']} | {x['steps']} | {x['params_nonembed']/1e6:.2f}M | {x['gpu'].replace('NVIDIA ', '')} | "
                 f"{x['train_h']:.2f} | {x['gate_a']} | {fmt_t(de['cal_0.5'])} {de['cal_0.5_ci']} | "
                 f"{fmt_t(de['test_0.5'])} {de['test_0.5_ci']} | {fmt_t(de['test_0.9'])} | {fmt_t(de['test_raw_0.5'])} | "
                 f"{P['D_call']} | {Rr['holds']} | {mm} | {gn} | {P['no_harm']} | {vv.get('pass')} |")

    def table(title, key_fn):
        L.append(f'\n## {title}\n')
        for dt in sorted({r['dtrain'] for r in grid}):
            dtest = C.dtest_of(dt)
            L.append(f'\nd_train = {dt}\n')
            L.append('| run | ' + ' | '.join(str(i) for i in range(1, dtest + 1)) + ' |')
            L.append('|---|' + '---|' * dtest)
            for n, x in runs.items():
                if x['dtrain'] == dt:
                    L.append(f'| {n} | ' + ' | '.join('-' if a is None else f'{a:.2f}' for a in key_fn(x)) + ' |')
    table('A_single(d), test, sequence success', lambda x: x['A_single'])
    table('A_hand(d), calibrated tau 0.5 policy, all calls', lambda x: x['policy_0.5']['A_hand'])
    table('A_hand(d), calibrated tau 0.5 policy, own targets only', lambda x: x['policy_0.5']['A_hand_own'])
    table('Calls per plan, tau 0.5 policy', lambda x: x['policy_0.5']['calls'])
    table('A_hand(d), calibrated tau 0.9 policy', lambda x: x['policy_0.9']['A_hand'])
    table('Always-split at D = d_train', lambda x: x[f"split{x['dtrain']}"]['A'])
    table('Always-split at D = 2', lambda x: x['split2']['A'])
    table('Per-pop accuracy, single call', lambda x: x['pop_acc'])
    table('Recency baseline sequence success', lambda x: x['recency_seq'])
    L.append('\nChance per pop = 1/8. Always-split curves for every D in 2..d_train are in summary.json (runs.*.splitD).')
    open(os.path.join(OUT, 'RESULTS.md.tmp'), 'w').write('\n'.join(L) + '\n')
    os.replace(os.path.join(OUT, 'RESULTS.md.tmp'), os.path.join(OUT, 'RESULTS.md'))


def write_table_tex(S, grid):
    """Cell summary for the appendix (seed means): d_ext_test, mean A_single and A_hand over B, calls on B."""
    L = ['% generated by code/analyze.py; status of the checks: ' + S['status'],
         '\\begin{tabular}{llrrrrr}', '\\toprule',
         'PE & $d_{\\mathrm{train}}$ & model & $d_{\\mathrm{ext}}$ & $\\bar A_{\\mathrm{single}}(B)$ & '
         '$\\bar A_{\\mathrm{hand}}(B)$ & calls on $B$ \\\\', '\\midrule']

    def mean(v):
        v = [x for x in v if x is not None]
        return None if not v else float(np.mean(v))

    def f(v, p=2):
        return '--' if v is None else f'{v:.{p}f}'
    for c, x in S['cells'].items():
        L.append(f"{'RoPE' if x['pe'] == 'rope' else 'APE'} & {x['dtrain']} & {x['model']} & {f(mean(x['d_ext_test']), 1)} & "
                 f"{f(mean(x['mean_single_B']))} & {f(mean(x['mean_hand_B']))} & {f(mean(x['mean_calls_B']), 1)} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(OUT, 'table_cells.tex'), 'w').write('\n'.join(L) + '\n')


def make_figures(S, grid, cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = S['runs']
    GRAY, BLUE, ORANGE = '#8a8986', '#2a78d6', '#eb6834'
    dts = sorted({r['dtrain'] for r in grid})
    cols = [(pe, m) for pe in C.PES for m in C.MODELS if any((pe, dt, m) in cells for dt in dts)]
    fig, axes = plt.subplots(len(dts), len(cols), figsize=(2.3 * len(cols), 2.0 * len(dts)), squeeze=False,
                             sharey=True)
    for i, dt in enumerate(dts):
        dtest = C.dtest_of(dt)
        x = np.arange(1, dtest + 1)
        for j, (pe, m) in enumerate(cols):
            ax = axes[i, j]
            names = [r['name'] for r in cells.get((pe, dt, m), []) if r['name'] in runs]
            ax.axvspan(0.5, dt + 0.5, color='#ececea', lw=0)
            if names:
                As = np.mean([runs[n]['A_single'] for n in names], axis=0)
                Ah = np.mean([runs[n]['policy_0.5']['A_hand'] for n in names], axis=0)
                for n in names:
                    ax.plot(x, runs[n]['A_single'], color=GRAY, lw=0.6, alpha=0.5)
                ax.plot(x, As, color=GRAY, lw=2, label='single call')
                ax.plot(x, Ah, color=BLUE, lw=2, label='split into calls')
                ax.axvline(np.mean([runs[n]['d_ext']['cal_0.5'][0] for n in names]), color=ORANGE, lw=1, ls='--')
            ax.set_ylim(-0.03, 1.03); ax.set_xlim(0.5, dtest + 0.5)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{'RoPE' if pe == 'rope' else 'APE'} {m}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f'd_train={dt}\nsuccess', fontsize=8)
            if i == len(dts) - 1:
                ax.set_xlabel('test depth d', fontsize=8)
    axes[0, 0].legend(fontsize=6, frameon=False, loc='lower left')
    fig.suptitle(f"Nested plans: seed-mean sequence success; gray band = training range, "
                 f"dashed = d_ext_cal", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'curves.png'), dpi=120); plt.close(fig)

    PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    fig, axes = plt.subplots(1, len(C.PES), figsize=(5.2 * len(C.PES), 3.6), squeeze=False, sharey=True)
    for j, pe in enumerate(C.PES):
        ax = axes[0, j]
        ax.plot(dts, dts, color='#bdbcb8', lw=1, ls=':', label='d_ext = d_train')
        for k, m in enumerate(C.MODELS):
            xs, mu, lo, hi = [], [], [], []
            for dt in dts:
                v = [runs[r['name']]['d_ext']['test_0.5'][0] for r in cells.get((pe, dt, m), []) if r['name'] in runs]
                if v:
                    xs.append(dt + (k - 2) * 0.08); mu.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v))
            if xs:
                ax.errorbar(xs, mu, yerr=[lo, hi], color=PAL[k], lw=2, marker='o', ms=5, capsize=2, label=m)
        ax.set_title('RoPE (GPT-NeoX)' if pe == 'rope' else 'APE (GPT-Neo)', fontsize=10)
        ax.set_xlabel('training depth d_train'); ax.set_xticks(dts)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.set_ylabel('d_ext (test, tau 0.5)')
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle('Calibrated extrapolation depth: seed mean, bars = seed range', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'd_ext.png'), dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
