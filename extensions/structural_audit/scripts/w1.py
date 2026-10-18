"""Exact W1 between the groups of each family and within-group baselines (App. M.2), with the estimator of
src/w1_robust.py (App. G): pooled z-scoring, uniform weights, Euclidean cost, ot.emd2, 10 seeds with
np.random.default_rng(seed), between = independent samples without replacement, within = disjoint halves of a
permutation.  Pool per group: all units, or a random 20,000 rows (seed 12345) if larger; groups with fewer than 200
units are dropped; all pairs of the remaining groups of a family are formed.
N per pair = min(2000, floor(n_a/2), floor(n_b/2)) for the between and both within computations.
Variants: main (z-scores pooled over the groups of the family), noS (main without the surface coordinates S_CLASS),
pairz (z-scores pooled over the two groups of each pair).
Restart-safe: every finished task is appended to results/tasks_<variant>.jsonl and skipped on restart; the design
(groups, units, pairs) is written to results/design_<variant>.json.
Usage: python w1.py <variant: main|noS|pairz> [n_procs]
"""
import json, os, sys, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd, ot
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(HERE, "features"); RES = os.path.join(HERE, "results")
NMAX, POOL_CAP, POOL_SEED, SEEDS, MIN_UNITS = 2000, 20000, 12345, list(range(10)), 200
sys.path.insert(0, os.path.join(HERE, "scripts"))

ALF = ["A_S2_actions", "A_T4_goto", "A_T5_locations", "A_T6_loc_revisit", "A_T8_loc_cycle_rank", "A_N1_max_open_depth",
       "A_T2_mean_open_depth", "A_C1_push_rate", "A_C2_pop_rate", "A_C7_search_misses", "A_C15_actions_before_take",
       "A_I1_manip", "A_I1_transform", "A_I2_verb_entropy"]
ETO = ["B_S2_actions", "B_T5_states", "B_T6_state_revisit", "B_T8_state_cycle_rank", "B_N1_max_depth", "B_T2_mean_depth",
       "B_C1_push_rate", "B_C2_pop_rate", "B_C7_neg_feedback", "B_I1_nav", "B_I1_manip", "B_I2_verb_entropy"]
HAR = ["T1_max_depth", "T2_mean_depth", "T3_sd_depth", "T4_tool_calls", "T5_distinct_files", "T6_file_reopen",
       "T6a_repeat_action", "C1_push_rate", "C2_pop_rate", "C7_failed_edits", "C9_failed_edit_cascade",
       "C14_run_calls", "C15_first_edit_pos", "I1_read_calls", "I1_write_calls", "I3_error_rate"]
VLA = ["S2_steps", "S_inert_share", "T4_segments", "T5_distinct_prims", "T6_seg_revisit", "T8_cycle_rank",
       "N1_max_hold", "N_holds", "T2_hold_share", "C_reversals", "C9_longest_stall", "C15_steps_before_grasp",
       "I1_translation", "I1_rotation", "I1_gripper", "I2_prim_entropy"]
BAB = ["chi1_maxdepth", "chi2_meandepth", "chi3_sddepth", "chi4_volume", "chi5_deepen", "chi6_backtrack",
       "chi8_explore", "chi9_explore_rate", "chi11_manip", "chi12_nav", "ref_len", "ref_turns", "n_objs"]
S_CLASS = {"A": ["A_S2_actions"], "B": ["B_S2_actions"], "B2": ["B_S2_actions"], "C_H1": [], "C_H2": [],
           "D": ["S2_steps", "S_inert_share"], "E": ["ref_len"]}
ALF_GROUPS = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
VLA_GROUPS = ["PICKPLACE", "OPEN", "CLOSE", "FOLD", "SWEEP", "FLIP", "TURN"]
BAB_GROUPS = ["goto", "pickup", "open", "putnext", "pick up seq go to"]


def load_families():
    fam = {}
    eto = pd.read_csv(os.path.join(F, "eto_features.csv"))
    a = eto[eto.env == "alfworld"]
    fam["A"] = dict(desc="ALFWorld task types (ETO expert trajectories)", cols=ALF,
                    groups={g: a[a.group_alf == g] for g in ALF_GROUPS}, unit=None)
    fam["B"] = dict(desc="Environments within one agent corpus (ETO)", cols=ETO,
                    groups={g: eto[eto.env == g] for g in ["alfworld", "sciworld", "webshop"]}, unit=None)
    w = eto[eto.env == "webshop"]
    fam["B2"] = dict(desc="WebShop demonstrations: human vs GPT-4 (ETO)", cols=ETO,
                     groups={g: w[w.group_src == g] for g in ["human", "gpt"]}, unit=None)
    h = pd.read_csv(os.path.join(F, "harness_features.csv"))
    G = ["SWEagent_llama70b", "OpenHands_qwen3coder480b"]
    shared = set(h[h.group == G[0]].instance_id) & set(h[h.group == G[1]].instance_id)
    hs = h[h.instance_id.isin(shared)]
    fam["C_H1"] = dict(desc="Harness: SWE-agent vs OpenHands, resolved runs, issues resolved by both", cols=HAR,
                       groups={g: hs[hs.group == g] for g in G}, unit="instance_id")
    fam["C_H2"] = dict(desc="Harness: SWE-agent vs OpenHands, resolved runs, all resolved issues", cols=HAR,
                       groups={g: h[h.group == g] for g in G}, unit="instance_id")
    v = pd.read_csv(os.path.join(F, "vla_features.csv"))
    fam["D"] = dict(desc="Bridge/ECoT demonstrations by skill (VLA)", cols=VLA,
                    groups={g: v[v.skill == g] for g in VLA_GROUPS}, unit=None)
    bb = pd.concat([pd.read_csv(os.path.join(F, "babyai_%s.csv" % k.replace(" ", "_"))) for k in BAB_GROUPS])
    bb = bb[(bb.bot_error.fillna("") == "") & (bb.chi10_oracle_success == 1.0)]
    fam["E"] = dict(desc="BabyAI task types (oracle bot subgoal stacks, fresh levels)", cols=BAB,
                    groups={g: bb[bb.kind == g] for g in BAB_GROUPS}, unit=None)
    return fam


def prepare(fam, variant):
    """Per family: candidate pools, coordinate arrays (rows) and unit index lists; all pairs of groups."""
    P = {}
    for name, f in fam.items():
        cols = [c for c in f["cols"] if not (variant == "noS" and c in S_CLASS[name])]
        pools = {}
        for g, d in f["groups"].items():
            d = d.reset_index(drop=True)
            if len(d) > POOL_CAP:
                d = d.iloc[np.sort(np.random.default_rng(POOL_SEED).choice(len(d), POOL_CAP, replace=False))].reset_index(drop=True)
            pools[g] = d
        units = {}
        for g, d in pools.items():
            if f["unit"]:
                units[g] = [np.flatnonzero(d[f["unit"]].values == u) for u in sorted(d[f["unit"]].unique())]
            else:
                units[g] = None
        nunits = {g: (len(units[g]) if units[g] is not None else len(pools[g])) for g in pools}
        groups = [g for g in f["groups"] if nunits[g] >= MIN_UNITS]
        X = {g: pools[g][cols].to_numpy(float) for g in groups}
        pairs = list(itertools.combinations(groups, 2))
        P[name] = dict(cols=cols, X=X, units=units, nunits=nunits, groups=groups, pairs=pairs, desc=f["desc"],
                       excluded=[g for g in f["groups"] if g not in groups])
    return P


def zstats(Xs):
    allv = np.vstack(Xs); mu, sd = allv.mean(0), allv.std(0); sd[sd == 0] = 1
    return mu, sd


def draw(units, idx, rng):
    """Row indices for sampled units: rows themselves, or one random trajectory per sampled issue."""
    if units is None:
        return idx
    return np.array([units[u][rng.integers(len(units[u]))] for u in idx])


def emd(A, B):
    N = len(A); w = np.ones(N) / N
    c, log = ot.emd2(w, w, cdist(A, B), numItermax=10_000_000, log=True)
    return float(c), (log.get("warning") or "")


STATE = {}


def init(variant):
    fam = load_families(); P = prepare(fam, variant)
    Z = {}
    if variant != "pairz":
        for name, p in P.items():
            mu, sd = zstats([p["X"][g] for g in p["groups"]])
            Z[name] = {g: (p["X"][g] - mu) / sd for g in p["groups"]}
    STATE.update(P=P, Z=Z, variant=variant)


def zarrays(name, a, b):
    P = STATE["P"][name]
    if STATE["variant"] == "pairz":  # statistics pooled over the two groups of the pair only
        mu, sd = zstats([P["X"][a], P["X"][b]])
        return (P["X"][a] - mu) / sd, (P["X"][b] - mu) / sd
    return STATE["Z"][name][a], STATE["Z"][name][b]


def run_task(t):
    kind, name, a, b, N, s = t
    P = STATE["P"][name]
    Za, Zb = zarrays(name, a, b)
    rng = np.random.default_rng(s)
    if kind == "between":
        ia = rng.choice(P["nunits"][a], N, replace=False); ib = rng.choice(P["nunits"][b], N, replace=False)
        A, B = Za[draw(P["units"][a], ia, rng)], Zb[draw(P["units"][b], ib, rng)]
    else:  # within group a: two disjoint halves of one permutation (b is the pair partner, used only for 'pairz')
        perm = rng.permutation(P["nunits"][a])
        A = Za[draw(P["units"][a], perm[:N], rng)]; B = Za[draw(P["units"][a], perm[N:2 * N], rng)]
    t0 = time.time(); c, warn = emd(A, B)
    return dict(kind=kind, family=name, a=a, b=b, N=int(N), seed=int(s), w1=c, warning=warn, sec=round(time.time() - t0, 3))


def tkey(t):
    kind, name, a, b, N, s = t
    if kind == "within" and STATE["variant"] != "pairz":
        b = None  # with family z-scoring the within baseline does not depend on the partner
    return "|".join(map(str, (kind, name, a, b, N, s)))


def tasks_for(P):
    T, seen = [], set()
    for name, p in P.items():
        for a, b in p["pairs"]:
            N = min(NMAX, p["nunits"][a] // 2, p["nunits"][b] // 2)
            for s in SEEDS:
                for t in (("between", name, a, b, N, s), ("within", name, a, b, N, s), ("within", name, b, a, N, s)):
                    k = tkey(t)
                    if k not in seen:
                        seen.add(k); T.append(t)
    return T


def main():
    variant = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.makedirs(RES, exist_ok=True)
    init(variant)
    P = STATE["P"]
    meta = {name: dict(desc=p["desc"], cols=p["cols"], groups=p["groups"], excluded=p["excluded"],
                       nunits={g: int(v) for g, v in p["nunits"].items()},
                       nrows={g: int(len(p["X"][g])) for g in p["groups"]}, pairs=p["pairs"]) for name, p in P.items()}
    json.dump(meta, open(os.path.join(RES, "design_%s.json" % variant), "w"), indent=1)
    out = os.path.join(RES, "tasks_%s.jsonl" % variant)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line); done.add(r["key"])
            except Exception:
                pass
    todo = [t for t in tasks_for(P) if tkey(t) not in done]
    todo.sort(key=lambda t: -t[4])
    print(variant, "tasks total", len(tasks_for(P)), "todo", len(todo), flush=True)
    t0 = time.time()
    with open(out, "a") as fh, Pool(nproc, initializer=init, initargs=(variant,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, todo, chunksize=1)):
            r["key"] = tkey((r["kind"], r["family"], r["a"], r["b"], r["N"], r["seed"]))
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print(i + 1, "done %.0fs" % (time.time() - t0), flush=True)
    print("finished", variant, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
