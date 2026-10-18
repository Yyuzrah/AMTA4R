# 34 structural coordinates of a Countdown (Stream of Search) reference trajectory, grouped into the classes of the
# coordinate catalogue. chi_1..chi_12 are recomputed with the definitions of src/distribution_metrics.py (gate.py
# checks the agreement); the other 22 are the further coordinates of App. M.1.1, added in the class order of CLASSES.
import re
import numpy as np

RE_NODE = re.compile(r"Node #(\d+(?:,\d+)*)")

# coordinate name -> class: TOP topology, CTRL control flow, TERM recovery and termination, MEM memory,
# NEST nesting, INTER interaction (actions and feedback), SURF surface
CLASSES = {
    "TOP": ["chi2_mean_depth", "chi3_std_depth", "chi4_volume", "unique_nodes", "revisit_ratio", "branch_mean", "branch_max"],
    "CTRL": ["chi5_deepen", "chi6_backtrack", "chi7_lateral", "pop_mean", "cross_subtree_rate", "backtrack_episodes"],
    "TERM": ["chi8_prunes", "chi9_prune_rate", "chi10_goal", "fail_streak_max", "goal_pos_frac", "post_goal_tokens", "goal_count"],
    "MEM": ["return_dist_mean", "return_dist_max", "live_ctx_mean"],
    "NEST": ["chi1_max_depth", "pop_max", "live_ctx_max", "restate_ratio"],
    "INTER": ["chi11_mult", "chi12_add", "n_actions", "actions_per_state", "feedback_ratio"],
    "SURF": ["len_tokens", "n_lines"],
}
# TERM is the recovery-and-termination part of control flow, kept as a class of its own.
NAMES = [n for c in CLASSES.values() for n in c]
F12_NAMES = ["chi1_max_depth", "chi2_mean_depth", "chi3_std_depth", "chi4_volume", "chi5_deepen", "chi6_backtrack",
             "chi7_lateral", "chi8_prunes", "chi9_prune_rate", "chi10_goal", "chi11_mult", "chi12_add"]


def _line_token_index(text, tok):
    """token index at which each line starts (GPT-Neo tokenizer), plus total token count"""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    starts = np.array([s for s, _ in enc["offset_mapping"]])
    line_starts, pos = [], 0
    for line in text.split("\n"):
        line_starts.append(pos); pos += len(line) + 1
    idx = np.searchsorted(starts, np.array(line_starts), side="right") - 1
    return np.clip(idx, 0, None), len(starts)


def extract(text, tok):
    f = dict.fromkeys(NAMES, 0.0)
    lines = text.split("\n")
    ltok, ntok = _line_token_index(text, tok)
    f["len_tokens"] = float(ntok); f["n_lines"] = float(len([l for l in lines if l.strip()]))
    # ---- chi_8..chi_12, defined as in src/distribution_metrics.py ----
    f["chi11_mult"] = text.count("*") + text.count("/")
    f["chi12_add"] = text.count("+") + text.count("-")
    f["chi10_goal"] = 1.0 if "Goal Reached" in text else 0.0
    prunes = len(re.findall(r"unequal: No Solution", text)); f["chi8_prunes"] = prunes
    n_gen = len(re.findall(r"Generated Node #", text))
    if n_gen: f["chi9_prune_rate"] = prunes / n_gen
    # ---- event stream ----
    refs, ref_tok, kinds = [], [], []          # node references in order, token position, 'G'/'M'
    n_state = n_act = 0; goal_tok = []; streak = best_streak = 0
    for li, line in enumerate(lines):
        m = RE_NODE.search(line)
        if m and (line.startswith("Generated Node") or line.startswith("Moving to Node")):
            refs.append(tuple(map(int, m.group(1).split(",")))); ref_tok.append(ltok[li]); kinds.append(line[0])
            if line.startswith("Generated"): streak = 0
        elif line.startswith("Current State"):
            n_state += 1
        elif line.startswith("Exploring Operation"):
            n_act += 1
        if "unequal: No Solution" in line:
            streak += 1; best_streak = max(best_streak, streak)
        elif "equal: Goal Reached" in line:
            goal_tok.append(ltok[li])
    f["n_actions"] = n_act; f["fail_streak_max"] = best_streak
    if n_act: f["feedback_ratio"] = prunes / n_act
    if n_state: f["actions_per_state"] = n_act / n_state
    f["goal_count"] = len(goal_tok)
    if goal_tok:
        f["goal_pos_frac"] = goal_tok[0] / max(ntok, 1); f["post_goal_tokens"] = ntok - goal_tok[0]
    else:
        f["goal_pos_frac"] = 1.0
    if not refs:
        return f
    N = len(refs); D = np.array([len(r) for r in refs], dtype=float)
    f["chi1_max_depth"] = D.max(); f["chi2_mean_depth"] = D.mean(); f["chi3_std_depth"] = D.std(); f["chi4_volume"] = N
    f["restate_ratio"] = n_state / N
    uniq = set(refs); f["unique_nodes"] = len(uniq); f["revisit_ratio"] = 1 - len(uniq) / N
    children = {}
    for r in uniq:
        if len(r) > 1: children.setdefault(r[:-1], set()).add(r)
    if children:
        b = [len(v) for v in children.values()]; f["branch_mean"] = float(np.mean(b)); f["branch_max"] = float(max(b))
    # control flow
    if N > 1:
        dD = np.diff(D); M = N - 1
        f["chi5_deepen"] = float((dD > 0).sum() / M); f["chi6_backtrack"] = float((dD < 0).sum() / M)
        f["chi7_lateral"] = float((dD == 0).sum() / M)
        pops = -dD[dD < 0]
        if len(pops): f["pop_mean"] = float(pops.mean()); f["pop_max"] = float(pops.max())
        cross = [(refs[i][:-1] != refs[i + 1][:-1]) for i in range(M) if dD[i] == 0 and len(refs[i]) > 1]
        f["cross_subtree_rate"] = float(sum(cross) / M)
        up = (dD < 0).astype(int); f["backtrack_episodes"] = float(((up[1:] == 1) & (up[:-1] == 0)).sum() + up[0])
    # memory: return distance = tokens since the previous reference to the node returned to (moves that are not
    # the immediate descent into a just-generated child); live contexts = nodes referenced before and after step i
    last_seen, ret = {}, []
    for i, r in enumerate(refs):
        if kinds[i] == "M" and r in last_seen and not (i > 0 and refs[i - 1] == r):
            ret.append(ref_tok[i] - last_seen[r])
        last_seen[r] = ref_tok[i]
    if ret: f["return_dist_mean"] = float(np.mean(ret)); f["return_dist_max"] = float(max(ret))
    first, last = {}, {}
    for i, r in enumerate(refs):
        first.setdefault(r, i); last[r] = i
    delta = np.zeros(N + 1)
    for r in uniq:
        if last[r] > first[r]:
            delta[first[r] + 1] += 1; delta[last[r]] -= 1   # open strictly between first and last reference
    live = np.cumsum(delta)[:N]
    f["live_ctx_max"] = float(live.max()); f["live_ctx_mean"] = float(live.mean())
    return f


def vec(d, names=NAMES):
    return np.array([d[n] for n in names], dtype=np.float64)
