"""Families A, B, B2 of App. M.2: task-side projections on ETO expert trajectories (agent-eto/eto-sft-trajectory, Apache-2.0;
ALFWorld MIT, ScienceWorld Apache-2.0, WebShop MIT).

Only the action strings and the environment observations are used (the GPT-written 'Thought:' text and the WebShop
'reward' field are never read into a coordinate).  Two projections (coordinate ids follow the catalogue of App. M.1):

Phi_ALF (family A, ALFWorld only, 14 coordinates):
  S2 n_actions | T4 n_goto (location references) | T5 distinct locations | T6 location revisit ratio 1-T5/T4 |
  T8 cycle rank of the undirected location graph (start node + go-to targets, consecutive visits, no multi-edges) |
  N1 max open-context depth (holding an object + receptacles opened and not closed) | T2 mean open-context depth |
  C1 push rate (depth increases / n) | C2 pop rate (depth decreases / n) |
  C7 search misses (receptacle checks made empty-handed whose go-to (+ immediately following open) observations do
     not show the target object type of the game file) | C15 actions before the first successful take |
  I1 manipulation actions (take/put/open/close) | I1 transformation actions (clean/heat/cool/use/toggle/slice) |
  I2 entropy of the action-verb distribution.
Phi_ETO (families B and B2; one feature map phi, per-environment adapters pi_D; 12 coordinates):
  S2 n_actions | T5 distinct states | T6 state revisit ratio 1-T5/(state-entering events) | T8 cycle rank of the state
  graph | N1 max depth | T2 mean depth | C1 push rate | C2 pop rate | C7 negative-feedback observations |
  I1 navigation actions | I1 manipulation actions | I2 verb entropy.
  States / depth: ALFWorld location / open contexts (as above); ScienceWorld room (teleport/go to) / held + opened +
  activated-or-connected counters; WebShop page (search, results:q:p, item:asin, item:asin:subpage) / page depth
  0-3.  Negative feedback: ALFWorld 'Nothing happens'; ScienceWorld 'No known action matches', 'Ambiguous request',
  "appears broken, and can't be"; WebShop 'invalid'.
Output: features/eto_features.csv. Idempotent.
"""
import json, math, os, re, sys
from collections import Counter
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eto", "data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")
R_ACT = re.compile(r"Action:\s*(.*)", re.S)

ALF_COLS = ["S2_actions", "T4_goto", "T5_locations", "T6_loc_revisit", "T8_loc_cycle_rank", "N1_max_open_depth",
            "T2_mean_open_depth", "C1_push_rate", "C2_pop_rate", "C7_search_misses", "C15_actions_before_take",
            "I1_manip", "I1_transform", "I2_verb_entropy"]
ETO_COLS = ["S2_actions", "T5_states", "T6_state_revisit", "T8_state_cycle_rank", "N1_max_depth", "T2_mean_depth",
            "C1_push_rate", "C2_pop_rate", "C7_neg_feedback", "I1_nav", "I1_manip", "I2_verb_entropy"]


def entropy(c):
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values()) if n else 0.0


def cycle_rank(seq, start="__start__"):
    nodes = [start] + list(seq)
    V = set(nodes)
    E = {frozenset((a, b)) for a, b in zip(nodes[:-1], nodes[1:]) if a != b}
    return len(E) - len(V) + 1  # one connected component (a walk)


def steps_of(conv):
    """(action, observation) pairs after the instruction/'OK'/task turns."""
    out = []
    for j in range(3, len(conv)):
        if conv[j]["from"] != "gpt":
            continue
        m = R_ACT.search(conv[j]["value"] or "")
        a = m.group(1).strip().split("\n")[0].strip() if m else ""
        o = conv[j + 1]["value"] if j + 1 < len(conv) and conv[j + 1]["from"] == "human" else ""
        o = re.sub(r"^\s*Observation:\s*", "", o or "")
        out.append((a, o))
    return out


def depth_stats(D):
    n = len(D)
    Dfull = [0] + D
    up = sum(1 for a, b in zip(Dfull[:-1], Dfull[1:]) if b > a)
    dn = sum(1 for a, b in zip(Dfull[:-1], Dfull[1:]) if b < a)
    return (max(Dfull), float(np.mean(D)) if D else 0.0, up / max(1, n), dn / max(1, n))


# ------------------------------------------------------------------ ALFWorld
ALF_MANIP = {"take", "put", "open", "close"}
ALF_TRANS = {"clean", "heat", "cool", "use", "toggle", "slice"}


def alf_target(game_file):
    parts = game_file.split("/")[3].split("-")
    return parts[0], re.sub(r"sliced$", "", parts[1].lower())


def alf_walk(steps):
    """Shared ALFWorld adapter: returns per-step depth, locations, verbs, holding flags."""
    holding, opened = False, set()
    D, locs, verbs, hold_before = [], [], [], []
    for a, o in steps:
        v = a.split(" ")[0].lower() if a else "none"
        verbs.append(v); hold_before.append(holding)
        ok = not o.startswith("Nothing happens")
        if v == "go" and a.lower().startswith("go to "):
            locs.append(a[6:].strip().lower())
        elif v == "take" and ok and o.startswith("You pick up"):
            holding = True
        elif v == "put" and ok and o.startswith("You put"):
            holding = False
        elif v == "open" and ok and o.startswith("You open"):
            opened.add(a[5:].strip().lower())
        elif v == "close" and ok and o.startswith("You close"):
            opened.discard(a[6:].strip().lower())
        D.append(int(holding) + len(opened))
    return D, locs, verbs, hold_before


def phi_alf(steps, target):
    D, locs, verbs, hold_before = alf_walk(steps)
    n = len(steps)
    N1, T2, C1, C2 = depth_stats(D)
    rt = re.compile(r"\b" + re.escape(target) + r" \d+")
    misses = 0
    for i, (a, o) in enumerate(steps):
        if verbs[i] == "go" and not hold_before[i]:
            seen = o
            if i + 1 < n and verbs[i + 1] == "open" and steps[i + 1][0][5:].strip().lower() == a[6:].strip().lower():
                seen += " " + steps[i + 1][1]
            misses += 0 if rt.search(seen.lower()) else 1
    first_take = next((i for i, (a, o) in enumerate(steps) if verbs[i] == "take" and o.startswith("You pick up")), n)
    T4 = len(locs); T5 = len(set(locs))
    return [n, T4, T5, 1 - T5 / max(1, T4), cycle_rank(locs), N1, T2, C1, C2, misses, first_take,
            sum(v in ALF_MANIP for v in verbs), sum(v in ALF_TRANS for v in verbs), entropy(Counter(verbs))]


def eto_alf(steps):
    D, locs, verbs, _ = alf_walk(steps)
    N1, T2, C1, C2 = depth_stats(D)
    neg = sum(o.startswith("Nothing happens") for _, o in steps)
    nav = sum(v == "go" for v in verbs); man = sum(v in ALF_MANIP or v in ALF_TRANS for v in verbs)
    U = len(set(locs))
    return [len(steps), U, 1 - U / max(1, len(locs)), cycle_rank(locs), N1, T2, C1, C2, neg, nav, man, entropy(Counter(verbs))]


# ------------------------------------------------------------------ ScienceWorld
SW_NAV = ("teleport to", "go to")
SW_MANIP = {"pick", "move", "drop", "pour", "connect", "disconnect", "activate", "deactivate", "open", "close", "mix",
            "use", "put", "dunk", "eat", "flush", "plant"}
R_SW_NEG = re.compile(r"No known action matches|Ambiguous request|appears broken, and can't be")


def sw_verb(a):
    al = a.lower()
    for p in ("teleport to", "go to", "look around", "look at", "pick up", "focus on"):
        if al.startswith(p):
            return p
    return al.split(" ")[0] if al else "none"


def eto_sci(steps):
    held = opened = active = 0
    D, states, verbs = [], [], []
    for a, o in steps:
        v = sw_verb(a); verbs.append(v)
        ok = not R_SW_NEG.search(o[:200])
        if v in SW_NAV:
            states.append(a.lower()[len(v):].strip())
        elif ok:
            if v == "pick up": held += 1
            elif v in ("move", "drop", "pour", "put") and held > 0: held -= 1
            elif v == "open": opened += 1
            elif v == "close" and opened > 0: opened -= 1
            elif v in ("activate", "connect"): active += 1
            elif v in ("deactivate", "disconnect") and active > 0: active -= 1
        D.append(held + opened + active)
    N1, T2, C1, C2 = depth_stats(D)
    neg = sum(bool(R_SW_NEG.search(o[:200])) for _, o in steps)
    nav = sum(v in SW_NAV for v in verbs)
    man = sum((v.split(" ")[0] in SW_MANIP) for v in verbs if v not in SW_NAV)
    U = len(set(states))
    return [len(steps), U, 1 - U / max(1, len(states)), cycle_rank(states), N1, T2, C1, C2, neg, nav, man,
            entropy(Counter(verbs))]


# ------------------------------------------------------------------ WebShop
WS_SUB = {"description", "features", "reviews", "attributes"}


def eto_web(steps):
    page, depth, query, pnum, item = "search", 0, "", 1, ""
    D, states, verbs = [], [], []
    for a, o in steps:
        al = a.strip().lower()
        neg = "invalid" in o[:200].lower()
        if al.startswith("search["):
            v = "search"; query = al[7:-1]; pnum = 1; page, depth = "results:%s:1" % query, 1
            states.append(page)
        elif al.startswith("click["):
            arg = al[6:-1] if al.endswith("]") else al[6:]
            if arg == "back to search":
                v = "back"; page, depth = "search", 0; states.append(page)
            elif arg == "next >":
                v = "next"; pnum += 1; page, depth = "results:%s:%d" % (query, pnum), 1; states.append(page)
            elif arg == "< prev":
                v = "prev"
                if depth >= 2: page, depth = "results:%s:%d" % (query, pnum), 1
                else: pnum = max(1, pnum - 1); page, depth = "results:%s:%d" % (query, pnum), 1
                states.append(page)
            elif re.fullmatch(r"b[0-9a-z]{9}", arg):
                v = "item"; item = arg; page, depth = "item:" + arg, 2; states.append(page)
            elif arg in WS_SUB:
                v = "sub:" + arg; page, depth = "item:%s:%s" % (item, arg), 3; states.append(page)
            elif arg == "buy now":
                v = "buy"
            else:
                v = "option"
        else:
            v = "none"
        verbs.append(v)
        D.append(depth)
    N1, T2, C1, C2 = depth_stats(D)
    nav = sum(v in ("search", "back", "next", "prev", "item") or v.startswith("sub:") for v in verbs)
    man = sum(v in ("buy", "option") for v in verbs)
    neg = sum("invalid" in o[:200].lower() for _, o in steps)
    U = len(set(states))
    return [len(steps), U, 1 - U / max(1, len(states)), cycle_rank(states), N1, T2, C1, C2, neg, nav, man,
            entropy(Counter(verbs))]


def main(limit=None):
    path = os.path.join(OUT, "eto_features.csv")
    if os.path.exists(path) and limit is None:
        print("exists", path); return
    rows = []
    for env in ["alfworld", "sciworld", "webshop"]:
        data = json.load(open(os.path.join(RAW, "%s_sft.json" % env)))
        if limit: data = data[:limit]
        for k, x in enumerate(data):
            steps = steps_of(x["conversations"])
            r = dict(env=env, uid="%s:%s" % (env, x.get("id", k)), group_alf="", group_src="")
            if env == "alfworld":
                atype, target = alf_target(x["game_file"])
                r["group_alf"] = atype; r["alf_target"] = target
                r.update({"A_" + c: v for c, v in zip(ALF_COLS, phi_alf(steps, target))})
                feats = eto_alf(steps)
            elif env == "sciworld":
                r["sci_task"] = str(x["id"]).split("_")[0]
                feats = eto_sci(steps)
            else:
                r["group_src"] = x["source"]
                feats = eto_web(steps)
            r.update({"B_" + c: v for c, v in zip(ETO_COLS, feats)})
            r["n_parsed_actions"] = len(steps); r["n_empty_actions"] = sum(1 for a, _ in steps if not a)
            rows.append(r)
    df = pd.DataFrame(rows)
    if limit:
        return df
    df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    print("wrote", path, df.shape, df.env.value_counts().to_dict())


if __name__ == "__main__":
    main()
