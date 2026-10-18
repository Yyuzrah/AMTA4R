"""Family D of App. M.2: task-side projection Phi_VLA on ECoT-annotated Bridge demonstrations
(HF Embodied-CoT/embodied_features_bridge, MIT; BridgeData V2 demonstrations, CC-BY-4.0).

Units: demonstration episodes that carry ECoT reasoning annotations and a non-empty human language instruction.
Coordinates are computed from the per-step movement primitives ('features.move_primitive', derived by ECoT from the
robot's proprioception), which are the 'move' layer of the embodied chain of thought; LLM-written plan/subtask text
is not used.  A primitive string is split at ', ' into components: translation ('move <dirs>'), rotation
('rotate ...'/'tilt ...'), gripper ('open gripper'/'close gripper'), 'stop'.
Hold context: 'close gripper' opens it (push), 'open gripper' closes it (pop).
Phi_VLA (16): S2 control steps H | S_inert share of 'stop' steps | T4 primitive segments (maximal runs of one
primitive string) | T5 distinct primitive strings | T6 segment revisit ratio 1-T5/T4 | T8 cycle rank of the undirected
segment-transition graph | N1 max hold depth | N_holds hold contexts opened (grasp events) | T2 mean hold depth (share
of steps holding) | C_rev translation reversals (consecutive translational segments with an opposite direction
component) | C9 longest run of 'stop' steps | C15 steps before the first grasp (H if none) | I1 translation steps |
I1 rotation steps | I1 gripper steps | I2 entropy of the primitive-string distribution over steps.
Skill group from the language instruction (lower-cased), first matching rule in this order:
  FOLD (un)fold | SWEEP sweep/wipe/brush/scrub | OPEN open | CLOSE close/closed/closing/shut |
  FLIP flip/upright/topple/knock/upside/turn over/stand | TURN turn/lever/faucet/knob/twist |
  PICKPLACE put/place/move/take/pick/pour/stack/remove/grab/drop/transfer/bring/get/lift | OTHER (not used).
Output: features/vla_features.csv. Idempotent.
"""
import json, math, os, re, sys
from collections import Counter
import numpy as np, pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ecot", "embodied_features_bridge.json")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")
COLS = ["S2_steps", "S_inert_share", "T4_segments", "T5_distinct_prims", "T6_seg_revisit", "T8_cycle_rank",
        "N1_max_hold", "N_holds", "T2_hold_share", "C_reversals", "C9_longest_stall", "C15_steps_before_grasp",
        "I1_translation", "I1_rotation", "I1_gripper", "I2_prim_entropy"]
RULES = [("FOLD", r"\b(un)?fold"), ("SWEEP", r"\b(sweep|wipe|brush|scrub)"), ("OPEN", r"\bopen"),
         ("CLOSE", r"\b(close|closed|closing|shut)\b"), ("FLIP", r"\b(flip|upright|topple|knock|upside|turn over|stand)"),
         ("TURN", r"\b(turn|lever|faucet|knob|twist)"),
         ("PICKPLACE", r"\b(put|place|move|take|pick|pour|stack|remove|grab|drop|transfer|bring|get|lift)")]
OPP = {"left": "right", "right": "left", "forward": "backward", "backward": "forward", "up": "down", "down": "up"}


def skill(instr):
    s = (instr or "").strip().lower()
    for name, pat in RULES:
        if re.search(pat, s):
            return name
    return "OTHER"


def stream_top_level(path, chunk=1 << 25):
    """Yield (key, value) pairs of a top-level JSON object without loading it whole."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf8") as f:
        buf = f.read(chunk); pos = buf.index("{") + 1
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \n\r\t,":
                    pos += 1
                if pos < len(buf): break
                more = f.read(chunk)
                if not more: return
                buf = buf[pos:] + more; pos = 0
            if buf[pos] == "}": return
            while True:
                try:
                    key, p2 = dec.raw_decode(buf, pos)
                    while buf[p2] in " \n\r\t": p2 += 1
                    assert buf[p2] == ":"; p2 += 1
                    while buf[p2] in " \n\r\t": p2 += 1
                    val, p3 = dec.raw_decode(buf, p2)
                    break
                except (json.JSONDecodeError, IndexError):
                    more = f.read(chunk)
                    if not more: raise
                    buf = buf[pos:] + more; pos = 0
            yield key, val
            pos = p3
            if pos > (1 << 26):
                buf = buf[pos:]; pos = 0


def comps(p):
    c = [x.strip() for x in (p or "").lower().split(",") if x.strip()]
    tr = set(); rot = False; grip = None; stop = False
    for x in c:
        if x.startswith("move"):
            tr |= {w for w in x.split()[1:] if w in OPP}
        elif x.startswith("rotate") or x.startswith("tilt"):
            rot = True
        elif x == "close gripper":
            grip = "close"
        elif x == "open gripper":
            grip = "open"
        elif x == "stop":
            stop = True
    return tr, rot, grip, stop


def phi_vla(prims):
    H = len(prims)
    segs = []
    for p in prims:
        if segs and segs[-1] == p: continue
        segs.append(p)
    T4 = len(segs); T5 = len(set(segs))
    V = set(segs); E = {frozenset((a, b)) for a, b in zip(segs[:-1], segs[1:]) if a != b}
    T8 = len(E) - len(V) + 1
    C = [comps(p) for p in prims]
    hold, holds, maxhold, held_steps, first_grasp = 0, 0, 0, 0, H
    for t, (tr, rot, grip, stop) in enumerate(C):
        if grip == "close" and hold == 0:
            hold = 1; holds += 1; first_grasp = min(first_grasp, t)
        elif grip == "open" and hold == 1:
            hold = 0
        maxhold = max(maxhold, hold); held_steps += hold
    trseg = [comps(s)[0] for s in segs]
    trseg = [x for x in trseg if x]
    rev = sum(1 for a, b in zip(trseg[:-1], trseg[1:]) if any(OPP[d] in a for d in b))
    longest = run = 0
    for (tr, rot, grip, stop) in C:
        run = run + 1 if (stop and not tr and not rot and grip is None) else 0
        longest = max(longest, run)
    n_stop = sum(1 for (tr, rot, grip, stop) in C if stop and not tr and not rot and grip is None)
    cnt = Counter(prims); ent = -sum(v / H * math.log(v / H) for v in cnt.values()) if H else 0.0
    return [H, n_stop / max(1, H), T4, T5, 1 - T5 / max(1, T4), T8, maxhold, holds, held_steps / max(1, H), rev,
            longest, first_grasp, sum(1 for c in C if c[0]), sum(1 for c in C if c[1]), sum(1 for c in C if c[2]), ent]


def main(limit_files=None):
    path = os.path.join(OUT, "vla_features.csv")
    if os.path.exists(path) and limit_files is None:
        print("exists", path); return
    rows = []; nf = 0; n_all = n_reason = n_instr = 0
    for fp, eps in stream_top_level(SRC):
        nf += 1
        for eid, ep in eps.items():
            n_all += 1
            if not ep.get("reasoning"):
                continue
            n_reason += 1
            md = ep.get("metadata") or {}
            instr = (md.get("language_instruction") or "").strip()
            if not instr:
                continue
            n_instr += 1
            prims = (ep.get("features") or {}).get("move_primitive") or []
            if len(prims) < 2:
                continue
            r = dict(uid="%s#%s" % (fp, eid), instr=instr.lower(), skill=skill(instr), n_steps_meta=md.get("n_steps"))
            r.update(dict(zip(COLS, phi_vla(prims))))
            rows.append(r)
        if limit_files and nf >= limit_files:
            break
        if nf % 500 == 0:
            print("files", nf, "rows", len(rows), flush=True)
    df = pd.DataFrame(rows)
    print("files", nf, "episodes", n_all, "with reasoning", n_reason, "with instruction", n_instr, "rows", len(df), flush=True)
    if limit_files:
        return df
    df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    print("wrote", path, df.shape)


if __name__ == "__main__":
    main()
