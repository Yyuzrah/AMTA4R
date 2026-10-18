"""Family E of App. M.2: BabyAI oracle (bot) reference trajectories on freshly generated levels.

Levels: BabyAI-MixedTrainLocal-v0 of the Minigrid fork used by BALROG (github.com/BartekCupial/Minigrid, version
2.3.1, Apache-2.0 license file), which also provides the BabyAI bot (minigrid.utils.baby_ai_bot, taken from the
original BabyAI code, BSD-3-Clause); see requirements-babyai.txt. One env object per instruction kind (goto, pickup,
open, putnext, pick up seq go to), default configuration (num_dists=8); level s = env.reset(seed=s) for s = 0, 1, 2, ...
The BabyAI bot's explicit subgoal stack is logged (push/pop) by LogStack; run_bot runs the bot, phi_ref computes the
per-level coordinates. Every generated level is written; w1.py keeps the levels the bot solves (the reference reaches
the goal; chi10 is then constant and is not a coordinate). Generation stops at TARGET solved levels per kind or after
MAX_SEEDS seeds.  Output: features/babyai_<kind>.csv (spaces in the kind replaced by '_'; one row per level).
Idempotent: a kind whose output exists is skipped.
Usage: python gen_babyai.py <kind_index or all> [TARGET (4000)] [MAX_SEEDS (20000)]
"""
import os, sys, time, json
import numpy as np, pandas as pd, gymnasium as gym, minigrid
from minigrid.utils.baby_ai_bot import BabyAIBot
minigrid.register_minigrid_envs()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")
KINDS = ["goto", "pickup", "open", "putnext", "pick up seq go to"]


class LogStack(list):
    def __init__(self, log): super().__init__(); self.log = log
    def append(self, x):
        super().append(x); self.log.append(('push', type(x).__name__, getattr(x, 'reason', None), len(self)))
    def pop(self, *a):
        x = super().pop(*a); self.log.append(('pop', type(x).__name__, getattr(x, 'reason', None), len(self))); return x


def run_bot(env, max_steps=None):
    u = env.unwrapped
    log = []
    bot = BabyAIBot(u)
    init = list(bot.stack); bot.stack = LogStack(log)
    for s in init: bot.stack.append(s)
    max_steps = max_steps or u.max_steps
    depth_t = []; acts = []; R = 0
    for t in range(max_steps):
        a = bot.replan()
        depth_t.append(len(bot.stack))
        if a == u.actions.done: break
        acts.append(int(a))
        o, r, te, tr, _ = env.step(a); R += r
        if te or tr: break
    return log, depth_t, acts, R


def phi_ref(log, depth_t, acts, R):
    D = np.array([e[3] for e in log], float) if log else np.zeros(1)
    n = len(log)
    dt = np.array(depth_t, float) if depth_t else np.zeros(1)
    d = np.diff(dt); M = max(1, len(d))
    pushes = [e for e in log if e[0] == 'push']
    expl = sum(1 for e in pushes if e[1] == 'ExploreSubgoal' or e[2] == 'Explore')
    manip = sum(1 for e in pushes if e[1] in ('PickupSubgoal', 'DropSubgoal', 'OpenSubgoal', 'CloseSubgoal'))
    nav = sum(1 for e in pushes if e[1] == 'GoNextToSubgoal')
    return dict(chi1_maxdepth=D.max(), chi2_meandepth=D.mean(), chi3_sddepth=D.std(), chi4_volume=n,
                chi5_deepen=(d > 0).sum() / M, chi6_backtrack=(d < 0).sum() / M, chi7_lateral=(d == 0).sum() / M,
                chi8_explore=expl, chi9_explore_rate=expl / max(1, len(pushes)), chi10_oracle_success=float(R > 0),
                chi11_manip=manip, chi12_nav=nav, ref_len=len(acts), ref_turns=sum(a in (0, 1) for a in acts),
                ref_interact=sum(a in (3, 4, 5) for a in acts))


def env_for(kind):
    for _ in range(10000):
        env = gym.make("BabyAI-MixedTrainLocal-v0")
        if env.unwrapped.action_kinds[0] == kind:
            return env
    raise RuntimeError(kind)


def gen(kind, target, max_seeds):
    path = os.path.join(OUT, "babyai_%s.csv" % kind.replace(" ", "_"))
    if os.path.exists(path):
        print("exists", path); return
    env = env_for(kind); rows = []; t0 = time.time(); n_ok = 0
    for s in range(max_seeds):
        obs, _ = env.reset(seed=s)
        u = env.unwrapped
        n_objs = sum(1 for i in range(u.width) for j in range(u.height)
                     if (o := u.grid.get(i, j)) is not None and o.type != 'wall')
        try:
            log, dt, acts, R = run_bot(env)
            row = dict(kind=kind, seed=s, mission=obs['mission'], n_objs=n_objs, bot_error="", **phi_ref(log, dt, acts, R))
        except Exception as e:
            row = dict(kind=kind, seed=s, mission=obs['mission'], n_objs=n_objs, bot_error=repr(e)[:100])
        rows.append(row)
        n_ok += int(row.get("chi10_oracle_success", 0) == 1.0)
        if n_ok >= target: break
    df = pd.DataFrame(rows)
    df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    print(kind, "seeds", len(df), "solved", n_ok, "errors", int((df.bot_error != "").sum()), "time %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    max_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    ks = KINDS if which == "all" else [KINDS[int(which)]]
    for k in ks:
        gen(k, target, max_seeds)
