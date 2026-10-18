"""Per-step parsing and per-trajectory features of SWE-agent runs (input of feat_harness.py, family C of App. M.2).

Data: HF nebius/SWE-agent-trajectories (CC-BY-4.0), 80,036 SWE-agent trajectories
from Llama-3.1-8B/70B/405B-based agents, each with a resolved label `target`.
Each trajectory is the agent's own trace (thought + one command per turn, then the
environment observation).

Output (data/harness/): phi_swe_full.csv (one row per trajectory, full-trace features; read by feat_harness.py)
        phi_swe_prefix.csv (one row per trajectory and prefix length k, features
                            on the first k agent turns only, for trajectories with > k turns)
        issues.csv (instance_id, issue text).
Usage: python extract_phi_swe.py [n_shards, 0 = all] [n_procs]
"""
import ast
import csv
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "swe_agent", "data")
OUT = os.path.join(ROOT, "data", "harness")
PREFIX_KS = (5, 10, 15, 20, 30)

FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)
TRAILER = re.compile(r"\(Open file: (.*?)\)\s*\n\(Current directory: (.*?)\)")

VIEW = {"open", "goto", "scroll_down", "scroll_up", "cat", "head", "tail", "less", "more"}
SEARCH = {"search_dir", "search_file", "find_file", "find", "grep", "ls", "tree", "find."}
EDIT = {"edit", "create", "rm", "sed", "mkdir", "touch", "mv", "cp"}
RUN_PREFIX = ("python", "pytest", "tox", "bash", "./", "sh", "make", "pip", "nosetests")


def un(s):
    if s is None:
        return ""
    try:
        v = ast.literal_eval(s)
    except Exception:
        return s
    return v if isinstance(v, str) else ("" if v is None else str(v))


def action_of(ai_text):
    m = FENCE.findall(ai_text)
    if not m:
        return "", ""
    act = m[-1].strip()
    first = act.split()[0] if act.split() else ""
    return act, first


def category(cmd):
    if cmd == "submit":
        return "submit"
    if cmd in VIEW:
        return "view"
    if cmd in SEARCH:
        return "search"
    if cmd in EDIT:
        return "edit"
    if cmd.startswith(RUN_PREFIX):
        return "run"
    return "other"


ERR_PATTERNS = [
    re.compile(p, re.M)
    for p in (
        r"command not found",
        r"^Usage: ",
        r"not found",
        r"No such file or directory",
        r"EXECUTION TIMED OUT",
        r"^ERROR",
        r"Error:",
    )
]


def obs_flags(obs):
    head = obs[:400]
    edit_fail = obs.startswith("Your proposed edit has introduced")
    traceback = "Traceback (most recent call last)" in obs
    err = edit_fail or traceback or any(p.search(head) for p in ERR_PATTERNS)
    return edit_fail, traceback, err


def parse(traj):
    """Return the issue text and a list of steps: dict(act, cmd, cat, obs, open_file, ai_len, flags)."""
    issue = ""
    steps = []
    msgs = [(un(s["role"]), un(s["text"])) for s in traj]
    for i, (role, text) in enumerate(msgs):
        if role == "user" and not issue and "ISSUE:" in text[:400]:
            issue = text
        if role != "ai":
            continue
        act, cmd = action_of(text)
        obs = ""
        if i + 1 < len(msgs) and msgs[i + 1][0] == "user":
            obs = msgs[i + 1][1]
        m = TRAILER.findall(obs[-400:]) if obs else []
        open_file = m[-1][0].strip() if m else None
        steps.append(dict(act=act, cmd=cmd, cat=category(cmd), obs_len=len(obs), flags=obs_flags(obs),
                          open_file=open_file, ai_len=len(text)))
    return issue, steps


def features(steps):
    """Phi_SWE on a list of steps. Coordinates named after the paper's chi where an analogue exists."""
    N = len(steps)
    f = {}
    # --- focus-file stack: a Dyck-style projection of navigation ------------------------------
    # node = file open in the editor after the step (from the observation trailer). Opening a file
    # that is not on the stack pushes it (down); returning to a file already on the stack pops back
    # to it (up = backtrack/revisit); staying on the same file (or no file) is lateral.
    stack, depths = [], []
    cur = None
    downs = ups = lats = 0
    for s in steps:
        nf = s["open_file"]
        if nf is None or nf == "n/a" or nf == cur:
            lats += 1 if depths else 0
        elif nf in stack:
            stack = stack[: stack.index(nf) + 1]
            ups += 1
            cur = nf
        else:
            stack.append(nf)
            downs += 1
            cur = nf
        depths.append(len(stack))
    D = np.array(depths, dtype=float) if depths else np.zeros(1)
    M = max(1, N - 1)
    f["chi1_max_depth"] = D.max()
    f["chi2_mean_depth"] = D.mean()
    f["chi3_std_depth"] = D.std()
    f["chi4_volume"] = N
    f["chi5_down_rate"] = downs / M
    f["chi6_up_rate"] = ups / M
    f["chi7_lateral_rate"] = lats / M
    # --- dead ends and failures ----------------------------------------------------------------
    edit_fail = trace = err = 0
    consec = maxconsec = 0
    recovered = 0
    fail_idx = []
    for i, s in enumerate(steps):
        ef, tb, e = s["flags"]
        edit_fail += ef
        trace += tb
        err += e
        if ef:
            fail_idx.append(i)
            consec += 1
            maxconsec = max(maxconsec, consec)
        elif s["cmd"] == "edit":
            consec = 0
    for i in fail_idx:
        nxt = steps[i + 1 : i + 4]
        if any(t["cmd"] == "edit" and not t["flags"][0] for t in nxt):
            recovered += 1
    n_edit = sum(s["cmd"] == "edit" for s in steps)
    f["chi8_failed_edits"] = edit_fail
    f["chi9_failed_edit_rate"] = edit_fail / n_edit if n_edit else 0.0
    f["err_obs"] = err
    f["err_obs_rate"] = err / N if N else 0.0
    f["tracebacks"] = trace
    f["max_failed_edit_cascade"] = maxconsec
    f["recovery_rate"] = recovered / edit_fail if edit_fail else 1.0
    # --- action mix (analogue of the operator-mix coordinates chi11/chi12) ----------------------
    cats = [s["cat"] for s in steps]
    for c in ("view", "search", "edit", "run", "other"):
        f["frac_" + c] = cats.count(c) / N if N else 0.0
    # --- revisits and loops ---------------------------------------------------------------------
    opened = [s["act"].split()[1] for s in steps if s["cmd"] == "open" and len(s["act"].split()) > 1]
    seen, rev = set(), 0
    for o in opened:
        rev += o in seen
        seen.add(o)
    f["open_revisit_rate"] = rev / len(opened) if opened else 0.0
    f["distinct_files_opened"] = len(seen)
    edited = {s["open_file"] for s in steps if s["cmd"] in ("edit", "create") and s["open_file"]}
    f["distinct_files_edited"] = len(edited)
    acts = [s["act"] for s in steps]
    seen_a, rep = set(), 0
    run = maxrun = 1 if acts else 0
    for i, a in enumerate(acts):
        rep += a in seen_a
        seen_a.add(a)
        if i and a == acts[i - 1]:
            run += 1
            maxrun = max(maxrun, run)
        elif i:
            run = 1
    f["repeat_action_rate"] = rep / N if N else 0.0
    f["max_identical_run"] = maxrun
    f["n_test_runs"] = cats.count("run")
    first_edit = next((i for i, c in enumerate(steps) if c["cmd"] == "edit"), N)
    f["first_edit_pos"] = first_edit / N if N else 1.0
    # --- position / length (not coordinates; feat_harness.py uses len_turns to turn shares into counts) ---
    ai_chars = sum(s["ai_len"] for s in steps)
    obs_chars = sum(s["obs_len"] for s in steps)
    f["len_turns"] = N
    f["len_chars"] = ai_chars + obs_chars
    f["obs_char_share"] = obs_chars / max(1, ai_chars + obs_chars)
    off = 0
    first_fail_off = -1
    for s in steps:
        off += s["ai_len"]
        if s["flags"][0]:
            first_fail_off = off
            break
        off += s["obs_len"]
    f["first_fail_char_offset"] = first_fail_off
    f["submitted_in_trace"] = int(any(s["cmd"] == "submit" for s in steps))
    return f


def process(path):
    t = pq.read_table(path, columns=["instance_id", "model_name", "target", "exit_status", "trajectory"])
    rows_full, rows_pref, issues = [], [], {}
    d = t.to_pydict()
    for j in range(t.num_rows):
        issue, steps = parse(d["trajectory"][j])
        base = dict(
            instance_id=d["instance_id"][j],
            model_name=d["model_name"][j],
            target=int(bool(d["target"][j])),
            exit_status=d["exit_status"][j],
            shard=os.path.basename(path),
            row=j,
        )
        if base["instance_id"] not in issues:
            issues[base["instance_id"]] = issue
        f = features(steps)
        rows_full.append({**base, **f, "issue_chars": len(issue)})
        for k in PREFIX_KS:
            if len(steps) > k:
                rows_pref.append({**base, "k": k, **features(steps[:k]), "issue_chars": len(issue)})
    return rows_full, rows_pref, issues


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, "train-*.parquet")))
    if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
        files = files[: int(sys.argv[1])]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(min(nproc, len(files))) as p:
        res = p.map(process, files)
    full = [r for a, _, _ in res for r in a]
    pref = [r for _, b, _ in res for r in b]
    issues = {}
    for _, _, c in res:
        issues.update(c)
    write(os.path.join(OUT, "phi_swe_full.csv"), full)
    write(os.path.join(OUT, "phi_swe_prefix.csv"), pref)
    with open(os.path.join(OUT, "issues.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "issue"])
        for k, v in issues.items():
            w.writerow([k, v])
    print("trajectories", len(full), "prefix rows", len(pref), "instances", len(issues))
