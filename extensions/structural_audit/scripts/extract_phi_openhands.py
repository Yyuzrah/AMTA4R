"""The features of extract_phi_swe.py on a second harness: nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0).

67,074 OpenHands v0.54.0 trajectories of Qwen3-Coder-480B-A35B-Instruct on SWE-rebench issues, with
`resolved` labels. Tools: execute_bash, str_replace_editor (view/create/str_replace/insert/undo_edit),
think, task_tracker (an explicit, externally held todo list), finish.
The same feature function as for SWE-agent is applied after mapping OpenHands tool calls to the
common step schema (act, cmd, cat, obs_len, flags, open_file, ai_len). Some coordinates change meaning
across harnesses (e.g. a failed edit is a lint rejection in SWE-agent but an editor error such as an old_str
mismatch here).
Output (data/harness/): phi_oh_full.csv (read by feat_harness.py), phi_oh_prefix.csv, issues_oh.csv.
Usage: python extract_phi_openhands.py [n_procs]  (after extract_phi_swe.py, which creates data/harness/)
"""
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_phi_swe import ERR_PATTERNS, PREFIX_KS, features, write  # noqa: E402

ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "openhands", "trajectories.parquet")
OUT = os.path.join(ROOT, "data", "harness")
EXIT = re.compile(r"exit code (\d+)")
SEARCH_B = {"grep", "find", "ls", "rg", "tree", "git", "pwd", "which"}
VIEW_B = {"cat", "head", "tail", "less", "nl", "awk"}
RUN_B = ("python", "pytest", "tox", "bash", "./", "sh", "make", "pip", "nosetests", "timeout", "PYTHONPATH")
EDIT_CMDS = {"create", "str_replace", "insert", "undo_edit"}


def bash_first(cmd):
    c = cmd.strip()
    c = re.sub(r"^(cd\s+\S+\s*(&&|;)\s*)+", "", c)
    toks = c.split()
    return toks[0] if toks else "", c


def step_of(fn, args, obs, content):
    ai_len = len(content or "") + len(json.dumps(args))
    open_file = None
    edit_fail = False
    if fn == "str_replace_editor":
        sub = args.get("command", "")
        path = args.get("path", "")
        is_dir = obs.startswith("Here's the files and directories")
        if sub == "view" and not is_dir:
            cmd, cat, act, open_file = "open", "view", "open " + path + " " + str(args.get("view_range")), path
        elif sub == "view":
            cmd, cat, act = "ls", "search", "ls " + path
        elif sub in EDIT_CMDS:
            cmd, cat, open_file = "edit", "edit", path
            # identifies identical edits within one run (repeat_action_rate); the value itself is not written
            act = "edit " + path + " " + str(hash(str(args.get("old_str")) + str(args.get("new_str")) + str(args.get("file_text"))))
            edit_fail = obs.startswith("ERROR")
        else:
            cmd, cat, act = "other", "other", "sre " + sub
    elif fn == "execute_bash":
        first, c = bash_first(args.get("command", ""))
        act = "bash " + c
        cmd = first
        if first in SEARCH_B:
            cat = "search"
        elif first in VIEW_B or (first == "sed" and "-n" in c.split()[:3]):
            cat = "view"
        elif first in ("sed", "rm", "mv", "cp", "mkdir", "touch") or first.startswith("echo") and ">" in c:
            cat = "edit"
        elif first.startswith(RUN_B):
            cat = "run"
        else:
            cat = "other"
    elif fn == "finish":
        cmd, cat, act = "submit", "submit", "submit"
    elif fn == "task_tracker":
        cmd, cat, act = "task_tracker", "other", "task_tracker " + str(args.get("command"))
    else:
        cmd, cat, act = fn, "other", fn
    m = EXIT.findall(obs[-300:]) if fn == "execute_bash" else []
    bad_exit = bool(m) and m[-1] != "0"
    traceback = "Traceback (most recent call last)" in obs
    err = edit_fail or traceback or bad_exit or obs.startswith("ERROR") or any(p.search(obs[:400]) for p in ERR_PATTERNS)
    return dict(act=act, cmd=cmd, cat=cat, obs_len=len(obs), flags=(edit_fail, traceback, err),
                open_file=open_file, ai_len=ai_len, fn=fn)


def parse(traj):
    obs = {m["tool_call_id"]: (m["content"] or "") for m in traj if m["role"] == "tool"}
    issue = ""
    steps = []
    for m in traj:
        if m["role"] == "user" and not issue:
            issue = m["content"] or ""
        if m["role"] != "assistant":
            continue
        for tc in (m["tool_calls"] or []):
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            steps.append(step_of(tc["function"]["name"], args, obs.get(tc["id"], ""), m["content"]))
    return issue, steps


def extra(steps):
    n = max(1, len(steps))
    return {"frac_task_tracker": sum(s["fn"] == "task_tracker" for s in steps) / n,
            "used_task_tracker": int(any(s["fn"] == "task_tracker" for s in steps)),
            "frac_think": sum(s["fn"] == "think" for s in steps) / n}


def process(rg):
    pf = pq.ParquetFile(SRC)
    t = pf.read_row_group(rg, columns=["trajectory_id", "instance_id", "repo", "trajectory", "exit_status", "resolved"])
    d = t.to_pydict()
    full, pref, issues = [], [], {}
    for j in range(t.num_rows):
        issue, steps = parse(d["trajectory"][j])
        m = re.search(r"<issue_description>(.*?)</issue_description>", issue, re.S)
        issue = m.group(1) if m else issue
        issues.setdefault(d["instance_id"][j], issue)
        es = d["exit_status"][j] or ""
        base = dict(instance_id=d["instance_id"][j], model_name="qwen3-coder-480b-openhands", target=int(d["resolved"][j] or 0),
                    exit_status=("submit" if es == "submit" else es.split(":")[0][:40]), shard="rg%02d" % rg, row=j)
        full.append({**base, **features(steps), **extra(steps), "issue_chars": len(issue)})
        for k in PREFIX_KS:
            if len(steps) > k:
                pref.append({**base, "k": k, **features(steps[:k]), **extra(steps[:k]), "issue_chars": len(issue)})
    return full, pref, issues


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_rg = pq.ParquetFile(SRC).metadata.num_row_groups
    with Pool(nproc) as p:
        res = p.map(process, range(n_rg))
    full = [r for a, _, _ in res for r in a]
    pref = [r for _, b, _ in res for r in b]
    issues = {}
    for _, _, c in res:
        issues.update(c)
    with open(os.path.join(OUT, "issues_oh.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "issue"])
        for k, v in issues.items():
            w.writerow([k, v])
    write(os.path.join(OUT, "phi_oh_full.csv"), full)
    write(os.path.join(OUT, "phi_oh_prefix.csv"), pref)
    print("trajectories", len(full), "prefix rows", len(pref))
