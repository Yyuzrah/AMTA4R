"""Task-side structural coordinates Phi_D of mini-SWE-agent and OpenHands trajectories: the 16 coordinates of family C
of the structural audit (App. M.2), computed from a common step schema. The feature function
(extensions/structural_audit/scripts/extract_phi_swe.py::features) and the coordinate mapping
(extensions/structural_audit/scripts/feat_harness.py::conv, repeated below as conv) are used unchanged; the bash
classifier is the execute_bash branch of extensions/structural_audit/scripts/extract_phi_openhands.py, applied to both
harnesses. Steps are the tool calls issued by the first 100 LLM calls of an episode.
focus_ext=1: a bash command that reads or writes exactly one file also sets the file focus (same rule in both
harnesses). focus_ext=0: the file rule of family C, where only editor views and edits set the file focus.
Output: one CSV row per (harness, instance) with the 16 coordinates, the exit status and call counts.
Usage: python phi_features.py <runs_tag> <out_csv> <focus_ext 1|0>"""
import json, os, posixpath, re, shlex, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "structural_audit", "scripts"))
from extract_phi_swe import ERR_PATTERNS, features  # noqa: E402  (feature function of the structural audit, unchanged)
from extract_phi_openhands import SEARCH_B, VIEW_B, RUN_B, bash_first  # noqa: E402

HARNESS_DIR = os.environ["HARNESS_DIR"]
COLS = ["T1_max_depth", "T2_mean_depth", "T3_sd_depth", "T4_tool_calls", "T5_distinct_files", "T6_file_reopen",
        "T6a_repeat_action", "C1_push_rate", "C2_pop_rate", "C7_failed_edits", "C9_failed_edit_cascade",
        "C14_run_calls", "C15_first_edit_pos", "I1_read_calls", "I1_write_calls", "I3_error_rate"]
READ1 = {"cat", "head", "tail", "nl", "less", "more"}
FILTERS = {"head", "tail", "nl", "cat", "less", "more", "sed"}
EMIT = {"cat", "echo", "printf"}
CTRL = {"|", "||", "&&", ";", "&", "|&"}
BUDGET = 100


def conv(f):  # conv() of structural_audit/scripts/feat_harness.py, applied to one row
    N = float(f["len_turns"])
    return dict(T1_max_depth=f["chi1_max_depth"], T2_mean_depth=f["chi2_mean_depth"], T3_sd_depth=f["chi3_std_depth"],
                T4_tool_calls=f["chi4_volume"], T5_distinct_files=f["distinct_files_opened"],
                T6_file_reopen=f["open_revisit_rate"], T6a_repeat_action=f["repeat_action_rate"],
                C1_push_rate=f["chi5_down_rate"], C2_pop_rate=f["chi6_up_rate"], C7_failed_edits=f["chi8_failed_edits"],
                C9_failed_edit_cascade=f["max_failed_edit_cascade"], C14_run_calls=f["n_test_runs"],
                C15_first_edit_pos=f["first_edit_pos"], I1_read_calls=round((f["frac_view"] + f["frac_search"]) * N),
                I1_write_calls=round(f["frac_edit"] * N), I3_error_rate=f["err_obs_rate"])


def norm(p, cwd, repo_root):
    if not p:
        return None
    p = p if p.startswith("/") else posixpath.join(cwd or repo_root, p)
    p = posixpath.normpath(p)
    for pre in (repo_root.rstrip("/") + "/", "/testbed/"):
        if p.startswith(pre):
            return p[len(pre):]
    return p


def tokens(line):
    lx = shlex.shlex(line, posix=True, punctuation_chars=True); lx.whitespace_split = True
    return list(lx)


def segments(toks):
    segs, cur = [], []
    for t in toks:
        if t in CTRL:
            segs.append((cur, t)); cur = []
        else:
            cur.append(t)
    segs.append((cur, None))
    return segs


def file_operands(seg, tool):
    ops, skip = [], False
    for k, t in enumerate(seg[1:]):
        if skip:
            skip = False; continue
        if t in (">", ">>", "<", "<<", ">&", "<<<"):
            skip = True; continue
        if t.startswith("-"):
            if tool in ("head", "tail") and t in ("-n", "-c"):
                skip = True
            continue
        ops.append(t)
    return ops


def single_file_focus(command):
    """('view'|'edit', path) when the command reads or writes exactly one file, else None."""
    c = command.strip()
    c = re.sub(r"^(cd\s+\S+\s*(&&|;)\s*)+", "", c)
    heredoc = "<<" in c.split("\n", 1)[0]
    if "\n" in c and not heredoc:
        return None
    line = c.split("\n", 1)[0]
    try:
        toks = tokens(line)
    except ValueError:
        return None
    if not toks:
        return None
    segs = segments(toks)
    first, _ = segs[0]
    if not first:
        return None
    tool = first[0]
    # writes: sed -i on one file; cat/echo/printf (or a here-document) redirected to one file; tee to one file
    targets = []
    for seg, _ in segs:
        for k, t in enumerate(seg):
            if t in (">", ">>") and k + 1 < len(seg) and seg[0] in EMIT:
                targets.append(seg[k + 1])
        if seg and seg[0] == "tee":
            targets += [t for t in seg[1:] if not t.startswith("-")]
    targets = [t for t in targets if t not in ("/dev/null",) and not t.isdigit() and not t.startswith("&")]
    if tool == "sed" and any(t.startswith("-i") or t == "--in-place" for t in first[1:]):
        ops = [t for t in file_operands(first, "sed")]
        has_e = any(t in ("-e", "--expression") for t in first[1:])
        files = ops if has_e else ops[1:]
        if len(set(files)) == 1 and len(segs) == 1:
            return ("edit", files[0])
        return None
    if targets:
        if len(set(targets)) == 1:
            return ("edit", targets[0])
        return None
    # reads: one file through cat/head/tail/nl/less/more or sed -n, optionally piped into display filters only
    if any(t in (">", ">>") for seg, _ in segs for t in seg):
        return None
    if tool in READ1 or (tool == "sed" and "-n" in first[1:3]):
        ops = file_operands(first, tool)
        if tool == "sed":
            ops = ops[1:]
        if len(ops) == 1 and not re.search(r"[*?\[]", ops[0]) and not ops[0].endswith("/"):
            if all(seg and seg[0] in FILTERS and (seg[0] != "sed" or "-n" in seg) for seg, _ in segs[1:]):
                return ("view", ops[0])
    return None


def bash_step(command, out, exit_code, cwd, repo_root, ai_len, focus_ext):
    """execute_bash classification of extract_phi_openhands.py, plus the single-file focus extension."""
    first, c = bash_first(command)
    act, cmd, open_file, edit_fail = "bash " + c, first, None, False
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
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in c:
        cmd, cat, act = "submit", "submit", "submit"
    elif focus_ext:
        m = re.match(r"^cd\s+(\S+)\s*(&&|;)", command.strip())
        wd = norm_dir(m.group(1), cwd, repo_root) if m else cwd
        f = single_file_focus(command)
        if f:
            rel = norm(f[1], wd, repo_root)
            if f[0] == "view":
                cmd, cat, act, open_file = "open", "view", f"open {rel} {c}", rel
            else:
                cmd, cat, act, open_file = "edit", "edit", f"edit {rel} {c}", rel
                edit_fail = exit_code not in (0, None)
    bad_exit = exit_code not in (0, None)
    tb = "Traceback (most recent call last)" in out
    err = edit_fail or tb or bad_exit or out.startswith("ERROR") or any(p.search(out[:400]) for p in ERR_PATTERNS)
    return dict(act=act, cmd=cmd, cat=cat, obs_len=len(out), flags=(edit_fail, tb, err), open_file=open_file, ai_len=ai_len)


def norm_dir(d, cwd, repo_root):
    d = d.strip("'\"")
    p = d if d.startswith("/") else posixpath.join(cwd or repo_root, d)
    return posixpath.normpath(p)


# ---------------------------------------------------------------- mini-SWE-agent 2.4.6 trajectories
def steps_mini(traj, focus_ext):
    msgs = traj["messages"]; repo_root = "/testbed"
    obs = {m.get("tool_call_id"): m for m in msgs if m.get("role") == "tool"}
    steps, n_calls, n_tc, n_known = [], 0, 0, 0
    for m in msgs:
        if m.get("role") == "user" and (m.get("extra") or {}).get("interrupt_type") == "FormatError":
            n_calls += 1; continue
        if m.get("role") != "assistant":
            continue
        n_calls += 1
        if n_calls > BUDGET:
            break
        ai_len = len(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            n_tc += 1
            fn = (tc.get("function") or {}).get("name")
            try:
                args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
            except Exception:
                args = {}
            if fn != "bash" or not isinstance(args, dict) or "command" not in args:
                steps.append(dict(act="other " + str(fn), cmd=str(fn), cat="other", obs_len=0, flags=(False, False, False),
                                  open_file=None, ai_len=ai_len)); continue
            n_known += 1
            o = obs.get(tc.get("id"))
            ex = (o or {}).get("extra") or {}
            out = ex.get("raw_output") if ex.get("raw_output") is not None else ((o or {}).get("content") or "")
            code = ex.get("returncode") if o else 0
            steps.append(bash_step(args["command"], out or "", code, repo_root, repo_root, ai_len, focus_ext))
    return steps, dict(n_llm_calls=min(n_calls, BUDGET), n_tool_calls=n_tc, n_known=n_known)


# ---------------------------------------------------------------- OpenHands software-agent-sdk 1.16.1 events
def _text(o):
    c = o.get("content")
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else (o.get("output") or "")


def steps_oh(events, repo, focus_ext):
    repo_root = f"/workspace/{repo.split('/')[-1]}"
    obs = {}
    for e in events:
        if e.get("kind") in ("ObservationEvent", "AgentErrorEvent", "UserRejectObservation") and e.get("tool_call_id"):
            obs[e["tool_call_id"]] = e
    order = []
    for e in events:
        if e.get("source") == "agent" and e.get("kind") in ("ActionEvent", "MessageEvent"):
            rid = e.get("llm_response_id") or e.get("id")
            if rid not in order:
                order.append(rid)
    allowed = set(order[:BUDGET])
    steps, n_tc, n_known, cwd = [], 0, 0, repo_root
    for e in events:
        if e.get("kind") != "ActionEvent" or e.get("source") != "agent":
            continue
        if (e.get("llm_response_id") or e.get("id")) not in allowed:
            continue
        n_tc += 1
        tn = e.get("tool_name"); a = e.get("action") or {}
        if not a:
            try:
                a = json.loads(((e.get("tool_call") or {}).get("arguments")) or "{}")
            except Exception:
                a = {}
        thought = e.get("thought") or []
        ai_len = sum(len(x.get("text", "")) for x in thought if isinstance(x, dict)) + len(json.dumps(a))
        oe = obs.get(e.get("tool_call_id")) or {}
        o = oe.get("observation") or {}
        out = _text(o) if o else (oe.get("error") or "")
        if tn == "terminal":
            n_known += 1
            code = o.get("exit_code")
            if code is None:
                code = (o.get("metadata") or {}).get("exit_code")
            if code == -1 and not o.get("timeout"):
                code = None  # still running / interactive: no exit code, as in extract_phi_openhands.py
            steps.append(bash_step(a.get("command", ""), out, code, cwd, repo_root, ai_len, focus_ext))
            wd = (o.get("metadata") or {}).get("working_dir")
            if wd:
                cwd = wd
        elif tn == "file_editor":
            n_known += 1
            sub, path = a.get("command", ""), a.get("path", "")
            rel = norm(path, cwd, repo_root)
            is_err = bool(o.get("is_error")) or out.startswith("ERROR") or oe.get("kind") == "AgentErrorEvent"
            if sub == "view" and not out.startswith("Here's the files and directories"):
                st = dict(cmd="open", cat="view", act=f"open {rel} {a.get('view_range')}", open_file=rel, ef=False)
            elif sub == "view":
                st = dict(cmd="ls", cat="search", act=f"ls {rel}", open_file=None, ef=False)
            elif sub in ("create", "str_replace", "insert", "undo_edit"):
                h = hash(str(a.get("old_str")) + str(a.get("new_str")) + str(a.get("file_text")))
                st = dict(cmd="edit", cat="edit", act=f"edit {rel} {h}", open_file=rel, ef=is_err)
            else:
                st = dict(cmd="other", cat="other", act="sre " + str(sub), open_file=None, ef=False)
            tb = "Traceback (most recent call last)" in out
            err = st["ef"] or tb or is_err or any(p.search(out[:400]) for p in ERR_PATTERNS)
            steps.append(dict(act=st["act"], cmd=st["cmd"], cat=st["cat"], obs_len=len(out), flags=(st["ef"], tb, err),
                              open_file=st["open_file"], ai_len=ai_len))
        elif tn == "finish":
            n_known += 1
            steps.append(dict(act="submit", cmd="submit", cat="submit", obs_len=len(out), flags=(False, False, False), open_file=None, ai_len=ai_len))
        elif tn in ("task_tracker", "think"):
            n_known += 1
            steps.append(dict(act=f"{tn} {a.get('command', '')}", cmd=tn, cat="other", obs_len=len(out), flags=(False, False, False), open_file=None, ai_len=ai_len))
        else:
            steps.append(dict(act=str(tn), cmd=str(tn), cat="other", obs_len=len(out), flags=(False, False, False), open_file=None, ai_len=ai_len))
    return steps, dict(n_llm_calls=len(order[:BUDGET]), n_llm_calls_total=len(order), n_tool_calls=n_tc, n_known=n_known)


def main(tag, out_csv, focus_ext):
    repo = {k: v["repo"] for k, v in json.load(open(f"{HARNESS_DIR}/data/swebench_verified.json")).items()}
    rows = []
    for h in ("mini", "oh"):
        base = os.path.join(HARNESS_DIR, "runs", tag, h)
        for iid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            d = os.path.join(base, iid)
            if not os.path.exists(os.path.join(d, "meta.json")):
                continue
            meta = json.load(open(os.path.join(d, "meta.json")))
            if h == "mini":
                steps, info = steps_mini(json.load(open(os.path.join(d, "traj.json"))), focus_ext)
            else:
                steps, info = steps_oh(json.load(open(os.path.join(d, "events.json"))), repo[iid], focus_ext)
            f = features(steps) if steps else features([])
            row = dict(harness=h, instance_id=iid, exit_status=meta.get("exit_status"), **conv(f), **info,
                       len_chars=f["len_chars"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv + ".tmp", index=False); os.replace(out_csv + ".tmp", out_csv)
    print(df.groupby("harness")[COLS + ["n_llm_calls", "n_tool_calls"]].mean().T.round(3).to_string())
    cov = df.groupby("harness")[["n_tool_calls", "n_known"]].sum()
    print("parse coverage", (cov.n_known / cov.n_tool_calls.clip(lower=1)).round(4).to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] == "1")
