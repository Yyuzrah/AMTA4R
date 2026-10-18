"""Episode scheduler (stdlib only). Runs one harness over an instance list with W concurrent episodes; each episode
is a separate process (own session) with a hard wall-clock kill (backstop for the in-episode 3000 s limit).
Restart-safe: an instance is skipped if runs/<tag>/<harness>/<iid>/meta.json exists; an instance is given up after
3 failed attempts (infrastructure errors or hard kills). Usage:
python scheduler.py <harness: mini|oh> <list_file> <workers> <base_url> <runs_dir>"""
import glob, json, os, signal, subprocess, sys, time

H, LIST, W, URL, RUNS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
HARNESS_DIR = os.environ["HARNESS_DIR"]
PY = {"mini": f"{HARNESS_DIR}/envs/mini2/bin/python", "oh": f"{HARNESS_DIR}/envs/oh2/bin/python"}[H]  # environments installed with real file copies (UV_LINK_MODE=copy)
SCRIPT = f"{HARNESS_DIR}/code/harness/run_one_{H}.py"
HARD_S, MAX_ATT = 3600, 3
ids = [l.strip() for l in open(LIST) if l.strip()]
running = {}  # iid -> (Popen, t0)


def attempts(d):
    """Failed attempts: infra errors, hard kills, and crashes after the sandbox had started. Crashes before any
    sandbox (import-time failures under file-system load) are retried without counting, up to 8 such crashes."""
    n = len(glob.glob(f"{d}/infra_error_*.json")) + len(glob.glob(f"{d}/killed_*.json"))
    early = 0
    for f in glob.glob(f"{d}/crashed_*.json"):
        try:
            c = json.load(open(f))
        except Exception:
            c = {}
        if c.get("sandbox_started", True):
            n += 1
        else:
            early += 1
    return n + (MAX_ATT if early >= 8 else 0)


def kill_episode(iid, why):
    p, t0 = running[iid]; d = f"{RUNS}/{H}/{iid}"
    for pg in [p.pid] + ([int(open(f"{d}/sandbox.pgid").read())] if os.path.exists(f"{d}/sandbox.pgid") else []):
        try:
            os.killpg(pg, signal.SIGKILL)
        except Exception:
            pass
    if why:
        json.dump({"iid": iid, "why": why, "after_s": round(time.time() - t0)}, open(f"{d}/killed_{int(time.time())}.json", "w"))


def shutdown(*a):
    for iid in list(running):
        kill_episode(iid, None)
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown); signal.signal(signal.SIGINT, shutdown)
todo = list(ids)
while todo or running:
    for iid, (p, t0) in list(running.items()):
        if p.poll() is not None:
            d = f"{RUNS}/{H}/{iid}"
            if p.returncode not in (0, 3) and not os.path.exists(f"{d}/meta.json"):  # crash outside the episode's own handler
                json.dump({"iid": iid, "rc": p.returncode, "after_s": round(time.time() - t0),
                           "sandbox_started": os.path.exists(f"{d}/sandbox.pgid")}, open(f"{d}/crashed_{int(time.time())}.json", "w"))
            del running[iid]
        elif time.time() - t0 > HARD_S:
            kill_episode(iid, "hard_timeout"); p.wait(); del running[iid]
    while todo and len(running) < W:
        iid = todo.pop(0); d = f"{RUNS}/{H}/{iid}"
        if os.path.exists(f"{d}/meta.json"):
            continue
        os.makedirs(d, exist_ok=True)
        if attempts(d) >= MAX_ATT:
            print("give up", iid, flush=True); continue
        if not os.path.exists(f"{HARNESS_DIR}/sif/{iid}.sqfs"):  # image not built yet: defer, not an attempt
            todo.append(iid)
            if all(not os.path.exists(f"{HARNESS_DIR}/sif/{j}.sqfs") for j in todo):
                break
            continue
        if os.path.exists(f"{d}/sandbox.pgid"):
            os.remove(f"{d}/sandbox.pgid")
        log = open(f"{d}/run.log", "a")
        env = dict(os.environ, PYTHONPATH=f"{HARNESS_DIR}/code/harness/pyshim") if H == "oh" else None  # shadow benchmarks' modal sitecustomize
        running[iid] = (subprocess.Popen([PY, SCRIPT, iid, d, URL], stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=env), time.time())
        print(time.strftime("%H:%M:%S"), "start", H, iid, flush=True)
    # retry pass: re-queue instances whose episode exited without meta.json and with attempts left
    if todo and not running and all(not os.path.exists(f"{HARNESS_DIR}/sif/{j}.sqfs") for j in todo):
        time.sleep(60)  # waiting for images
    if not todo and not running:
        todo = [i for i in ids if not os.path.exists(f"{RUNS}/{H}/{i}/meta.json") and attempts(f"{RUNS}/{H}/{i}") < MAX_ATT]
        if not todo:
            break
    time.sleep(5)
print("scheduler done", H, flush=True)
