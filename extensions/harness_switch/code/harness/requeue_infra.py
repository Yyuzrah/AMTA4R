"""Episodes whose ending was caused by the infrastructure are moved to runs_archive/ and re-run by the scheduler (an
infra_error record is left; it counts as one of the 3 attempts):
  (a) OpenHands: ConversationRunError after transient losses of the agent-server connection;
  (b) both harnesses: TimeExceeded episodes that finished before CONCURRENCY_CHANGE_TIME (2026-10-05 20:04 local
      time), the time at which the number of concurrent episodes per GPU was lowered from 16 to 4 per harness; before
      it, the time limit was reached because of serving latency;
  (c) OpenHands: 'error' episodes whose agent-server log shows an LLM timeout / connection / 5xx failure.
Grading results of archived episodes are moved along. Usage: python requeue_infra.py <runs_tag> [--dry]"""
import glob, json, os, shutil, sys, time
HARNESS_DIR = os.environ["HARNESS_DIR"]; tag = sys.argv[1]; dry = "--dry" in sys.argv
CONCURRENCY_CHANGE_TIME = time.mktime(time.strptime("2026-10-05 20:04:00", "%Y-%m-%d %H:%M:%S"))
SERVING = ("litellm.Timeout", "APIConnectionError", "InternalServerError", "ServiceUnavailable")
n = 0
for h in ("mini", "oh"):
    for f in sorted(glob.glob(f"{HARNESS_DIR}/runs/{tag}/{h}/*/meta.json")):
        m = json.load(open(f)); d = os.path.dirname(f); st = m.get("exit_status")
        why = None
        if h == "oh" and st == "ConversationRunError":
            why = "a: lost agent-server connection"
        elif st == "TimeExceeded" and os.path.getmtime(f) < CONCURRENCY_CHANGE_TIME:
            why = "b: wall limit reached under saturated serving (before 2026-10-05 20:04)"
        elif h == "oh" and st == "error" and os.path.exists(f"{d}/sandbox.log"):
            txt = open(f"{d}/sandbox.log", errors="replace").read()
            if any(s in txt for s in SERVING):
                why = "c: LLM serving failure ended the conversation"
        if not why:
            continue
        iid = os.path.basename(d); n += 1
        print("requeue", h, iid, st, m.get("wall_s"), why)
        if dry:
            continue
        arch = f"{HARNESS_DIR}/runs_archive/{tag}/{h}/{iid}_{int(time.time())}"
        os.makedirs(os.path.dirname(arch), exist_ok=True)
        shutil.move(d, arch)
        for cond in ([f"{h}_native", f"{h}_common"] if h == "mini" else ["oh_native"]):
            ev = f"{HARNESS_DIR}/eval/{tag}/{cond}/{iid}"
            if os.path.isdir(ev):
                shutil.move(ev, f"{arch}_eval_{cond}")
        os.makedirs(d)
        json.dump({"instance_id": iid, "infra_error": f"requeued ({why}); archived at {arch}", "previous_exit_status": st},
                  open(f"{d}/infra_error_{int(time.time())}.json", "w"))
print("requeued", n)
