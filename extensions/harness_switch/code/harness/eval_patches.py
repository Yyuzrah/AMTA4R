"""Grading (CPU): official swebench 5.0.2 eval_script + get_eval_report, executed with Apptainer in the instance image
(`--fakeroot --writable-tmpfs --containall --cleanenv`).
Conditions: gold (dataset patch), mini_native, mini_common, oh_native (patch files of runs/<tag>/<harness>/<iid>/).
An empty patch is not executed and counts as not resolved (as in swebench). Skips tasks whose result.json exists.
Output: eval/<tag>/<cond>/<iid>/result.json (resolved, reason) and the logs of the test run.
Usage: python eval_patches.py <runs_tag> <task_file> <start> <count>   (task_file lines: "<iid> <cond>")"""
import json, os, subprocess, sys, time
from pathlib import Path
from datasets import load_dataset
from swebench.harness.grading import get_eval_report
from swebench.harness.utils import make_test_spec

HARNESS_DIR = Path(os.environ["HARNESS_DIR"]); APPT = os.environ["APPTAINER_BIN"] + "/apptainer"
TAG = sys.argv[1]; TASKS = [l.split() for l in open(sys.argv[2]) if l.strip()]
START, COUNT = int(sys.argv[3]), int(sys.argv[4])
PATCH = {"mini_native": ("mini", "patch_native.diff"), "mini_common": ("mini", "patch_common.diff"),
         "oh_native": ("oh", "patch_native.diff")}
DRIVER = r"""#!/bin/bash
cd /testbed
ok=0; i=0
for c in "git apply --verbose" "git apply --verbose --reject" "patch --batch --fuzz=5 -p1 -i"; do
  if [ $i -gt 0 ]; then git checkout -- . ; git clean -fdq; fi
  if $c /eval/patch.diff >> /eval/apply.log 2>&1; then ok=1; echo ">>>>> Applied Patch ($c)" >> /eval/apply.log; break; fi
  i=$((i+1))
done
if [ $ok = 0 ]; then echo ">>>>> Patch Apply Failed" >> /eval/apply.log; exit 3; fi
git -c core.fileMode=false diff > /eval/diff_before.txt 2>&1
/bin/bash /eval/eval.sh > /eval/test_output.txt 2>&1
echo "eval.sh exit $?" >> /eval/apply.log
"""


def main():
    ds = {x["instance_id"]: x for x in load_dataset("SWE-bench/SWE-bench_Verified", split="test")}
    for iid, cond in TASKS[START:START + COUNT]:
        d = HARNESS_DIR / "eval" / TAG / cond / iid; d.mkdir(parents=True, exist_ok=True)
        if (d / "result.json").exists():
            continue
        inst = ds[iid]; spec = make_test_spec(inst)
        if cond == "gold":
            patch = inst["patch"]
        else:
            h, f = PATCH[cond]; pf = HARNESS_DIR / "runs" / TAG / h / iid / f
            if not (HARNESS_DIR / "runs" / TAG / h / iid / "meta.json").exists():
                continue  # episode not finished yet
            patch = pf.read_text() if pf.exists() else ""
        out = {"instance_id": iid, "cond": cond, "host": os.uname().nodename}
        if not patch.strip():
            out.update(resolved=False, reason="empty_patch")
        else:
            for f in ("apply.log", "test_output.txt"):
                (d / f).unlink(missing_ok=True)
            (d / "patch.diff").write_text(patch if patch.endswith("\n") else patch + "\n")
            (d / "eval.sh").write_text(spec.eval_script); (d / "driver.sh").write_text(DRIVER)
            cmd = [APPT, "exec", "--fakeroot", "--writable-tmpfs", "--containall", "--cleanenv", "--bind", f"{d}:/eval",
                   "--pwd", "/testbed", str(HARNESS_DIR / "sif" / f"{iid}.sqfs"), "/bin/bash", "/eval/driver.sh"]
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                rc, tail = r.returncode, (r.stdout[-3000:] + "\n---\n" + r.stderr[-3000:])
            except subprocess.TimeoutExpired:
                rc, tail = "timeout", ""
            (d / "apptainer_out.txt").write_text(tail)
            out.update(exec_rc=rc, exec_s=round(time.time() - t0, 1))
            apply_log = (d / "apply.log").read_text() if (d / "apply.log").exists() else ""
            if "Patch Apply Failed" in apply_log:
                out.update(resolved=False, reason="patch_apply_failed")
            elif rc == "timeout":
                out.update(resolved=False, reason="eval_timeout")
            else:
                pred = {"instance_id": iid, "model_name_or_path": cond, "model_patch": patch}
                try:
                    rep = get_eval_report(test_spec=spec, prediction=pred, test_log_path=str(d / "test_output.txt"),
                                          include_tests_status=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=1))
                    out.update(resolved=bool(rep[iid]["resolved"]), reason="graded")
                except Exception as e:
                    out.update(resolved=False, reason="grading_error", error=repr(e)[:1000])
            for f in ("test_output.txt",):  # keep logs small: truncate the test log to its tail
                p = d / f
                if p.exists() and p.stat().st_size > 2_000_000:
                    p.write_text(p.read_text(errors="replace")[-200_000:])
        (d / "result.json.tmp").write_text(json.dumps(out, indent=1)); os.replace(d / "result.json.tmp", d / "result.json")
        print(iid, cond, out.get("resolved"), out.get("reason"), flush=True)


if __name__ == "__main__":
    main()
