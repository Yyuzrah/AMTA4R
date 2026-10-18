"""One mini-SWE-agent 2.4.6 episode on one SWE-bench Verified instance (envs/mini2).
Config = mini's builtin benchmarks/swebench.yaml with these overrides: model hosted_vllm/Qwen/Qwen3-8B behind the
sampling proxy of proxy.py (which fixes the sampling parameters for both harnesses), step_limit=100 LLM calls, cost
limit off, wall-time limit 3000 s; environment = the sandbox of sandbox.py (same image and container flags as
OpenHands).
Writes <out>/traj.json, <out>/patch_native.diff (the submitted patch), <out>/patch_common.diff (git add -A; git diff
--cached of the final working tree), and <out>/meta.json LAST (atomic; its presence marks the episode done).
Usage: python run_one_mini.py <instance_id> <out_dir> <base_url>"""
import json, os, sys, time, traceback
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:  # import everything the episode needs up front; a failed import re-executes the process (max 3)
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge
    import litellm, openai.types, minisweagent.models.litellm_model  # noqa: F401  (pre-load lazy imports)
except (ImportError, AttributeError, OSError) as _e:
    _n = int(os.environ.get("HARNESS_IMPORT_TRY", "0"))
    if _n < 3:
        os.environ["HARNESS_IMPORT_TRY"] = str(_n + 1)
        print("import failed, re-exec:", repr(_e)[:300], flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    raise
from harness.sandbox import Sandbox, image_path
from harness.mini_env import HarnessSandboxEnvironment

HARNESS_DIR = os.environ["HARNESS_DIR"]
STEP_LIMIT, WALL_S = 100, 3000


def main(iid, out, base_url):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        print("done", iid); return
    inst = json.load(open(f"{HARNESS_DIR}/data/swebench_verified.json"))[iid]
    cfg = recursive_merge(get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml")), {
        "agent": {"step_limit": STEP_LIMIT, "cost_limit": 0, "wall_time_limit_seconds": WALL_S},
        "model": {"model_name": "hosted_vllm/Qwen/Qwen3-8B", "cost_tracking": "ignore_errors",
                  "model_kwargs": {"api_base": base_url, "api_key": os.environ.get("HARNESS_LLM_KEY", "harness-local")}}})
    env_cfg = {k: v for k, v in cfg["environment"].items() if k != "environment_class"}
    t0 = time.time(); meta = {"instance_id": iid, "harness": "mini-swe-agent", "harness_version": "2.4.6",
                              "sif": image_path(iid), "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID")}
    sb = agent = None; exit_status, submission, extra = None, "", {}
    try:
        sb = Sandbox(meta["sif"], str(out / "sandbox.log"), pgid_file=str(out / "sandbox.pgid"))
        meta["sandbox_startup_s"] = sb.startup_s
        env = HarnessSandboxEnvironment(sandbox=sb, **env_cfg)
        model = get_model(config=cfg["model"])
        agent = DefaultAgent(model, env, **cfg["agent"])
        try:
            info = agent.run(inst["problem_statement"])
            exit_status, submission = info.get("exit_status"), info.get("submission") or ""
        except Exception as e:  # e.g. ContextWindowExceededError (abort exception in mini) -> exit like mini's runner
            if type(e).__name__ in ("APIConnectionError", "InternalServerError", "ServiceUnavailableError", "Timeout",
                                    "BadGatewayError", "AuthenticationError", "NotFoundError"):
                raise RuntimeError(f"serving failure: {type(e).__name__}: {str(e)[:500]}")  # infrastructure, not the agent
            exit_status, submission = type(e).__name__, ""
            extra = {"exception_str": str(e)[:2000]}
        # common-rule patch: final working tree incl. new files (as OpenHands extracts it)
        c, so, se, _ = sb.exec("cd /testbed && git add -A >/dev/null 2>&1 && git diff --cached --no-color HEAD", timeout=120)
        (out / "patch_common.diff").write_text(so if c == 0 else "")
        meta["common_patch_rc"] = c
    except Exception as e:
        meta["infra_error"] = repr(e)[:2000]; meta["traceback"] = traceback.format_exc()[-4000:]
    finally:
        if sb is not None:
            sb.close()
    if "infra_error" in meta:  # not marked done: the scheduler may retry (at most 3 attempts in all)
        (out / f"infra_error_{int(time.time())}.json").write_text(json.dumps(meta, indent=1)); sys.exit(3)
    agent.save(out / "traj.json", {"info": {"exit_status": exit_status, "submission": submission, **extra}, "instance_id": iid})
    (out / "patch_native.diff").write_text(submission)
    meta.update(exit_status=exit_status, n_calls=agent.n_calls, wall_s=round(time.time() - t0, 1),
                n_messages=len(agent.messages))
    (out / "meta.json.tmp").write_text(json.dumps(meta, indent=1)); os.replace(out / "meta.json.tmp", out / "meta.json")
    print("finished", iid, exit_status, agent.n_calls)


if __name__ == "__main__":
    main(*sys.argv[1:4])
