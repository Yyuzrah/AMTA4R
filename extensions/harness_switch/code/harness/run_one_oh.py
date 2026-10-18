"""One OpenHands episode on one SWE-bench Verified instance (envs/oh2: OpenHands/benchmarks of 3 June 2026 with
software-agent-sdk 1.16.1, checked out in $HARNESS_DIR/code/benchmarks).
Mirrors benchmarks/swebench/run_infer.py::SWEBenchEvaluation.evaluate_instance of that checkout (default tool preset
without browser, cli_mode system prompt, default LLMSummarizingCondenser max_size=240 keep_first=2, public skills from
a pre-cloned OpenHands/extensions checkout (3 June 2026) that is never updated, default.j2 instruction, /testbed copied
to /workspace/<repo>/, fake-user-response loop with <=10 fake responses, patch = git diff base_commit HEAD after
git add -A + commit). Differences from run_infer: the agent-server container is the sandbox of sandbox.py (localhost
bind); the episode budget is 100 agent LLM calls in total (the conversation is paused once 100 distinct LLM responses
have produced events; steps beyond the 100th are dropped from Phi), and a 3000 s wall limit; LLM = hosted_vllm/Qwen/
Qwen3-8B behind the sampling proxy of proxy.py. Writes <out>/events.json, <out>/patch_native.diff (== patch_common.diff),
<out>/meta.json last (its presence marks the episode done). Usage: python run_one_oh.py <instance_id> <out_dir> <base_url>"""
import json, os, sys, threading, time, traceback
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.chdir(os.path.join(os.environ["HARNESS_DIR"], "code", "benchmarks"))  # benchmarks.utils.version runs `git submodule status` at import
try:  # heavy import chain; a failed import re-executes the process (max 3)
    import openhands.sdk.skills.skill as _sk


    def _pinned_skills_repo(repo_url, branch, cache_dir):  # use the pre-cloned, pinned checkout; never fetch
        p = Path(cache_dir) / "public-skills"
        return p if p.exists() else None


    _sk.update_skills_repository = _pinned_skills_repo
    from benchmarks.swebench import constants
    from benchmarks.swebench.run_infer import get_instruction, get_tools_for_preset
    from benchmarks.utils.agent_context import create_agent_context
    from benchmarks.utils.fake_user_response import fake_user_response, _agent_finished_with_finish_action, _agent_sent_message
    from benchmarks.utils.litellm_proxy import build_eval_llm
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.sdk.event import ActionEvent, MessageEvent
    from openhands.sdk.workspace import RemoteWorkspace
except (ImportError, AttributeError, OSError) as _e:
    _n = int(os.environ.get("HARNESS_IMPORT_TRY", "0"))
    if _n < 3:
        os.environ["HARNESS_IMPORT_TRY"] = str(_n + 1)
        print("import failed, re-exec:", repr(_e)[:300], flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    raise

from harness.sandbox import Sandbox, image_path

HARNESS_DIR = os.environ["HARNESS_DIR"]
BUDGET, WALL_S, MAX_FAKE = 100, 3000, 10
PROMPT = os.path.join(HARNESS_DIR, "code/benchmarks/benchmarks/swebench/prompts/default.j2")


def main(iid, out, base_url):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        print("done", iid); return
    inst = dict(json.load(open(f"{HARNESS_DIR}/data/swebench_verified.json"))[iid])
    llm = LLM(model="hosted_vllm/Qwen/Qwen3-8B", base_url=base_url, api_key=os.environ.get("HARNESS_LLM_KEY", "harness-local"),
              temperature=0.7, top_p=0.8, top_k=20, max_output_tokens=4096, native_tool_calling=True, usage_id="agent")
    t0 = time.time(); meta = {"instance_id": iid, "harness": "openhands", "harness_version": "benchmarks 2026-06-03 / sdk 1.16.1",
                              "sif": image_path(iid), "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID")}
    sb = conv = None; events = []; status = None; n_fake = 0; patch = ""; err = None
    seen, budget_hit = set(), threading.Event()
    try:
        sb = Sandbox(meta["sif"], str(out / "sandbox.log"), pgid_file=str(out / "sandbox.pgid"))
        meta["sandbox_startup_s"] = sb.startup_s
        ws = RemoteWorkspace(host=sb.host, working_dir="/workspace", api_key=sb.key)
        ws.execute_command("export PIP_CACHE_DIR=~/.cache/pip")  # env_setup_commands of run_infer
        agent = Agent(llm=build_eval_llm(llm), tools=get_tools_for_preset(preset="default", enable_browser=False),
                      system_prompt_kwargs={"cli_mode": True},
                      condenser=LLMSummarizingCondenser(llm=build_eval_llm(llm, usage_id="condenser"), max_size=240, keep_first=2),
                      agent_context=create_agent_context())

        def budget_cb(ev):
            if isinstance(ev, ActionEvent) or (isinstance(ev, MessageEvent) and ev.source == "agent"):
                seen.add(getattr(ev, "llm_response_id", None) or ev.id)
                if len(seen) >= BUDGET and not budget_hit.is_set():
                    budget_hit.set()
                    threading.Thread(target=lambda: conv.pause(), daemon=True).start()

        conv = Conversation(agent=agent, workspace=ws, callbacks=[budget_cb], max_iteration_per_run=BUDGET, delete_on_close=True)
        repo_path = f"/workspace/{inst['repo'].split('/')[-1]}/"
        inst["repo_path"] = repo_path
        r = ws.execute_command(f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}", timeout=600)
        assert r.exit_code == 0, f"cp_testbed_repo failed: {r.stderr}"
        r = ws.execute_command(f"cd {repo_path} ; git reset --hard", timeout=300)
        assert r.exit_code == 0, f"git reset failed: {r.stderr}"
        instruction = get_instruction(instance=inst, metadata=SimpleNamespace(details={}, prompt_path=PROMPT),
                                      workspace_path=ws.working_dir)
        conv.send_message(instruction)
        run_retries, skip_run, idle_reruns = 0, False, 0

        def safe_status():
            for _ in range(5):
                try:
                    return conv.state.execution_status
                except Exception:
                    time.sleep(5)
            return None

        while True:  # run_conversation_with_fake_user_response with the shared budget / wall limit
            remaining = WALL_S - (time.time() - t0)
            if remaining <= 5:
                err = "TimeExceeded"; break
            if not skip_run:
                try:
                    conv.run(timeout=remaining)
                except Exception as e:  # stuck/error end state, wall limit, or a transient HTTP timeout to a busy server
                    meta.setdefault("run_exceptions", []).append(repr(e)[:500])
                    st = safe_status()
                    if st is not None and st.value in ("stuck", "error"):
                        err = st.value; break
                    if budget_hit.is_set():
                        err = "LimitsExceeded"; break
                    # transient: the server-side run continues; wait for it to settle within the wall limit
                    run_retries += 1
                    while time.time() - t0 < WALL_S - 5:
                        st = safe_status()
                        if st is not None and st.value not in ("running",):
                            break
                        if budget_hit.is_set():
                            break
                        time.sleep(15)
                    if WALL_S - (time.time() - t0) <= 10:
                        err = "TimeExceeded"; break
                    if run_retries > 10:
                        raise RuntimeError("conversation unreachable: " + repr(e)[:300])  # infrastructure
                    skip_run = True
                    continue
            skip_run = False
            status = conv.state.execution_status
            if budget_hit.is_set():
                err = err or "LimitsExceeded"; break
            if status is not None and status.value in ("stuck", "error"):
                err = status.value; break
            if status != ConversationExecutionStatus.FINISHED:
                if status is not None and status.value in ("idle", "paused") and idle_reruns < 3:
                    idle_reruns += 1
                    continue  # e.g. a settled transient: run again
                break
            evs = list(conv.state.events)
            if _agent_finished_with_finish_action(evs) or not _agent_sent_message(evs) or n_fake >= MAX_FAKE:
                break
            conv.send_message(fake_user_response(conv)); n_fake += 1
        meta["run_retries"] = run_retries
        if err in ("TimeExceeded", "LimitsExceeded"):
            try:
                conv.pause()
            except Exception:
                pass
        status = conv.state.execution_status
        events = [json.loads(e.model_dump_json()) for e in conv.state.events]
        ws.execute_command(f"cd {repo_path} ; git add -A", timeout=300)
        ws.execute_command(f"cd {repo_path} && git config --global user.email '{constants.GIT_USER_EMAIL}' && "
                           f"git config --global user.name '{constants.GIT_USER_NAME}' && "
                           f"git commit --no-verify -m '{constants.GIT_COMMIT_MESSAGE}'", timeout=300)
        r = ws.execute_command(f"cd {repo_path} ; git --no-pager diff --no-color {inst['base_commit']} HEAD", timeout=300)
        patch = r.stdout if r.exit_code == 0 else ""
        meta["patch_rc"] = r.exit_code
        try:
            meta["metrics"] = json.loads(conv.conversation_stats.get_combined_metrics().model_dump_json())
        except Exception:
            pass
    except Exception as e:
        meta["infra_error"] = repr(e)[:2000]; meta["traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            if conv is not None:
                conv.close()
        except Exception:
            pass
        if sb is not None:
            sb.close()
    if "infra_error" in meta:
        (out / f"infra_error_{int(time.time())}.json").write_text(json.dumps(meta, indent=1)); sys.exit(3)
    (out / "events.json").write_text(json.dumps(events))
    (out / "patch_native.diff").write_text(patch); (out / "patch_common.diff").write_text(patch)
    meta.update(exit_status=err or (status.value if status is not None else None), execution_status=status.value if status else None,
                n_llm_responses=len(seen), n_fake_responses=n_fake, n_events=len(events), wall_s=round(time.time() - t0, 1))
    (out / "meta.json.tmp").write_text(json.dumps(meta, indent=1)); os.replace(out / "meta.json.tmp", out / "meta.json")
    print("finished", iid, meta["exit_status"], len(seen))


if __name__ == "__main__":
    main(*sys.argv[1:4])
