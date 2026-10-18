"""mini-SWE-agent environment adapter for the shared sandbox of sandbox.py. Semantics follow minisweagent.environments.singularity /
docker in mini 2.4.6: every action runs in a fresh `bash -c` (interpreter of swebench.yaml) in cwd (default /testbed),
with the config's env vars set, stdout and stderr merged, the config timeout, and the COMPLETE_TASK... submission check."""
import shlex
from typing import Any
from pydantic import BaseModel
from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class HarnessSandboxEnvironmentConfig(BaseModel):
    cwd: str = "/testbed"
    env: dict[str, str] = {}
    timeout: int = 60
    interpreter: list[str] = ["bash", "-c"]


class HarnessSandboxEnvironment:
    def __init__(self, *, sandbox, config_class: type = HarnessSandboxEnvironmentConfig, **kwargs):
        self.sandbox = sandbox
        self.config = config_class(**{k: v for k, v in kwargs.items() if k in config_class.model_fields})

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config.model_dump(mode="json"),
                                    "environment_type": "harness.mini_env.HarnessSandboxEnvironment"}}}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        t = timeout or self.config.timeout
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.config.env.items())
        interp = " ".join(shlex.quote(x) for x in self.config.interpreter)
        full = f"env {envs} timeout -s KILL {t} {interp} {shlex.quote(command)} 2>&1"
        try:
            code, out, err, timed_out = self.sandbox.exec(full, cwd=cwd or self.config.cwd, timeout=t)
            out = out + err
            if timed_out or code in (-1, 137):
                output = {"output": out, "returncode": -1,
                          "exception_info": f"An error occurred while executing the command: Command '{command}' timed out after {t} seconds",
                          "extra": {"exception_type": "TimeoutExpired", "exception": f"timed out after {t} seconds"}}
            else:
                output = {"output": out, "returncode": code, "exception_info": ""}
        except Exception as e:
            output = {"output": "", "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {e}",
                      "extra": {"exception_type": type(e).__name__, "exception": str(e)}}
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted", "submission": submission}})
