"""Shared sandbox: one OpenHands agent-server container per episode, started from the instance image with the
unprivileged Apptainer install, bound to 127.0.0.1 only and protected by a random per-container session key.
Both harnesses run in this sandbox (identical image, flags and resources):
  - OpenHands talks to the agent-server natively (RemoteWorkspace / RemoteConversation);
  - mini-SWE-agent's environment adapter (mini_env.py) runs each bash action through the server's
    /api/bash/start_bash_command endpoint (one fresh `bash -c` per action, as in mini's own docker/singularity envs).
Container flags = those of OpenHands' ApptainerWorkspace (--fakeroot --compat --no-mount hostfs,bind-paths) except
the bind host (127.0.0.1 instead of 0.0.0.0), --no-home (keeps the image's /root, which holds the .bashrc that
activates the SWE-bench conda environment) and the side services (VSCode/VNC servers off; they would bind 0.0.0.0
and are unused by agents). No host directory is bound; all writes go to the in-memory overlay.
Stdlib + httpx only."""
import os, secrets, signal, socket, subprocess, time
import httpx

APPTAINER = os.environ["APPTAINER_BIN"] + "/apptainer"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Sandbox:
    def __init__(self, sif, log_path, health_timeout=300, pgid_file=None):
        self.sif, self.port, self.key = sif, free_port(), secrets.token_hex(16)
        self.host = f"http://127.0.0.1:{self.port}"
        cmd = [APPTAINER, "run", "--fakeroot", "--compat", "--no-home", "--no-mount", "hostfs", "--no-mount", "bind-paths",
               "--env", f"SESSION_API_KEY={self.key}", "--env", "OH_ENABLE_VSCODE=false", "--env", "OH_ENABLE_VNC=false", sif, "--host", "127.0.0.1", "--port", str(self.port)]
        self.cmd = cmd
        self._log = open(log_path, "a")
        self.t_start = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        if pgid_file:  # lets the scheduler kill an orphaned container if this episode process is killed
            open(pgid_file, "w").write(str(os.getpgid(self.proc.pid)))
        self.client = httpx.Client(timeout=60, headers={"X-Session-API-Key": self.key})
        t0 = time.time()
        while time.time() - t0 < health_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"sandbox exited rc={self.proc.returncode} before healthy")
            try:
                if self.client.get(self.host + "/health", timeout=2).status_code == 200:
                    self.startup_s = round(time.time() - t0, 1)
                    return
            except Exception:
                pass
            time.sleep(1)
        self.close()
        raise RuntimeError("sandbox not healthy in time")

    def exec(self, command, cwd=None, timeout=60):
        """Run one shell command via the agent-server; returns (exit_code, stdout, stderr, timed_out)."""
        payload = {"command": command, "timeout": int(timeout) + 5}
        if cwd:
            payload["cwd"] = cwd
        r = self.client.post(self.host + "/api/bash/start_bash_command", json=payload, timeout=30)
        r.raise_for_status()
        cid = r.json()["id"]
        out, err, code, last = [], [], None, -1
        t0 = time.time()
        while time.time() - t0 < timeout + 30:
            params = {"command_id__eq": cid, "sort_order": "TIMESTAMP", "limit": 100, "kind__eq": "BashOutput"}
            if last >= 0:
                params["order__gt"] = last
            r = self.client.get(self.host + "/api/bash/bash_events/search", params=params, timeout=30)
            r.raise_for_status()
            for ev in r.json().get("items", []):
                if ev.get("order") is not None and ev["order"] > last:
                    last = ev["order"]
                if ev.get("stdout"):
                    out.append(ev["stdout"])
                if ev.get("stderr"):
                    err.append(ev["stderr"])
                if ev.get("exit_code") is not None:
                    code = ev["exit_code"]
            if code is not None:
                break
            time.sleep(0.1)
        timed_out = code is None or code == -1
        return (code if code is not None else -1), "".join(out), "".join(err), timed_out

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
        try:
            self._log.close()
        except Exception:
            pass


def image_path(iid):
    """Instance image built by build_sqfs.py (Apptainer squashfs image format)."""
    return f"{os.environ['HARNESS_DIR']}/sif/{iid}.sqfs"
