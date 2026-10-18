"""Image builder without extracting the root file system to disk. For one instance: fetch the GHCR agent-server
image manifest (OCI index -> linux/amd64) and its gzip layer blobs (shared blobs cached under cache/blobs, each checked
against its registry digest), compute the merged filesystem of the layers with OCI whiteout semantics (.wh.<name> deletes, .wh..wh..opq
makes a directory opaque, a non-directory replacing a directory removes its subtree), and stream the merged tree as
one tar into `mksquashfs - <img> -tar`, with all files owned by root (`-all-root -comp gzip`, the options
`apptainer build` passes to mksquashfs). /.singularity.d comes from an apptainer 1.5.4 build (generic files, read from
data/singularity_d_template.tar) plus the env file and runscript generated from the image config following apptainer's
docker conversion; the / symlinks are added as apptainer does. `apptainer build` from docker:// gives the same file tree
but first extracts every image (about 120,000 files) to disk.
Output: sif/<iid>.sqfs (Apptainer 'squashfs' image format), written atomically; skipped if present.
Usage: python build_sqfs.py <instance_list> <index>"""
import gzip, hashlib, io, json, os, shlex, subprocess, sys, tarfile, time, urllib.request

HARNESS_DIR = os.environ["HARNESS_DIR"]; MKSQ = os.environ["APPTAINER_BIN"] + "/../x86_64/libexec/apptainer/bin/mksquashfs"
REPO = "https://ghcr.io/v2/openhands/eval-agent-server"
ACC = ("application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,"
       "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json")
BLOBS = f"{HARNESS_DIR}/cache/blobs"; os.makedirs(BLOBS, exist_ok=True)


def token():
    return json.loads(urllib.request.urlopen("https://ghcr.io/token?scope=repository:openhands/eval-agent-server:pull", timeout=60).read())["token"]


def get(url, tok, accept=ACC, raw=False):
    for a in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "Accept": accept}), timeout=120)
            return r if raw else (r.headers.get("Docker-Content-Digest"), json.loads(r.read()))
        except Exception as e:
            err = e; time.sleep(5 * (a + 1))
    raise err


def manifest(tag, tok):
    dg, m = get(f"{REPO}/manifests/{tag}", tok)
    if "manifests" in m:
        dg = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64" and x.get("platform", {}).get("os") == "linux"][0]["digest"]
        _, m = get(f"{REPO}/manifests/{dg}", tok)
    _, cfg = get(f"{REPO}/blobs/{m['config']['digest']}", tok, accept="*/*")
    return dg, m, cfg


def blob(digest, size, tok):
    p = f"{BLOBS}/{digest.replace(':', '_')}"
    if os.path.exists(p):
        return p
    lock = p + ".lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); break
        except FileExistsError:
            if os.path.exists(p):
                return p
            if time.time() - os.path.getmtime(lock) > 1800:
                os.remove(lock); continue
            time.sleep(10)
    try:
        if os.path.exists(p):
            return p
        for a in range(5):
            try:
                h = hashlib.sha256(); r = get(f"{REPO}/blobs/{digest}", tok, accept="*/*", raw=True)
                with open(p + ".tmp", "wb") as f:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        h.update(b); f.write(b)
                if "sha256:" + h.hexdigest() != digest:
                    raise RuntimeError("digest mismatch " + digest)
                os.replace(p + ".tmp", p); return p
            except Exception as e:
                err = e; time.sleep(10 * (a + 1)); tok = token()
        raise err
    finally:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


def norm(n):
    n = n[2:] if n.startswith("./") else n
    return n.strip("/")


def index_layers(paths):
    """final: path -> (layer, ordinal, isdir); dirs: path -> TarInfo (final definition)."""
    final, children, dirs = {}, {}, {}
    links = []  # per layer: target -> [(ordinal, link name)]

    def add_child(p):
        par = os.path.dirname(p)
        children.setdefault(par, set()).add(p)

    def delete(p, keep_self=False):
        stack = [p]
        while stack:
            q = stack.pop()
            for c in children.pop(q, ()):
                stack.append(c)
            if q == p and keep_self:
                continue
            final.pop(q, None); dirs.pop(q, None)
            par = os.path.dirname(q)
            if par in children:
                children[par].discard(q)
        if keep_self:
            children[p] = set()

    for i, lp in enumerate(paths):
        wh, opq, ents = [], [], []
        lk = {}
        links.append(lk)
        with gzip.open(lp, "rb") as g, tarfile.open(fileobj=g, mode="r|") as tf:
            for k, m in enumerate(tf):
                n = norm(m.name)
                if not n:
                    continue
                b = os.path.basename(n)
                if b == ".wh..wh..opq":
                    opq.append(os.path.dirname(n))
                elif b.startswith(".wh."):
                    wh.append(os.path.join(os.path.dirname(n), b[4:]))
                else:
                    ents.append((n, k, m.isdir(), m if m.isdir() else None))
                    if m.islnk():
                        lk.setdefault(norm(m.linkname), []).append((k, n))
        for p in wh:
            if p in final:
                delete(p)
        for d in opq:
            delete(d, keep_self=True)
        for n, k, isdir, ti in ents:
            old = final.get(n)
            if old is not None and old[2] and not isdir:
                delete(n)
            if old is not None and not old[2] and isdir:
                final.pop(n, None)
            final[n] = (i, k, isdir); add_child(n)
            if isdir:
                ti.name = n; dirs[n] = ti
    return final, dirs, links


def runscript(ep, cmd):
    q = lambda xs: " ".join('"%s"' % x.replace('"', '\\"') for x in xs)
    s = lambda xs: " ".join(shlex.quote(x) for x in xs)
    set_ep = f"set -- {s(ep)} \"$@\"" if ep else ":"
    set_cmd = f"set -- {s(cmd)} \"$@\"" if cmd else ":"
    set_both = f"set -- {s(ep + cmd)} \"$@\"" if (ep or cmd) else ":"
    return f"""#!/bin/sh
OCI_ENTRYPOINT='{q(ep)}'
OCI_CMD='{q(cmd)}'

# When SINGULARITY_NO_EVAL set, use OCI compatible behavior that does
# not evaluate resolved CMD / ENTRYPOINT / ARGS through the shell, and
# does not modify expected quoting behavior of args.
if [ -n "$SINGULARITY_NO_EVAL" ]; then
    # ENTRYPOINT only - run entrypoint plus args
    if [ -z "$OCI_CMD" ] && [ -n "$OCI_ENTRYPOINT" ]; then
        {set_ep}
        exec "$@"
    fi

    # CMD only - run CMD or override with args
    if [ -n "$OCI_CMD" ] && [ -z "$OCI_ENTRYPOINT" ]; then
        if [ $# -eq 0 ]; then
            {set_cmd}
        fi
        exec "$@"
    fi

    # ENTRYPOINT and CMD - run ENTRYPOINT with CMD as default args
    # override with user provided args
    if [ $# -eq 0 ]; then
        {set_both}
    else
        {set_ep}
    fi
    exec "$@"
fi

# Standard Apptainer behavior evaluates CMD / ENTRYPOINT / ARGS
# combination through shell before exec, and requires special quoting
# due to concatenation of CMDLINE_ARGS.
CMDLINE_ARGS=""
# prepare command line arguments for evaluation
for arg in "$@"; do
        CMDLINE_ARGS="${{CMDLINE_ARGS}} \\"$arg\\""
done

# ENTRYPOINT only - run entrypoint plus args
if [ -z "$OCI_CMD" ] && [ -n "$OCI_ENTRYPOINT" ]; then
    if [ $# -gt 0 ]; then
        SINGULARITY_OCI_RUN="${{OCI_ENTRYPOINT}} ${{CMDLINE_ARGS}}"
    else
        SINGULARITY_OCI_RUN="${{OCI_ENTRYPOINT}}"
    fi
fi

# CMD only - run CMD or override with args
if [ -n "$OCI_CMD" ] && [ -z "$OCI_ENTRYPOINT" ]; then
    if [ $# -gt 0 ]; then
        SINGULARITY_OCI_RUN="${{CMDLINE_ARGS}}"
    else
        SINGULARITY_OCI_RUN="${{OCI_CMD}}"
    fi
fi

# ENTRYPOINT and CMD - run ENTRYPOINT with CMD as default args
# override with user provided args
if [ -n "$OCI_CMD" ] && [ -n "$OCI_ENTRYPOINT" ]; then
    if [ $# -gt 0 ]; then
        SINGULARITY_OCI_RUN="${{OCI_ENTRYPOINT}} ${{CMDLINE_ARGS}}"
    else
        SINGULARITY_OCI_RUN="${{OCI_ENTRYPOINT}} ${{OCI_CMD}}"
    fi
fi

# Evaluate shell expressions first and set arguments accordingly,
# then execute final command as first container process
eval "set ${{SINGULARITY_OCI_RUN}}"
exec "$@"
"""


def envfile(env):
    esc = lambda v: v.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    lines = ["#!/bin/sh"]
    for kv in env:
        k, v = kv.split("=", 1)
        lines.append(f'export {k}="{esc(v)}"' if k == "PATH" else f'export {k}="${{{k}:-"{esc(v)}"}}"')
    return "\n".join(lines) + "\n"


def add_bytes(out, name, data, mode=0o755, mtime=None, pax=None):
    ti = tarfile.TarInfo(name); ti.size = len(data); ti.mode = mode; ti.mtime = mtime or int(time.time())
    if pax:
        ti.pax_headers = dict(pax)
    out.addfile(ti, io.BytesIO(data))


def main(lst, idx):
    iid = [l.strip() for l in open(lst) if l.strip()][idx]
    img = os.environ.get("HARNESS_IMG_OUT") or f"{HARNESS_DIR}/sif/{iid}.sqfs"
    if os.path.exists(img):
        print("exists", img); return
    ref = json.load(open(f"{HARNESS_DIR}/data/image_map.json"))[iid]; tag = ref.split(":", 1)[1]
    t0 = time.time(); tok = token()
    dg, m, cfg = manifest(tag, tok)
    assert all("gzip" in l["mediaType"] for l in m["layers"]), m["layers"]
    paths = [blob(l["digest"], l["size"], tok) for l in m["layers"]]
    t_dl = time.time() - t0
    final, dirs, links = index_layers(paths)
    t_idx = time.time() - t0 - t_dl
    tmp = img + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    proc = subprocess.Popen([MKSQ, "-", tmp, "-tar", "-noappend", "-comp", "gzip", "-all-root",
                             "-processors", str(os.cpu_count() if not os.environ.get("SLURM_CPUS_PER_TASK") else int(os.environ["SLURM_CPUS_PER_TASK"])),
                             "-exports", "-mem", "3G", "-quiet", "-no-progress"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stats = dict(files=0, dirs=0, hardlinks=0, hardlink_problems=0, skipped_dupes=0)
    emitted = set()
    with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as out:
        for n in sorted(dirs, key=lambda p: (p.count("/"), p)):
            out.addfile(dirs[n]); stats["dirs"] += 1
        for i, lp in enumerate(paths):
            alias = {}  # target overridden/deleted later but kept alive by a surviving hardlink of this layer
            with gzip.open(lp, "rb") as g, tarfile.open(fileobj=g, mode="r|") as tf:
                for k, mb in enumerate(tf):
                    n = norm(mb.name)
                    if not n or mb.isdir() or os.path.basename(n).startswith(".wh."):
                        continue
                    f = final.get(n)
                    is_final = f is not None and f[0] == i and f[1] == k
                    if not is_final:
                        if mb.isreg():
                            alive = [ln for kl, ln in links[i].get(n, []) if final.get(ln) == (i, kl, False)]
                            if alive:
                                mb.name = alive[0]; out.addfile(mb, tf.extractfile(mb))
                                alias[n] = alive[0]; emitted.add(alive[0]); stats["files"] += 1; stats["hardlinks_materialized"] = stats.get("hardlinks_materialized", 0) + 1
                        continue
                    mb.name = n
                    if mb.islnk():
                        t = norm(mb.linkname)
                        if t in alias:
                            if alias[t] == n:
                                continue
                            mb.linkname = alias[t]
                        elif t in emitted and final.get(t, (None,))[0] == i:
                            mb.linkname = t
                        else:
                            stats["hardlink_problems"] += 1; continue
                        stats["hardlinks"] += 1; out.addfile(mb)
                    elif mb.isreg():
                        out.addfile(mb, tf.extractfile(mb))
                    else:
                        out.addfile(mb)
                    emitted.add(n); stats["files"] += 1
        # apptainer metadata: generic files from an apptainer 1.5.4 build + generated env and runscript
        c = cfg["config"]; now = int(time.time())
        with tarfile.open(f"{HARNESS_DIR}/data/singularity_d_template.tar") as tpl:
            for mb in tpl.getmembers():
                n = norm(mb.name)
                if n in (".singularity.d/env/10-docker2singularity.sh", ".singularity.d/runscript", ".singularity.d/labels.json", ".singularity.d/Singularity"):
                    continue
                mb.name = n
                out.addfile(mb, tpl.extractfile(mb) if mb.isreg() else None)
        add_bytes(out, ".singularity.d/env/10-docker2singularity.sh", envfile(c.get("Env") or []).encode())
        add_bytes(out, ".singularity.d/runscript", runscript(c.get("Entrypoint") or [], c.get("Cmd") or []).encode())
        add_bytes(out, ".singularity.d/Singularity", f"bootstrap: docker\nfrom: {ref}\n\n".encode(), 0o644,
                  pax={"SCHILY.xattr.user.harnessbuild": "1"} if os.environ.get("HARNESS_XATTR") else None)
        labels = dict(c.get("Labels") or {}); labels.update({"org.label-schema.build-arch": "amd64", "org.label-schema.schema-version": "1.0",
                      "org.opencontainers.image.base.name": ref, "org.opencontainers.image.base.digest": dg, "harness.builder": "build_sqfs.py"})
        add_bytes(out, ".singularity.d/labels.json", json.dumps(labels, indent=1).encode(), 0o644)
        for name, target in [("environment", ".singularity.d/env/90-environment.sh"), (".exec", ".singularity.d/actions/exec"),
                             (".run", ".singularity.d/actions/run"), (".shell", ".singularity.d/actions/shell"),
                             ("singularity", ".singularity.d/runscript"), (".test", ".singularity.d/actions/test")]:
            if name not in final:
                ti = tarfile.TarInfo(name); ti.type = tarfile.SYMTYPE; ti.linkname = target; ti.mode = 0o775; ti.mtime = now
                out.addfile(ti)
    proc.stdin.close(); msg = proc.stdout.read().decode(errors="replace"); rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"mksquashfs rc={rc}: {msg[-2000:]}")
    os.replace(tmp, img)
    rec = dict(instance_id=iid, image=ref, digest=dg, n_layers=len(paths), seconds=round(time.time() - t0, 1), t_download=round(t_dl, 1),
               t_index=round(t_idx, 1), bytes=os.path.getsize(img), host=os.uname().nodename, mksquashfs_msg=msg[-500:], **stats)
    with open(f"{HARNESS_DIR}/logs/build/sqfs_builds.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
