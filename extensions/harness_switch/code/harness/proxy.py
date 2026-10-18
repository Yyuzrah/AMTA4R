"""Sampling proxy (127.0.0.1 only): sits between both harnesses and vLLM and makes the decoding configuration
identical for both harnesses. For every /v1/chat/completions request it overwrites temperature=0.7, top_p=0.8,
top_k=20, min_p=0, presence/frequency/repetition penalties neutral, max_tokens=4096 (max_completion_tokens removed),
chat_template_kwargs.enable_thinking=False, and removes seed/n/logit_bias/reasoning_effort; everything else
(messages, tools, tool_choice, parallel_tool_calls, stop, stream) is the harness's own and passed through.
Every request is logged (one JSON line: harness, which keys the harness had set and to what, token usage, finish
reason, latency). Usage: python proxy.py <listen_port> <harness_label> <upstream_base> <log_jsonl>"""
import asyncio, json, sys, time
from aiohttp import web, ClientSession, ClientTimeout

PORT, LABEL, UP, LOG = int(sys.argv[1]), sys.argv[2], sys.argv[3].rstrip("/"), sys.argv[4]
FORCE = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
         "frequency_penalty": 0.0, "repetition_penalty": 1.0, "max_tokens": 4096}
DROP = ["max_completion_tokens", "seed", "n", "logit_bias", "reasoning_effort", "best_of"]
logf = open(LOG, "a")


async def handle(request):
    path = request.match_info["tail"]
    body = await request.read()
    hdrs = {k: v for k, v in request.headers.items() if k.lower() in ("authorization", "content-type", "accept")}
    rec = {"t": time.time(), "harness": LABEL, "path": path}
    stream = False
    if request.method == "POST" and path.endswith("chat/completions"):
        req = json.loads(body)
        rec["harness_set"] = {k: req.get(k) for k in list(FORCE) + DROP + ["chat_template_kwargs"] if k in req}
        rec["model"] = req.get("model"); rec["n_messages"] = len(req.get("messages", [])); rec["n_tools"] = len(req.get("tools") or [])
        rec["tool_choice"] = req.get("tool_choice"); rec["parallel_tool_calls"] = req.get("parallel_tool_calls")
        for k in DROP:
            req.pop(k, None)
        req.update(FORCE)
        ctk = dict(req.get("chat_template_kwargs") or {}); ctk["enable_thinking"] = False; req["chat_template_kwargs"] = ctk
        stream = bool(req.get("stream")); rec["stream"] = stream
        if stream:
            so = dict(req.get("stream_options") or {}); so["include_usage"] = True; req["stream_options"] = so
        body = json.dumps(req).encode()
    t0 = time.time()
    async with ClientSession(timeout=ClientTimeout(total=1800)) as s:
        async with s.request(request.method, f"{UP}/{path}", params=request.query, data=body if body else None, headers=hdrs) as r:
            if stream:
                resp = web.StreamResponse(status=r.status, headers={"Content-Type": r.headers.get("Content-Type", "text/event-stream")})
                await resp.prepare(request)
                tail = b""
                async for chunk in r.content.iter_any():
                    await resp.write(chunk); tail = (tail + chunk)[-4000:]
                await resp.write_eof()
                rec.update(status=r.status, latency=round(time.time() - t0, 3), stream_tail=tail.decode("utf-8", "replace")[-1500:])
            else:
                data = await r.read()
                resp = web.Response(status=r.status, body=data, headers={"Content-Type": r.headers.get("Content-Type", "application/json")})
                rec.update(status=r.status, latency=round(time.time() - t0, 3))
                if path.endswith("chat/completions"):
                    try:
                        j = json.loads(data)
                        if r.status == 200:
                            rec["usage"] = j.get("usage"); rec["finish_reason"] = j["choices"][0].get("finish_reason")
                            rec["n_tool_calls"] = len(j["choices"][0]["message"].get("tool_calls") or [])
                        else:
                            rec["error"] = str(j)[:500]
                    except Exception as e:
                        rec["parse_error"] = repr(e)[:200]
    if path.endswith("chat/completions"):
        logf.write(json.dumps(rec) + "\n"); logf.flush()
    return resp


app = web.Application(client_max_size=256 * 1024 * 1024)
app.router.add_route("*", "/{tail:.*}", handle)
web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)
