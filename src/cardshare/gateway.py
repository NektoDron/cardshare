"""The gateway: one OpenAI + Anthropic compatible endpoint in front of every configured model.

A request names a model (or an alias).  If the model is up, the request is proxied to it as is —
streaming answers pass through chunk by chunk.  If not, the gateway asks the arbiter to bring it up and
holds the request meanwhile (streaming clients get SSE comments as keepalive).  Several requests for a
model that is coming up wait for the same start.  When the model cannot get cards because others are
serving requests, the request waits in a queue up to ``queue_timeout_sec`` and then gets 503 with
Retry-After.

While a request is in flight the gateway keeps an activity lease for its model (see activity.py), so
the arbiter's gc never stops a model in the middle of an answer and counts every request as use.

Needs aiohttp (``pip install cardshare[gateway]``).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import os
import shutil
import sys
import time

from aiohttp import (ClientConnectorError, ClientError, ClientSession, ClientTimeout, ServerDisconnectedError,
                     TCPConnector, web)

from . import __version__
from .activity import write_lease
from .arbiter import EX_BUDGET, EX_NOROOM, Arbiter, ArbiterError
from .security import host_allowed, host_names, key_of_request, load_keys, match_key, origin_allowed

log = logging.getLogger("cardshare.gateway")

ROUTES = {
    "/v1/chat/completions": "openai",
    "/v1/completions": "openai",
    "/v1/embeddings": "openai",
    "/v1/responses": "responses",
    "/v1/messages": "anthropic",
    "/v1/messages/count_tokens": "anthropic",
}
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "host", "content-length"}
AUTH_HEADERS = {"authorization", "x-api-key"}
KEY_NAME = web.RequestKey("key_name", str) if hasattr(web, "RequestKey") else "key_name"


# ── talking to the arbiter ────────────────────────────────────────────────

class ArbResult:
    def __init__(self, ok, code=0, error=None, data=None):
        self.ok, self.code, self.error, self.data = ok, code, error, data or {}


class InProcessArbiter:
    """Calls the arbiter in a worker thread (the gateway runs as root, or in tests)."""

    def __init__(self, cfg, host=None):
        self.cfg, self.host = cfg, host

    def _call(self, op, args, kw):
        arb = Arbiter(self.cfg, self.host, say=lambda m: log.info("arbiter: %s", m))
        try:
            return ArbResult(True, 0, None, getattr(arb, op)(*args, **kw))
        except ArbiterError as exc:
            return ArbResult(False, exc.code, str(exc))
        except Exception as exc:                      # never let a bug in a thread hang a request
            log.exception("arbiter %s failed", op)
            return ArbResult(False, 1, "%s: %s" % (type(exc).__name__, exc))

    async def call(self, op, *args, **kw):
        return await asyncio.get_running_loop().run_in_executor(None, self._call, op, args, kw)

    async def up(self, name):
        return await self.call("up", name)

    async def gc(self):
        return await self.call("gc")


class ExecArbiter:
    """Runs the `cardshare` CLI (usually through `sudo -n`), so the gateway itself needs no root."""

    def __init__(self, cmd, config_path):
        self.cmd, self.config_path = list(cmd), config_path

    async def _run(self, *args):
        argv = self.cmd + (["--config", self.config_path] if self.config_path else []) + list(args)
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        for line in err.decode("utf-8", "replace").splitlines():
            log.info("arbiter: %s", line)
        data = {}
        for line in reversed(out.decode("utf-8", "replace").splitlines()):
            try:
                data = json.loads(line)
                break
            except ValueError:
                continue
        if proc.returncode == 0:
            return ArbResult(True, 0, None, data)
        msg = data.get("error") or (err.decode("utf-8", "replace").strip().splitlines() or ["exit %d" % proc.returncode])[-1]
        return ArbResult(False, proc.returncode, msg, data)

    async def up(self, name):
        return await self._run("up", name, "--json")

    async def gc(self):
        return await self._run("gc", "--json")


def make_arbiter_client(cfg, host=None):
    mode = cfg.gateway.arbiter
    if mode == "inprocess" or (mode == "auto" and os.geteuid() == 0):
        return InProcessArbiter(cfg, host)
    if mode == "auto":
        exe = shutil.which("cardshare")
        cmd = ["sudo", "-n", exe] if exe else ["sudo", "-n", sys.executable, "-m", "cardshare"]
        return ExecArbiter(cmd, cfg.path)
    return ExecArbiter(mode, cfg.path)


# ── request log ───────────────────────────────────────────────────────────

class RequestLog:
    """JSONL, one line per request; when the file passes max_bytes it becomes .1 (.1 -> .2 ...)."""

    def __init__(self, path, max_bytes, keep):
        self.path, self.max_bytes, self.keep = path, max_bytes, keep

    def write(self, rec):
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            if self.max_bytes and os.path.exists(self.path) and os.path.getsize(self.path) >= self.max_bytes:
                for i in range(self.keep - 1, 0, -1):
                    if os.path.exists("%s.%d" % (self.path, i)):
                        os.replace("%s.%d" % (self.path, i), "%s.%d" % (self.path, i + 1))
                os.replace(self.path, self.path + ".1") if self.keep > 0 else os.unlink(self.path)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("request log %s: %s", self.path, exc)


# ── errors in both dialects ───────────────────────────────────────────────

_ANTHROPIC_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                    404: "not_found_error", 503: "overloaded_error"}


def error_body(status, message, anthropic, code=None):
    if anthropic:
        return {"type": "error", "error": {"type": _ANTHROPIC_TYPES.get(status, "api_error"), "message": message}}
    etype = "invalid_request_error" if status < 500 else "server_error"
    return {"error": {"message": message, "type": etype, "code": code}}


def error_response(status, message, anthropic, code=None, retry_after=None):
    headers = {"Retry-After": str(int(retry_after))} if retry_after else None
    return web.json_response(error_body(status, message, anthropic, code), status=status, headers=headers)


def sse_error(status, message, anthropic, code=None):
    body = json.dumps(error_body(status, message, anthropic, code), ensure_ascii=False)
    return (("event: error\ndata: %s\n\n" % body) if anthropic else ("data: %s\n\n" % body)).encode()


# ── the gateway ───────────────────────────────────────────────────────────

class Gateway:
    def __init__(self, cfg, arbiter_client=None, host=None):
        self.cfg = cfg
        self.gw = cfg.gateway
        self.arb = arbiter_client or make_arbiter_client(cfg, host)
        self.reader = Arbiter(cfg, host=host, say=lambda m: None)   # reads state and urls, never locks
        self.inflight = {}
        self.last = {}
        self.cls_idle = {}
        self.pending = {}
        self.place_lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.keys = load_keys(self.gw.api_key_file) if self.gw.api_key_file else {}
        self.names = host_names(self.gw.allowed_hosts)
        self.any_host = "*" in self.gw.allowed_hosts
        self.reqlog = RequestLog(self.gw.request_log, self.gw.request_log_max_bytes, self.gw.request_log_keep)
        self.session = None
        self._lease_warned = False
        self._tasks = []

    # ── app ──────────────────────────────────────────────────────────────
    def app(self):
        app = web.Application(middlewares=[self.guard], client_max_size=256 * 1024 * 1024)
        app.router.add_get("/health", self.h_health)
        app.router.add_get("/v1/models", self.h_models)
        app.router.add_get("/v1/models/{id:.+}", self.h_model)
        app.router.add_get("/cardshare/status", self.h_status)
        for path in ROUTES:
            app.router.add_post(path, self.h_api)
        app.router.add_route("OPTIONS", "/{tail:.*}", self.h_options)
        app.on_response_prepare.append(self._cors)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app):
        self.session = ClientSession(auto_decompress=False, connector=TCPConnector(limit=0))
        self._tasks.append(asyncio.create_task(self._heartbeat()))
        if self.gw.gc_interval > 0:
            self._tasks.append(asyncio.create_task(self._gc_loop()))

    async def _cleanup(self, app):
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        await self.session.close()

    # ── guard: host, origin, key; CORS ───────────────────────────────────
    @web.middleware
    async def guard(self, request, handler):
        anthropic = request.path.startswith("/v1/messages") or "anthropic-version" in request.headers
        if request.method == "OPTIONS":
            return await handler(request)
        host = request.headers.get("Host")
        if not self.keys:
            if not host_allowed(host, self.names, self.any_host):
                return error_response(403, "Host %r is not allowed (DNS rebinding protection); add it to "
                                      "gateway.allowed_hosts or $CARDSHARE_ALLOWED_HOSTS" % host, anthropic)
            origin = request.headers.get("Origin")
            if origin and not origin_allowed(origin, host, self.names, self.gw.cors_origins):
                return error_response(403, "Origin %r is not allowed; add it to gateway.cors_origins" % origin,
                                      anthropic)
        elif request.path != "/health":
            name = match_key(key_of_request(request.headers), self.keys)
            if name is None:
                return error_response(401, "invalid or missing API key", anthropic)
            request[KEY_NAME] = name
        return await handler(request)

    async def _cors(self, request, response):
        if not self.gw.cors_origins or not request.path.startswith("/v1/"):
            return
        origin = (request.headers.get("Origin") or "").rstrip("/")
        if "*" in self.gw.cors_origins:
            allow = "*"
        elif origin in self.gw.cors_origins:
            allow = origin
            response.headers["Vary"] = "Origin"
        else:
            return
        response.headers["Access-Control-Allow-Origin"] = allow
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "Access-Control-Request-Headers") or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta"
        response.headers["Access-Control-Max-Age"] = "600"

    async def h_options(self, request):
        return web.Response(status=204)

    # ── activity leases ──────────────────────────────────────────────────
    def _write_lease(self, name):
        try:
            write_lease(self.cfg.run_dir, name, self.inflight.get(name, 0), self.last.get(name, 0),
                        time.time(), self.cls_idle.get(name))
        except OSError as exc:
            if not self._lease_warned:
                log.warning("cannot write activity lease in %s (%s): gc may stop a model mid-request",
                            self.cfg.run_dir, exc)
                self._lease_warned = True

    def acquire(self, name, idle_minutes):
        self.inflight[name] = self.inflight.get(name, 0) + 1
        self.last[name] = time.time()
        self.cls_idle[name] = idle_minutes
        self._write_lease(name)

    def release(self, name):
        self.inflight[name] = max(0, self.inflight.get(name, 0) - 1)
        self.last[name] = time.time()
        self._write_lease(name)
        ev, self.changed = self.changed, asyncio.Event()
        ev.set()

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(self.gw.heartbeat_sec)
            for name, n in list(self.inflight.items()):
                if n > 0:
                    self._write_lease(name)

    async def _gc_loop(self):
        while True:
            await asyncio.sleep(self.gw.gc_interval)
            async with self.place_lock:
                res = await self.arb.gc()
            if not res.ok:
                log.warning("gc: %s", res.error)

    # ── bringing models up ───────────────────────────────────────────────
    def ready_info(self, name):
        """{cards, api_url} if the state says the model is up and awake, else None."""
        st = self.reader.load_state()
        rec = st["models"].get(name)
        if not rec or rec.get("starting") or rec.get("asleep"):
            return None
        return {"cards": rec["cards"], **self.reader.urls(self.cfg.models[name], rec["cards"])}

    async def ensure_up(self, name):
        """(info, None) or (None, (status, message, retry_after)).  One start per model at a time:
        concurrent callers await the same task."""
        task = self.pending.get(name)
        if task is None or task.done():
            task = asyncio.ensure_future(self._up_loop(name))
            self.pending[name] = task
        return await asyncio.shield(task)

    async def _up_loop(self, name):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.gw.queue_timeout
        while True:
            ev = self.changed
            async with self.place_lock:
                res = await self.arb.up(name)
            if res.ok:
                info = self.ready_info(name)
                if info is None and res.data.get("api_url"):
                    info = {"cards": res.data.get("cards", []), "api_url": res.data["api_url"]}
                if info:
                    return info, None
                return None, (502, "%s came up but the state does not show it" % name, None)
            if res.code == EX_NOROOM:
                left = deadline - loop.time()
                if left <= 0:
                    return None, (503, "%s; waited %d s in the queue" % (res.error, self.gw.queue_timeout),
                                  self.gw.retry_after_sec)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(ev.wait(), timeout=min(5.0, left))
                continue
            if res.code == EX_BUDGET:
                return None, (503, res.error, max(60, self.gw.retry_after_sec))
            return None, (502, "could not bring %s up: %s" % (name, res.error), None)

    # ── client classes ───────────────────────────────────────────────────
    def classify(self, request):
        """(class name, idle minutes or None).  An explicit class header naming a known class wins,
        then the first rule that matches; otherwise "default"."""
        classes = self.gw.client_classes
        want = request.headers.get(self.gw.class_header)
        for c in classes:
            if want and want == c["name"]:
                return c["name"], c["idle_minutes"]
        peer = request.remote or ""
        try:
            ip = ipaddress.ip_address(peer)
        except ValueError:
            ip = None
        for c in classes:
            if c["header"] and not all(p.search(request.headers.get(h, "")) for h, p in c["header"].items()):
                continue
            if c["source"] and not (ip and any(ip in n for n in c["source"])):
                continue
            if c["key"] and request.get(KEY_NAME) not in c["key"]:
                continue
            if c["path"] and not c["path"].search(request.path):
                continue
            if c["header"] or c["source"] or c["key"] or c["path"]:
                return c["name"], c["idle_minutes"]
        return "default", None

    # ── handlers ─────────────────────────────────────────────────────────
    async def h_health(self, request):
        return web.json_response({"status": "ok", "version": __version__})

    def _model_entries(self):
        st = self.reader.load_state()
        out = []
        for name in sorted(self.cfg.models):
            m = self.cfg.models[name]
            if m.hidden or m.missing():
                continue
            rec = st["models"].get(name)
            state = "off"
            if name in self.pending and not self.pending[name].done():
                state = "starting"
            if rec:
                state = "starting" if rec.get("starting") else "asleep" if rec.get("asleep") else "running"
            meta = {"state": state, "loaded": state == "running", "cards": rec["cards"] if rec else None,
                    "class": m.klass, "gpus": m.gpus, "idle_minutes": m.idle_minutes, "sleep": bool(m.sleep),
                    "api": m.api, "aliases": [a for a, t in self.cfg.aliases.items() if t[0] == name],
                    "description": m.desc, "inflight": self.inflight.get(name, 0)}
            out.append((name, m, meta))
            if self.gw.list_aliases:
                for a in meta["aliases"]:
                    out.append((a, m, dict(meta, alias_of=name)))
        return out

    async def h_models(self, request):
        entries = [(n, m, meta) for n, m, meta in self._model_entries() if m.listed]
        capacity = lambda m: {} if m.context_window is None else {"context_window": m.context_window}
        if "anthropic-version" in request.headers:
            data = [{"type": "model", "id": n, "display_name": m.desc or n,
                     "created_at": "1970-01-01T00:00:00Z", "cardshare": meta, **capacity(m)} for n, m, meta in entries]
            return web.json_response({"data": data, "has_more": False,
                                      "first_id": data[0]["id"] if data else None,
                                      "last_id": data[-1]["id"] if data else None})
        return web.json_response({"object": "list", "data": [
            {"id": n, "object": "model", "created": 0, "owned_by": "cardshare", "cardshare": meta, **capacity(m)}
            for n, m, meta in entries]})

    async def h_model(self, request):
        want = request.match_info["id"]
        for n, m, meta in self._model_entries():
            if n == want:
                capacity = {} if m.context_window is None else {"context_window": m.context_window}
                if "anthropic-version" in request.headers:
                    return web.json_response({"type": "model", "id": n, "display_name": m.desc or n,
                                              "created_at": "1970-01-01T00:00:00Z", "cardshare": meta, **capacity})
                return web.json_response({"id": n, "object": "model", "created": 0, "owned_by": "cardshare",
                                          "cardshare": meta, **capacity})
        return error_response(404, "no model %r" % want, "anthropic-version" in request.headers,
                              code="model_not_found")

    async def h_status(self, request):
        st = self.reader.status(probe=False)
        st["gateway"] = {"inflight": {k: v for k, v in self.inflight.items() if v},
                         "starting": [n for n, t in self.pending.items() if not t.done()]}
        return web.json_response(st)

    async def h_api(self, request):
        kind = ROUTES[request.path]
        anthropic = kind == "anthropic"
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "t": round(time.time(), 3), "path": request.path,
               "client": request.remote, "model": None, "status": None}
        try:
            resp = await self._api(request, kind, anthropic, rec)
            rec["status"] = rec["status"] or resp.status
            return resp
        except asyncio.CancelledError:
            rec["status"] = 499
            raise
        finally:
            rec["duration_sec"] = round(loop.time() - t0, 3)
            self.reqlog.write(rec)

    async def _api(self, request, kind, anthropic, rec):
        raw = await request.read()
        try:
            body = json.loads(raw)
        except ValueError:
            return error_response(400, "the body is not JSON", anthropic)
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            return error_response(400, "'model' is required", anthropic)
        rec["model"] = body["model"]
        found = self.cfg.resolve(body["model"])
        m = self.cfg.models.get(found[0]) if found else None
        if not m or m.hidden or m.missing():
            return error_response(404, "no model %r; GET /v1/models lists them" % body["model"], anthropic,
                                  code="model_not_found")
        name, upstream = found[0], found[1] or m.upstream_model
        rec["preset"] = name
        if kind not in m.api:
            hint = {"anthropic": "use /v1/chat/completions", "responses": "use /v1/chat/completions",
                    "openai": "use /v1/messages"}[kind]
            return error_response(400, "model %s does not support the %s API (it speaks: %s); %s"
                                  % (name, kind, ", ".join(m.api), hint), anthropic, code="unsupported_api")
        if upstream and body["model"] != upstream:
            body["model"] = upstream
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        stream = bool(body.get("stream")) and not request.path.endswith("count_tokens")
        rec["stream"] = stream
        cls, idle = self.classify(request)
        rec["class"] = cls
        if request.get(KEY_NAME):
            rec["key"] = request[KEY_NAME]

        self.acquire(name, idle)
        try:
            resp = None
            info = self.ready_info(name)
            rec["cold"] = info is None
            if info is None:
                loop = asyncio.get_running_loop()
                t_up = loop.time()
                ka = None
                if stream:
                    resp = await self._start_sse(request, name)
                    ka = asyncio.create_task(self._keepalive(resp, name, t_up))
                try:
                    info, err = await self.ensure_up(name)
                finally:
                    if ka:
                        ka.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await ka
                rec["wait_up_sec"] = round(loop.time() - t_up, 3)
                if err:
                    status, msg, retry = err
                    rec["status"], rec["error"] = status, msg
                    if resp is not None:
                        with contextlib.suppress(ConnectionError, RuntimeError):
                            await resp.write(sse_error(status, msg, anthropic))
                            await resp.write_eof()
                        return resp
                    return error_response(status, msg, anthropic, retry_after=retry)
            rec["cards"] = info["cards"]
            return await self._proxy(request, name, m, info, raw, resp, anthropic, rec)
        finally:
            self.release(name)

    async def _start_sse(self, request, name):
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        await resp.write((": cardshare: starting %s\n\n" % name).encode())
        return resp

    async def _keepalive(self, resp, name, t0):
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.gw.keepalive_sec)
            await resp.write((": cardshare: loading %s, %d s\n\n" % (name, loop.time() - t0)).encode())

    def _upstream_headers(self, request, m):
        drop = HOP | (AUTH_HEADERS if self.keys else set())
        h = {k: v for k, v in request.headers.items() if k.lower() not in drop}
        h.pop(self.gw.class_header, None)
        h.update(m.upstream_headers)
        return h

    async def _proxy(self, request, name, m, info, raw, resp, anthropic, rec, retried=False):
        base = info["api_url"].rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = base + request.path_qs
        timeout = ClientTimeout(total=None, sock_connect=10, sock_read=self.gw.upstream_read_timeout)
        sent = 0
        try:
            async with self.session.post(url, data=raw, headers=self._upstream_headers(request, m),
                                         timeout=timeout) as r:
                rec["status"] = r.status
                if resp is None:
                    resp = web.StreamResponse(status=r.status, headers={
                        k: v for k, v in r.headers.items() if k.lower() not in HOP})
                    await resp.prepare(request)
                elif r.status != 200:
                    text = (await r.read()).decode("utf-8", "replace")[:2000]
                    await resp.write(sse_error(r.status, "%s answered %d: %s" % (name, r.status, text), anthropic))
                    await resp.write_eof()
                    return resp
                async for chunk in r.content.iter_any():
                    await resp.write(chunk)
                    sent += len(chunk)
                    self.last[name] = time.time()
                await resp.write_eof()
                rec["bytes"] = sent
                return resp
        except (ClientConnectorError, ServerDisconnectedError) as exc:
            # the state said "up" but nobody listens (or the server dropped us before answering):
            # let the arbiter check the model and restart it if it died, then try once more
            if not retried and rec["status"] is None and sent == 0:
                log.warning("%s: %s — asking the arbiter to check it", name, exc)
                info2, err = await self.ensure_up(name)
                if not err:
                    return await self._proxy(request, name, m, info2, raw, resp, anthropic, rec, retried=True)
            msg = "%s is not reachable: %s" % (name, exc)
            rec["status"], rec["error"] = 502, msg
            if resp is not None and resp.prepared:
                with contextlib.suppress(ConnectionError, RuntimeError):
                    await resp.write(sse_error(502, msg, anthropic))
                    await resp.write_eof()
                return resp
            return error_response(502, msg, anthropic)
        except (ClientError, asyncio.TimeoutError) as exc:
            msg = "%s: upstream failed: %s" % (name, exc or type(exc).__name__)
            rec["status"], rec["error"] = 502, msg
            log.warning(msg)
            if resp is not None and resp.prepared:
                with contextlib.suppress(ConnectionError, RuntimeError):
                    await resp.write_eof()
                return resp
            return error_response(502, msg, anthropic)
        except ConnectionError:                     # the client went away
            rec["status"] = 499
            rec["bytes"] = sent
            return resp


def is_loopback(host):
    if host in ("localhost", "") or host.endswith(".localhost"):
        return host != ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def run(cfg, host=None, port=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if host:
        cfg.gateway.host = host.strip("[]")
    if port:
        cfg.gateway.port = int(port)
    gw = Gateway(cfg)
    h, p = cfg.gateway.host, cfg.gateway.port
    if not is_loopback(h) and not gw.keys:
        bar = "!" * 78
        log.warning(bar)
        log.warning("listening on %s:%d WITHOUT API keys: anyone who can reach this address can use", h, p)
        log.warning("every model and make this machine start and stop them. Set gateway.api_key_file")
        log.warning("(or $CARDSHARE_API_KEY_FILE), or listen on 127.0.0.1 if you do not need remote access.")
        log.warning(bar)
    log.info("cardshare gateway %s on %s:%d, %d models, arbiter %s", __version__, h, p,
             len(cfg.models), type(gw.arb).__name__)
    # Cancel a disconnected client even while waiting for the next upstream chunk.
    # _proxy closes its response and _api releases its lease during cancellation;
    # ensure_up shields the shared model-start task from an individual client.
    web.run_app(gw.app(), host=h, port=p, print=None, handler_cancellation=True)
    return 0
