"""Test doubles: a Host that keeps systemd units, VRAM and model health in memory, and a stub model server.

FakeHost never runs a command.  A unit is "active" after unit_start; a model's cards come from its
CUDA_VISIBLE_DEVICES (exec mode) or from ":<cards>.service" at the end of its unit name.  VRAM per card
is derived from what is active on it.  Time is a fake clock unless real_time=True (gateway tests).
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cardshare.config import Config, expand  # noqa: E402
from cardshare.host import Host  # noqa: E402

BG_MIB = 29000
MODEL_MIB = 20000
IDLE_MIB = 300


class FakeHost(Host):
    def __init__(self, cfg, real_time=False):
        super().__init__(cfg)
        self.real_time = real_time
        self.t = 1_000_000.0
        self.units = {}
        self.unit_cards = {}
        self.started_at = {}
        self.calls = []                 # ("start"|"stop", unit)
        self.load_delay = {}            # model -> seconds before /health is ok
        self.crash_on_load = set()      # models whose unit dies while loading
        self.start_fails = set()        # models whose start command fails
        self.busy_models = set()        # models whose busy_url says is_processing
        self.stuck_vram = set()         # cards that never free up
        self.http_posts = []
        self.on_start = {}              # model -> callable(cards) (gateway tests start stub servers)
        self.on_stop = {}
        self.real_http = False
        self.lock = threading.Lock()

    # clock
    def now(self):
        return time.time() if self.real_time else self.t

    def sleep(self, sec):
        if self.real_time:
            time.sleep(min(sec, 0.05))
        else:
            self.t += sec

    def advance(self, sec):
        self.t += sec

    # mapping units to models
    def _model_of_unit(self, unit):
        for m in self.cfg.models.values():
            if m.exec and m.transient_unit(self.cfg.unit_prefix) == unit:
                return m
            if m.unit and re.fullmatch(re.escape(m.unit).replace(re.escape("{card}"), r"[\d,]+")
                                       .replace(re.escape("{cards}"), r"[\d,]+")
                                       .replace(re.escape("{name}"), re.escape(m.name)), unit):
                return m
        return None

    def _bg_cards(self, unit):
        return [c for c, b in self.cfg.background.items() if b.unit == unit]

    # systemd
    def run(self, cmd, timeout=120, env=None):
        with self.lock:
            self.calls.append(("run", tuple(cmd)))
        return 0, ""

    def unit_active(self, unit):
        with self.lock:
            m = self._model_of_unit(unit)
            if m and m.name in self.crash_on_load and self.units.get(unit):
                if self.now() - self.started_at.get(unit, 0) >= 0.0:
                    self.units[unit] = False
            return bool(self.units.get(unit))

    def unit_start(self, unit, timeout=180):
        m = self._model_of_unit(unit)
        with self.lock:
            self.calls.append(("start", unit))
            if m and m.name in self.start_fails:
                return 1, "Job for %s failed" % unit
            self.units[unit] = True
            self.started_at[unit] = self.now()
            if unit not in self.unit_cards:
                mm = re.search(r":([\d,]+)\.service$", unit)
                if mm:
                    self.unit_cards[unit] = mm.group(1).split(",")
        if m and m.name in self.on_start:
            self.on_start[m.name](self.unit_cards.get(unit, []))
        return 0, ""

    def unit_stop(self, unit, timeout=180):
        m = self._model_of_unit(unit)
        with self.lock:
            self.calls.append(("stop", unit))
            self.units[unit] = False
        if m and m.name in self.on_stop:
            self.on_stop[m.name]()
        return 0, ""

    def transient_start(self, unit, cmd, env, user=None, properties=()):
        with self.lock:
            self.unit_cards[unit] = str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",")
            self.env_of = getattr(self, "env_of", {})
            self.env_of[unit] = dict(env)
        return self.unit_start(unit)

    # GPUs
    def gpu_used_mib(self):
        if not self.cfg.gpu_query:
            return None
        with self.lock:
            used = {c: IDLE_MIB for c in (self.cfg.cards or ["0", "1"])}
            for unit, on in self.units.items():
                if not on:
                    continue
                cards = self._bg_cards(unit) or self.unit_cards.get(unit, [])
                for c in cards:
                    if c in used:
                        used[c] += BG_MIB if self._bg_cards(unit) else MODEL_MIB
            for c in self.stuck_vram:
                used[c] += BG_MIB
            for m in self.cfg.models.values():
                if getattr(self, "asleep", {}).get(m.name):
                    unit = m.transient_unit(self.cfg.unit_prefix) if m.exec else None
                    for c in self.unit_cards.get(unit, []):
                        used[c] -= MODEL_MIB
            return used

    # HTTP to models
    def http(self, method, url, timeout=3, body=None):
        if self.real_http:
            if method == "POST":
                self.http_posts.append(url)
            return super().http(method, url, timeout, body)
        with self.lock:
            if method == "POST":
                self.http_posts.append(url)
            for m in self.cfg.models.values():
                unit = m.transient_unit(self.cfg.unit_prefix) if m.exec else None
                if unit is None:
                    for u, on in self.units.items():
                        if on and self._model_of_unit(u) is m:
                            unit = u
                if unit is None or not self.units.get(unit):
                    continue
                cards = self.unit_cards.get(unit, [])
                urls = {k: expand(getattr(m, k), m.name, cards) for k in ("health_url", "busy_url") if getattr(m, k)}
                if m.sleep and url in (m.sleep["sleep_url"], m.sleep["wake_url"]):
                    self.asleep = getattr(self, "asleep", {})
                    self.asleep[m.name] = url == m.sleep["sleep_url"]
                    return 200, {"ok": True}
                if url == urls.get("health_url"):
                    if self.now() - self.started_at[unit] >= self.load_delay.get(m.name, 0):
                        return 200, {"status": "ok"}
                    return 503, {"status": "loading"}
                if url == urls.get("busy_url"):
                    return 200, [{"id": 0, "is_processing": m.name in self.busy_models}]
        return None, None


def two_card_config(tmp, models=None, **over):
    """Two cards, a background unit per card (bg0, bg1), exec-mode models with {card} in the env."""
    raw = {
        "state_file": os.path.join(tmp, "state.json"),
        "lock_file": os.path.join(tmp, "cardshare.lock"),
        "run_dir": os.path.join(tmp, "run"),
        "cards": {"0": {"background": {"unit": "bg0.service"}}, "1": {"background": {"unit": "bg1.service"}}},
        "gpu": {"free_threshold_mib": 2000, "free_wait_sec": 10},
        "policy": {"evict_grace_sec": 0, "poll_sec": 1},
        "defaults": {"idle_minutes": 3},
        "models": models if models is not None else default_models(),
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = dict(raw[k], **v)
        else:
            raw[k] = v
    return Config(raw, os.path.join(tmp, "config.json"))


def model(port, klass="small", gpus=(1, 0), **kw):
    m = {"class": klass, "gpus": list(gpus) if gpus != "all" else "all",
         "exec": ["/bin/serve", "--port", str(port)],
         "env": {"CUDA_VISIBLE_DEVICES": "{card}" if gpus != "all" else "{cards}"},
         "health_url": "http://127.0.0.1:%d/health" % port,
         "api_url": "http://127.0.0.1:%d" % port}
    m.update(kw)
    return m


def default_models():
    return {
        "bigA": model(9001, "big", [0]),
        "bigB": model(9002, "big", [0]),
        "smallA": model(9003),
        "smallB": model(9004),
        "smallC": model(9005),
    }


class TmpDir:
    def __init__(self):
        self.td = tempfile.TemporaryDirectory(prefix="cardshare-test-")
        self.path = self.td.name

    def cleanup(self):
        self.td.cleanup()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── a stub model server ───────────────────────────────────────────────────

class StubBackend:
    """OpenAI + Anthropic shaped answers.  stream=True answers in `chunks` SSE events; if `gate` is set,
    the server waits for it before sending the second chunk (to prove the gateway does not buffer)."""

    def __init__(self, port, name="stub", load_delay=0.0, chunks=3):
        self.port, self.name, self.load_delay, self.chunks = port, name, load_delay, chunks
        self.requests = []
        self.gate = None
        self.hold = None                 # threading.Event: answers wait for it (keeps a request in flight)
        self.inflight = 0
        self.httpd = None
        self.t0 = 0.0

    def start(self):
        if self.httpd:
            return
        stub = self
        self.t0 = time.time()

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")      # a stopped stub must really be gone
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self):
                if self.path == "/health":
                    ok = time.time() - stub.t0 >= stub.load_delay
                    return self._json(200 if ok else 503, {"status": "ok" if ok else "loading"})
                if self.path == "/slots":
                    return self._json(200, [{"id": 0, "is_processing": stub.inflight > 0}])
                return self._json(404, {"error": "no"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                stub.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                stub.inflight += 1
                try:
                    if stub.hold is not None:
                        stub.hold.wait(10)
                    if self.path == "/v1/messages/count_tokens":
                        return self._json(200, {"input_tokens": 7})
                    anthropic = self.path == "/v1/messages"
                    if body.get("stream"):
                        return self._stream(anthropic, body)
                    if anthropic:
                        return self._json(200, {"id": "msg_1", "type": "message", "role": "assistant",
                                                "model": body.get("model"),
                                                "content": [{"type": "text", "text": "hi from " + stub.name}],
                                                "stop_reason": "end_turn"})
                    return self._json(200, {"id": "c1", "object": "chat.completion", "model": body.get("model"),
                                            "choices": [{"index": 0, "message": {"role": "assistant",
                                                         "content": "hi from " + stub.name}}]})
                finally:
                    stub.inflight -= 1

            def _stream(self, anthropic, body):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def send(data):
                    raw = data.encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
                    self.wfile.flush()

                for i in range(stub.chunks):
                    if i == 1 and stub.gate is not None:
                        stub.gate.wait(10)
                    if anthropic:
                        send("event: content_block_delta\ndata: %s\n\n" % json.dumps(
                            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "t%d" % i}}))
                    else:
                        send("data: %s\n\n" % json.dumps({"choices": [{"delta": {"content": "t%d" % i}}]}))
                send("event: message_stop\ndata: {}\n\n" if anthropic else "data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
