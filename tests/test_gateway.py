"""Gateway end to end: aiohttp gateway + in-process arbiter on FakeHost + stub model servers on loopback."""

import asyncio
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from fakes import FakeHost, StubBackend, TmpDir, free_port, model, two_card_config

try:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from cardshare.activity import read_lease
    from cardshare.arbiter import Arbiter
    from cardshare.gateway import Gateway, InProcessArbiter, run
except ImportError:                              # aiohttp is optional
    TestClient = None

CHAT = [{"role": "user", "content": "hi"}]


@unittest.skipIf(TestClient is None, "aiohttp not installed")
class GatewayBase(unittest.IsolatedAsyncioTestCase):
    gateway = {}
    api_key = None

    async def asyncSetUp(self):
        self.tmp = TmpDir()
        names = ["bigA", "bigB", "smallA", "glm"]
        self.ports = {n: free_port() for n in names}
        p = self.ports
        models = {
            "bigA": model(p["bigA"], "big", [0]),
            "bigB": model(p["bigB"], "big", [0]),
            "smallA": model(p["smallA"], aliases=["small-alias"]),
            "glm": model(p["glm"], "small", [1], api=["openai"]),
        }
        gw = {"queue_timeout_sec": 1.5, "keepalive_sec": 0.1, "retry_after_sec": 7, "heartbeat_sec": 0.2,
              "request_log": os.path.join(self.tmp.path, "requests.jsonl"),
              "client_classes": [{"name": "agent", "match": {"header": {"User-Agent": "claude-cli"}},
                                  "idle_minutes": 2},
                                 {"name": "chat", "idle_minutes": 3}]}
        gw.update(self.gateway)
        if self.api_key:
            kf = os.path.join(self.tmp.path, "keys")
            with open(kf, "w") as fh:
                fh.write("# keys\nbot:%s\n" % self.api_key)
            gw["api_key_file"] = kf
        self.cfg = two_card_config(self.tmp.path, models, gateway=gw,
                                   aliases={"deep": {"model": "bigA", "upstream_model": "real-name"}},
                                   policy={"evict_grace_sec": 0, "poll_sec": 0.05})
        self.host = FakeHost(self.cfg, real_time=True)
        self.host.real_http = True
        self.host.units.update({"bg0.service": True, "bg1.service": True})
        self.stubs = {n: StubBackend(p[n], n) for n in names}
        for n, s in self.stubs.items():
            self.host.on_start[n] = lambda cards, s=s: s.start()
            self.host.on_stop[n] = s.stop
        self.gw = Gateway(self.cfg, InProcessArbiter(self.cfg, self.host), host=self.host)
        self.client = TestClient(TestServer(self.gw.app(), handler_cancellation=True))
        await self.client.start_server()

    async def asyncTearDown(self):
        for s in self.stubs.values():
            if s.hold:
                s.hold.set()
            if s.gate:
                s.gate.set()
        await self.client.close()
        loop = asyncio.get_running_loop()
        await asyncio.gather(*[loop.run_in_executor(None, s.stop) for s in self.stubs.values()])
        self.tmp.cleanup()

    def starts(self, name):
        return sum(1 for k, u in self.host.calls if k == "start" and u == "cardshare-%s.service" % name)

    def log_lines(self):
        try:
            with open(self.cfg.gateway.request_log) as fh:
                return [json.loads(x) for x in fh]
        except FileNotFoundError:
            return []

    def state(self):
        return Arbiter(self.cfg, self.host, say=lambda m: None).load_state()

    async def chat(self, model_name, stream=False, **kw):
        return await self.client.post("/v1/chat/completions",
                                      json={"model": model_name, "messages": CHAT, "stream": stream}, **kw)

    async def wait_for(self, cond, timeout=5.0):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > timeout:
                raise AssertionError("condition not met in %.1f s" % timeout)
            await asyncio.sleep(0.02)


class Basics(GatewayBase):
    async def test_first_request_brings_the_model_up(self):
        r = await self.chat("smallA")
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["choices"][0]["message"]["content"], "hi from smallA")
        self.assertEqual(self.state()["models"]["smallA"]["cards"], ["1"])
        self.assertFalse(self.host.units["bg1.service"])
        r = await self.chat("smallA")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.starts("smallA"), 1)
        await self.wait_for(lambda: len(self.log_lines()) == 2)
        log = self.log_lines()
        self.assertEqual([x["cold"] for x in log], [True, False])
        self.assertEqual(log[0]["cards"], ["1"])
        self.assertEqual(log[0]["status"], 200)
        self.assertIn("wait_up_sec", log[0])
        self.assertIn("duration_sec", log[1])
        self.assertEqual(log[0]["class"], "default")

    async def test_concurrent_requests_share_one_start(self):
        self.stubs["smallA"].load_delay = 0.4
        rs = await asyncio.gather(*[self.chat("smallA") for _ in range(4)])
        self.assertEqual([r.status for r in rs], [200] * 4)
        self.assertEqual(self.starts("smallA"), 1)

    async def test_streaming_passes_chunk_by_chunk(self):
        await self.chat("smallA")                              # warm
        stub = self.stubs["smallA"]
        stub.gate = threading.Event()
        r = await self.chat("smallA", stream=True)
        self.assertEqual(r.status, 200)
        first = b""
        while b"t0" not in first:
            first += await asyncio.wait_for(r.content.readline(), 3)
        self.assertFalse(stub.gate.is_set())                   # got t0 while t1 is not even sent
        stub.gate.set()
        rest = (await asyncio.wait_for(r.content.read(), 3)).decode()
        self.assertIn("t1", rest)
        self.assertIn("t2", rest)
        self.assertIn("[DONE]", rest)

    async def test_cold_stream_gets_keepalive_comments(self):
        self.stubs["smallA"].load_delay = 0.5
        r = await self.chat("smallA", stream=True)
        self.assertEqual(r.status, 200)
        body = (await asyncio.wait_for(r.content.read(), 5)).decode()
        self.assertTrue(body.startswith(": cardshare: starting smallA"))
        self.assertIn(": cardshare: loading smallA", body)
        self.assertIn("t2", body)
        self.assertLess(body.index(": cardshare"), body.index("data:"))

    async def test_anthropic_messages_and_count_tokens(self):
        r = await self.client.post("/v1/messages", json={"model": "smallA", "max_tokens": 10, "messages": CHAT},
                                   headers={"anthropic-version": "2023-06-01"})
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["type"], "message")
        r = await self.client.post("/v1/messages/count_tokens", json={"model": "smallA", "messages": CHAT})
        self.assertEqual((await r.json())["input_tokens"], 7)
        r = await self.client.post("/v1/messages", json={"model": "smallA", "max_tokens": 10,
                                                         "messages": CHAT, "stream": True})
        body = (await r.read()).decode()
        self.assertIn("event: content_block_delta", body)
        self.assertIn("message_stop", body)

    async def test_responses_route(self):
        r = await self.client.post("/v1/responses", json={"model": "smallA", "input": "hi"})
        self.assertEqual(r.status, 200)
        self.assertEqual(self.stubs["smallA"].requests[-1]["path"], "/v1/responses")

    async def test_anthropic_to_an_openai_only_backend_is_a_clear_error(self):
        r = await self.client.post("/v1/messages", json={"model": "glm", "max_tokens": 10, "messages": CHAT})
        self.assertEqual(r.status, 400)
        j = await r.json()
        self.assertEqual(j["type"], "error")
        self.assertEqual(j["error"]["type"], "invalid_request_error")
        self.assertIn("/v1/chat/completions", j["error"]["message"])
        self.assertEqual(self.starts("glm"), 0)

    async def test_unknown_model(self):
        r = await self.chat("nope")
        self.assertEqual(r.status, 404)
        self.assertEqual((await r.json())["error"]["code"], "model_not_found")
        r = await self.client.post("/v1/chat/completions", data=b"not json")
        self.assertEqual(r.status, 400)

    async def test_models_list(self):
        r = await self.client.get("/v1/models")
        j = await r.json()
        ids = {x["id"]: x["cardshare"] for x in j["data"]}
        self.assertEqual(set(ids), {"bigA", "bigB", "smallA", "glm"})
        self.assertEqual(ids["smallA"]["state"], "off")
        self.assertEqual(sorted(ids["bigA"]["aliases"]), ["deep"])
        await self.chat("smallA")
        j = await (await self.client.get("/v1/models")).json()
        st = {x["id"]: x["cardshare"] for x in j["data"]}["smallA"]
        self.assertEqual((st["state"], st["loaded"], st["cards"]), ("running", True, ["1"]))
        j = await (await self.client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})).json()
        self.assertEqual(j["data"][0]["type"], "model")
        self.assertFalse(j["has_more"])
        r = await self.client.get("/v1/models/glm")
        self.assertEqual((await r.json())["id"], "glm")

    async def test_alias_and_upstream_model_rewrite(self):
        r = await self.chat("deep")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.stubs["bigA"].requests[-1]["body"]["model"], "real-name")
        r = await self.chat("small-alias")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.stubs["smallA"].requests[-1]["body"]["model"], "small-alias")

    async def test_declared_context_available_offline_for_catalog_and_lookup(self):
        self.cfg.models["bigA"].context_window = 65536
        self.cfg.models["smallA"].context_window = 32768
        self.cfg.models["smallA"].listed = False
        self.cfg.gateway.list_aliases = True
        calls_before = list(self.host.calls)

        def no_backend_probe(*args, **kwargs):
            self.fail("model discovery must not query or start a backend")

        self.host.http = no_backend_probe
        for headers in ({}, {"anthropic-version": "2023-06-01"}):
            response = await self.client.get("/v1/models", headers=headers)
            self.assertEqual(response.status, 200)
            entries = {entry["id"]: entry for entry in (await response.json())["data"]}
            for name in ("bigA", "deep"):
                self.assertEqual(entries[name]["context_window"], 65536)
            self.assertNotIn("context_window", entries["bigB"])
            self.assertNotIn("smallA", entries)
            for name, expected in (("bigA", 65536), ("deep", 65536), ("smallA", 32768), ("small-alias", 32768), ("bigB", None)):
                response = await self.client.get("/v1/models/" + name, headers=headers)
                self.assertEqual(response.status, 200)
                entry = await response.json()
                self.assertEqual(entry["id"], name)
                if expected is None:
                    self.assertNotIn("context_window", entry)
                else:
                    self.assertEqual(entry["context_window"], expected)
                self.assertEqual(entry.get("type") if headers else entry.get("object"), "model")
        self.assertEqual(self.host.calls, calls_before)
        self.assertEqual(self.state()["models"], {})
        st = self.gw.reader.load_state()
        st["models"]["bigA"] = {"cards": ["0"], "asleep": True, "since": self.host.now()}
        self.gw.reader.save_state(st)
        for headers in ({}, {"anthropic-version": "2023-06-01"}):
            entries = (await (await self.client.get("/v1/models", headers=headers)).json())["data"]
            entry = next(item for item in entries if item["id"] == "bigA")
            self.assertEqual(entry["cardshare"]["state"], "asleep")
            self.assertEqual(entry["context_window"], 65536)
        self.assertEqual(self.host.calls, calls_before)

    async def test_unlisted_model_routes_without_appearing_in_catalog(self):
        self.cfg.models["bigA"].listed = False
        self.cfg.gateway.list_aliases = True
        for headers in ({}, {"anthropic-version": "2023-06-01"}):
            response = await self.client.get("/v1/models", headers=headers)
            self.assertEqual(response.status, 200)
            ids = {entry["id"] for entry in (await response.json())["data"]}
            self.assertNotIn("bigA", ids)
            self.assertNotIn("deep", ids)
            self.assertIn("smallA", ids)  # listed defaults to true
            self.assertIn("small-alias", ids)
        response = await self.client.get("/v1/models/bigA")
        self.assertEqual(response.status, 200)
        for name, upstream in (("bigA", "bigA"), ("deep", "real-name")):
            response = await self.chat(name)
            self.assertEqual(response.status, 200)
            self.assertEqual(self.stubs["bigA"].requests[-1]["body"]["model"], upstream)

    async def test_hidden_model_still_rejects_generation(self):
        self.cfg.models["bigA"].hidden = True
        self.cfg.models["bigA"].listed = False
        for name in ("bigA", "deep"):
            response = await self.chat(name)
            self.assertEqual(response.status, 404)
        self.assertEqual(self.starts("bigA"), 0)

    async def test_status_endpoint(self):
        await self.chat("smallA")
        j = await (await self.client.get("/cardshare/status")).json()
        self.assertEqual(j["cards"]["1"]["model"], "smallA")


class Queue(GatewayBase):
    async def _hold_bigA(self):
        await self.chat("bigA")                                # up and idle
        self.stubs["bigA"].hold = threading.Event()
        task = asyncio.create_task(self.chat("bigA"))
        await self.wait_for(lambda: self.stubs["bigA"].inflight > 0)
        return task

    async def test_busy_big_makes_the_other_big_wait_then_503(self):
        task = await self._hold_bigA()
        t0 = time.time()
        r = await self.chat("bigB")
        self.assertEqual(r.status, 503)
        self.assertEqual(r.headers.get("Retry-After"), "7")
        self.assertGreaterEqual(time.time() - t0, 1.4)
        self.assertIn("busy", (await r.json())["error"]["message"])
        self.assertTrue(self.host.units["cardshare-bigA.service"])
        self.stubs["bigA"].hold.set()
        self.assertEqual((await task).status, 200)

    async def test_queued_request_proceeds_when_the_big_one_is_done(self):
        task = await self._hold_bigA()
        waiting = asyncio.create_task(self.chat("bigB"))
        await asyncio.sleep(0.3)
        self.assertFalse(waiting.done())
        self.stubs["bigA"].hold.set()
        self.assertEqual((await task).status, 200)
        r = await waiting
        self.assertEqual(r.status, 200)
        self.assertEqual(self.state()["models"].keys(), {"bigB"})
        self.assertFalse(self.host.units["cardshare-bigA.service"])

    async def test_small_on_card_1_while_big_is_busy(self):
        task = await self._hold_bigA()
        r = await self.chat("smallA")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.state()["models"]["smallA"]["cards"], ["1"])
        self.stubs["bigA"].hold.set()
        await task


class Failures(GatewayBase):
    async def test_backend_crash_on_start(self):
        self.host.crash_on_load.add("bigB")
        r = await self.chat("bigB")
        self.assertEqual(r.status, 502)
        self.assertIn("crashed", (await r.json())["error"]["message"])
        self.assertTrue(self.host.units["bg0.service"])        # background returned
        self.assertEqual(self.state()["models"], {})
        await self.wait_for(lambda: len(self.log_lines()) == 1)
        self.assertEqual(self.log_lines()[-1]["status"], 502)

    async def test_backend_crash_on_start_streaming(self):
        self.host.crash_on_load.add("bigB")
        r = await self.client.post("/v1/messages", json={"model": "bigB", "max_tokens": 5, "messages": CHAT,
                                                         "stream": True})
        self.assertEqual(r.status, 200)                        # the SSE stream had started
        body = (await r.read()).decode()
        self.assertIn("event: error", body)
        self.assertIn("crashed", body)
        self.assertTrue(self.host.units["bg0.service"])

    async def test_backend_died_after_start_is_restarted(self):
        await self.chat("smallA")
        await asyncio.get_running_loop().run_in_executor(None, self.stubs["smallA"].stop)
        self.host.units["cardshare-smallA.service"] = False    # the process is gone
        r = await self.chat("smallA")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.starts("smallA"), 2)


class Leases(GatewayBase):
    def test_runner_cancels_disconnected_handlers(self):
        with patch("cardshare.gateway.logging.basicConfig"), patch("cardshare.gateway.web.run_app") as serve:
            run(self.cfg)
        self.assertTrue(serve.call_args.kwargs["handler_cancellation"])

    async def test_cold_disconnect_does_not_cancel_shared_model_start(self):
        self.stubs["smallA"].load_delay = 0.3
        abort = await self.chat("smallA", stream=True)
        peer = await self.chat("smallA", stream=True)
        for response in (abort, peer):
            self.assertIn(b"starting smallA", await response.content.readline())
        self.assertEqual(self.gw.inflight["smallA"], 2)
        abort.close()
        await self.wait_for(lambda: self.gw.inflight["smallA"] == 1)
        self.assertFalse(self.gw.pending["smallA"].cancelled())
        body = await asyncio.wait_for(peer.read(), 3)
        self.assertIn(b"[DONE]", body)
        self.assertEqual(self.starts("smallA"), 1)
        await self.wait_for(lambda: self.gw.inflight["smallA"] == 0)

    async def test_disconnect_cancels_quiet_upstream_and_preserves_peer(self):
        # Real TCP at both hops. The backend sends one chunk, then stays quiet:
        # cancellation must not wait for another token or a read timeout.
        disconnected = {name: asyncio.Event() for name in ("abort", "peer")}
        finish_peer = asyncio.Event()
        closed_at = {}
        loop = asyncio.get_running_loop()

        async def backend(request):
            name = (await request.json())["test_request"]
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b"data: first\n\n")
            try:
                if name == "peer":
                    await finish_peer.wait()
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                    return response
                await asyncio.Event().wait()
            finally:
                closed_at[name] = loop.time()
                disconnected[name].set()

        app = web.Application()
        app.router.add_post("/v1/chat/completions", backend)
        upstream = TestServer(app, handler_cancellation=True)
        await upstream.start_server()
        info = {"cards": ["1"], "api_url": str(upstream.make_url("/"))}
        try:
            with patch.object(self.gw, "ready_info", return_value=info):
                abort, peer = await asyncio.gather(*[
                    self.client.post("/v1/chat/completions", json={
                        "model": "smallA", "stream": True, "test_request": name})
                    for name in ("abort", "peer")])
                for response in (abort, peer):
                    self.assertEqual(await response.content.readuntil(b"\n\n"), b"data: first\n\n")
                self.assertEqual(self.gw.inflight["smallA"], 2)
                started = loop.time()
                abort.close()
                await asyncio.wait_for(disconnected["abort"].wait(), 0.5)
                latency_ms = (closed_at["abort"] - started) * 1000
                print("quiet-upstream disconnect observed: %.3f ms" % latency_ms)
                # A generous correctness bound avoids making host scheduler load
                # a performance test. Report actual timings separately.
                self.assertLess(latency_ms, 500)
                self.assertEqual(self.gw.inflight["smallA"], 1)
                active, _, _ = read_lease(self.cfg.run_dir, "smallA", time.time(), 60)
                self.assertEqual(active, 1)
                self.assertFalse(disconnected["peer"].is_set())
                finish_peer.set()
                self.assertIn(b"[DONE]", await asyncio.wait_for(peer.read(), 0.5))
                await self.wait_for(lambda: self.gw.inflight["smallA"] == 0)
                active, _, _ = read_lease(self.cfg.run_dir, "smallA", time.time(), 60)
                self.assertEqual(active, 0)
                self.assertEqual(sorted(x["status"] for x in self.log_lines()), [200, 499])
        finally:
            finish_peer.set()
            await upstream.close()

    async def test_gc_does_not_stop_a_model_mid_request(self):
        self.cfg.models["smallA"].idle_minutes = 0.002         # 0.12 s
        await self.chat("smallA")
        self.stubs["smallA"].hold = threading.Event()
        task = asyncio.create_task(self.chat("smallA"))
        await self.wait_for(lambda: self.stubs["smallA"].inflight > 0)
        await asyncio.sleep(0.4)
        await self.gw.arb.gc()
        self.assertIn("smallA", self.state()["models"])
        self.stubs["smallA"].hold.set()
        self.assertEqual((await task).status, 200)
        await asyncio.sleep(0.3)
        await self.gw.arb.gc()
        self.assertNotIn("smallA", self.state()["models"])
        self.assertTrue(self.host.units["bg1.service"])

    async def test_client_class_sets_the_idle_hint(self):
        await self.chat("smallA", headers={"User-Agent": "claude-cli/2.0"})
        _, _, idle = read_lease(self.cfg.run_dir, "smallA", time.time(), 60)
        self.assertEqual(idle, 2)
        await self.chat("smallA", headers={"X-Cardshare-Class": "chat"})
        _, _, idle = read_lease(self.cfg.run_dir, "smallA", time.time(), 60)
        self.assertEqual(idle, 3)
        await self.wait_for(lambda: len(self.log_lines()) == 2)
        self.assertEqual([x["class"] for x in self.log_lines()], ["agent", "chat"])
        self.assertNotIn("X-Cardshare-Class", self.stubs["smallA"].requests[-1]["headers"])


class Security(GatewayBase):
    gateway = {"cors_origins": ["https://chat.example.com"]}

    async def test_host_allow_list(self):
        r = await self.chat("smallA", headers={"Host": "evil.example:8100"})
        self.assertEqual(r.status, 403)
        self.assertIn("allowed_hosts", (await r.json())["error"]["message"])
        r = await self.client.get("/v1/models", headers={"Host": "localhost:8100"})
        self.assertEqual(r.status, 200)

    async def test_origin(self):
        r = await self.client.get("/v1/models", headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status, 403)
        r = await self.client.get("/v1/models", headers={"Origin": "https://chat.example.com"})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "https://chat.example.com")

    async def test_preflight(self):
        r = await self.client.options("/v1/chat/completions", headers={
            "Origin": "https://chat.example.com", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type"})
        self.assertEqual(r.status, 204)
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "https://chat.example.com")
        self.assertIn("authorization", r.headers["Access-Control-Allow-Headers"])


class ApiKey(GatewayBase):
    api_key = "s3cret-key"

    async def test_key_required_and_not_forwarded(self):
        r = await self.chat("smallA")
        self.assertEqual(r.status, 401)
        r = await self.client.get("/health")
        self.assertEqual(r.status, 200)
        r = await self.chat("smallA", headers={"Authorization": "Bearer s3cret-key", "Host": "any.name:1"})
        self.assertEqual(r.status, 200)
        self.assertNotIn("Authorization", self.stubs["smallA"].requests[-1]["headers"])
        r = await self.client.post("/v1/messages", json={"model": "smallA", "max_tokens": 1, "messages": CHAT},
                                   headers={"x-api-key": "s3cret-key"})
        self.assertEqual(r.status, 200)
        await self.wait_for(lambda: len(self.log_lines()) >= 2)      # 401s are not logged
        self.assertEqual(self.log_lines()[-1]["key"], "bot")
        r = await self.chat("smallA", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(r.status, 401)


@unittest.skipIf(TestClient is None, "aiohttp not installed")
class StartupAddress(unittest.TestCase):
    def _run(self, host=None, port=None, keys=False, **gw):
        import cardshare.gateway as G
        tmp = TmpDir()
        try:
            if keys:
                kf = os.path.join(tmp.path, "keys")
                with open(kf, "w") as fh:
                    fh.write("k\n")
                gw["api_key_file"] = kf
            c = two_card_config(tmp.path, {"m": model(free_port())}, gateway=gw)
            seen = {}
            orig = G.web.run_app
            G.web.run_app = lambda app, host, port, **kw: seen.update(host=host, port=port)
            try:
                with self.assertLogs("cardshare.gateway", "INFO") as logs:
                    G.run(c, host=host, port=port)
            finally:
                G.web.run_app = orig
            return seen, "\n".join(logs.output)
        finally:
            tmp.cleanup()

    def test_flags_win_and_loopback_is_quiet(self):
        seen, out = self._run(host="127.0.0.1", port=9555)
        self.assertEqual(seen, {"host": "127.0.0.1", "port": 9555})
        self.assertNotIn("WITHOUT API keys", out)

    def test_open_address_without_keys_warns(self):
        seen, out = self._run(host="0.0.0.0")
        self.assertEqual(seen["port"], 8100)
        self.assertIn("WITHOUT API keys", out)
        _, out = self._run(host="0.0.0.0", keys=True)
        self.assertNotIn("WITHOUT API keys", out)


if __name__ == "__main__":
    unittest.main()
