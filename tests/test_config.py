import json
import os
import unittest

from fakes import TmpDir, model

from cardshare.config import Config, ConfigError, expand, load_config
from cardshare.security import host_allowed, host_names, load_keys, match_key, origin_allowed


def cfg(models=None, **kw):
    raw = {"models": models or {"m": model(9000)}}
    raw.update(kw)
    return Config(raw)


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        c = cfg()
        m = c.models["m"]
        self.assertEqual(m.idle_minutes, 3)
        self.assertEqual(m.klass, "small")
        self.assertEqual(m.api, ["openai", "anthropic", "responses"])
        self.assertEqual(m.ports, [9000])
        self.assertIsNone(m.sleep)
        self.assertTrue(m.listed)
        self.assertIsNone(m.context_window)
        self.assertEqual(c.gateway.port, 8100)

    def test_unlisted_model_keeps_canonical_and_alias_routes(self):
        c = cfg({"m": model(9000, listed=False, aliases=["alias"])})
        self.assertFalse(c.models["m"].listed)
        self.assertFalse(c.models["m"].hidden)
        self.assertEqual(c.resolve("m"), ("m", None))
        self.assertEqual(c.resolve("alias"), ("m", None))

    def test_context_window_requires_positive_integer(self):
        self.assertEqual(cfg({"m": model(9000, context_window=65536)}).models["m"].context_window, 65536)
        for value in (None, False, True, 0, -1, "65536", 65536.0, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "context_window"):
                cfg({"m": model(9000, context_window=value)})

    def test_defaults_section_and_env(self):
        c = cfg(defaults={"idle_minutes": 7})
        self.assertEqual(c.models["m"].idle_minutes, 7)
        os.environ["CARDSHARE_IDLE_MINUTES"] = "2"
        try:
            self.assertEqual(cfg().models["m"].idle_minutes, 2)
        finally:
            del os.environ["CARDSHARE_IDLE_MINUTES"]

    def test_gateway_address_priority(self):
        self.assertEqual((cfg().gateway.host, cfg().gateway.port), ("127.0.0.1", 8100))
        c = cfg(gateway={"host": "0.0.0.0", "port": 9000})
        self.assertEqual((c.gateway.host, c.gateway.port), ("0.0.0.0", 9000))
        os.environ.update(CARDSHARE_HOST="::", CARDSHARE_PORT="9100")
        try:
            c = cfg(gateway={"host": "0.0.0.0", "port": 9000})
            self.assertEqual((c.gateway.host, c.gateway.port), ("::", 9100))
        finally:
            del os.environ["CARDSHARE_HOST"], os.environ["CARDSHARE_PORT"]
        with self.assertRaises(ConfigError):
            cfg(gateway={"port": "eighty"})

    def test_sleep_urls(self):
        m = cfg({"m": model(9000, sleep=True)}).models["m"]
        self.assertEqual(m.sleep["sleep_url"], "http://127.0.0.1:9000/sleep")
        self.assertTrue(m.sleep["frees_card"])

    def test_aliases(self):
        c = cfg({"a": model(1, aliases=["x"]), "b": model(2)},
                aliases={"y": "b", "z": {"model": "a", "upstream_model": "real"}})
        self.assertEqual(c.resolve("x"), ("a", None))
        self.assertEqual(c.resolve("y"), ("b", None))
        self.assertEqual(c.resolve("z"), ("a", "real"))
        self.assertEqual(c.resolve("a"), ("a", None))
        self.assertIsNone(c.resolve("q"))
        with self.assertRaises(ConfigError):
            cfg({"a": model(1)}, aliases={"y": "nope"})

    def test_validation(self):
        bad = [
            {"m": {"health_url": "http://x/health"}},
            {"m": dict(model(1), **{"class": "huge"})},
            {"m": dict(model(1), gpus="sometimes")},
            {"m": dict(model(1), start=["x"])},
            {"m": dict(model(1), api=["grpc"])},
        ]
        for models in bad:
            with self.assertRaises(ConfigError, msg=str(models)):
                cfg(models)
        with self.assertRaises(ConfigError):
            cfg(cards={"0": {"background": {"start": ["x"]}}})
        with self.assertRaises(ConfigError):
            cfg(gateway={"client_classes": [{"name": "a", "match": {"ip": "x"}}]})

    def test_expand_only_three_tokens(self):
        self.assertEqual(expand(["${HOME}", "{name}@{card}", "{cards}", "{other}"], "m", ["1", "0"]),
                         ["${HOME}", "m@1", "1,0", "{other}"])

    def test_shared_background(self):
        c = cfg(cards={"0": {"background": {"unit": "u"}}, "1": {"background": {"unit": "u"}}, "2": {}})
        self.assertEqual(c.background["0"].cards, ["0", "1"])
        self.assertEqual(c.cards, ["0", "1", "2"])

    def test_load_json_and_toml(self):
        tmp = TmpDir()
        try:
            p = os.path.join(tmp.path, "c.json")
            with open(p, "w") as fh:
                json.dump({"models": {"m": model(1)}}, fh)
            self.assertEqual(list(load_config(p).models), ["m"])
            t = os.path.join(tmp.path, "c.toml")
            with open(t, "w") as fh:
                fh.write('[models.m]\nexec = ["/bin/x"]\nhealth_url = "http://127.0.0.1:5/health"\n'
                         'gpus = [1]\n[gateway]\nhost = "0.0.0.0"\nport = 8200\n')
            c = load_config(t)
            self.assertEqual(c.models["m"].gpus, ["1"])
            self.assertEqual(c.gateway.port, 8200)
            os.environ["CARDSHARE_CONFIG"] = p
            try:
                self.assertEqual(load_config().path, p)
            finally:
                del os.environ["CARDSHARE_CONFIG"]
            with self.assertRaises(ConfigError):
                load_config(os.path.join(tmp.path, "missing.json"))
        finally:
            tmp.cleanup()


class SecurityTests(unittest.TestCase):
    def test_hosts(self):
        names = host_names(["box.lan", ".home.arpa"])
        for ok in ("127.0.0.1:8100", "198.51.100.5:8100", "[::1]:8100", "localhost", "a.localhost",
                   "box.lan:8100", "x.home.arpa", "home.arpa", None, ""):
            self.assertTrue(host_allowed(ok, names), ok)
        for bad in ("evil.com", "box.lan.evil.com:8100"):
            self.assertFalse(host_allowed(bad, names), bad)
        self.assertTrue(host_allowed("evil.com", names, any_host=True))

    def test_origins(self):
        names = host_names(["box.lan"])
        self.assertTrue(origin_allowed("http://box.lan:3000", "box.lan:8100", names))
        self.assertTrue(origin_allowed("http://203.0.113.2:8100", "203.0.113.2:8100", names))   # own page
        self.assertFalse(origin_allowed("http://203.0.113.3:8100", "203.0.113.2:8100", names))
        self.assertFalse(origin_allowed("null", "box.lan", names))
        self.assertTrue(origin_allowed("chrome-extension://abc", "box.lan", names))
        self.assertTrue(origin_allowed("https://chat.x", "box.lan", names, ["https://chat.x"]))

    def test_keys(self):
        tmp = TmpDir()
        try:
            p = os.path.join(tmp.path, "k")
            with open(p, "w") as fh:
                fh.write("# c\nalice:k1\nplain-key-without-name\n")
            keys = load_keys(p)
            self.assertEqual(match_key("k1", keys), "alice")
            self.assertEqual(match_key("plain-key-without-name", keys), "key3")
            self.assertIsNone(match_key("k2", keys))
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
