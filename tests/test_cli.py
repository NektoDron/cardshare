"""The `cardshare` CLI as a subprocess, with real start/stop/check commands (files as "processes")."""

import json
import os
import subprocess
import sys
import time
import unittest

from fakes import StubBackend, TmpDir, free_port

from cardshare.activity import write_lease

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def flag_cmds(path):
    return {"start": ["touch", path], "stop": ["rm", "-f", path], "check": ["test", "-f", path]}


class CLI(unittest.TestCase):
    def setUp(self):
        self.tmp = TmpDir()
        d = self.tmp.path
        self.port = free_port()
        self.stub = StubBackend(self.port)
        self.stub.start()
        f = lambda n: os.path.join(d, n)
        cfg = {
            "state_file": f("state.json"), "lock_file": f("lock"), "run_dir": f("run"),
            "gpu": {"query": None},
            "policy": {"poll_sec": 0.05, "evict_grace_sec": 0},
            "cards": {"0": {"background": flag_cmds(f("bg0"))}, "1": {"background": flag_cmds(f("bg1"))}},
            "models": {
                "big1": dict(flag_cmds(f("big1.{card}")), **{"class": "big", "gpus": [0],
                             "health_url": "http://127.0.0.1:%d/health" % self.port}),
                "big2": dict(flag_cmds(f("big2.{card}")), **{"class": "big", "gpus": [0],
                             "health_url": "http://127.0.0.1:%d/health" % self.port}),
                "small": dict(flag_cmds(f("small.{card}")), **{"gpus": [1, 0], "aliases": ["tiny"],
                              "health_url": "http://127.0.0.1:%d/health" % self.port}),
            },
        }
        self.cfg_path = f("config.json")
        with open(self.cfg_path, "w") as fh:
            json.dump(cfg, fh)
        for b in ("bg0", "bg1"):
            open(f(b), "w").close()
        self.f = f

    def tearDown(self):
        self.stub.stop()
        self.tmp.cleanup()

    def cs(self, *args, env=None):
        e = dict(os.environ, PYTHONPATH=SRC)
        e.pop("CARDSHARE_CONFIG", None)
        e.update(env or {})
        p = subprocess.run([sys.executable, "-m", "cardshare", "--config", self.cfg_path] + list(args),
                           capture_output=True, text=True, env=e, timeout=60)
        return p.returncode, p.stdout, p.stderr

    def test_up_status_down(self):
        rc, out, err = self.cs("up", "small", "--json")
        self.assertEqual(rc, 0, err)
        res = json.loads(out)
        self.assertEqual(res["cards"], ["1"])
        self.assertTrue(os.path.exists(self.f("small.1")))
        self.assertFalse(os.path.exists(self.f("bg1")))
        self.assertTrue(os.path.exists(self.f("bg0")))
        rc, out, _ = self.cs("status", "--json")
        st = json.loads(out)
        self.assertEqual(st["cards"]["1"]["model"], "small")
        self.assertEqual(st["cards"]["0"]["background"], "running")
        rc, out, _ = self.cs("status")
        self.assertIn("card 1", out)
        self.assertIn("small", out)
        rc, out, _ = self.cs("down", "small", "--if-owned")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(self.f("bg1")))
        self.assertFalse(os.path.exists(self.f("small.1")))

    def test_no_room_exit_code_and_json(self):
        self.assertEqual(self.cs("up", "big1")[0], 0)
        write_lease(self.f("run"), "big1", 1, time.time(), time.time())
        rc, out, err = self.cs("up", "big2", "--json")
        self.assertEqual(rc, 75)
        self.assertEqual(json.loads(out)["blockers"], ["big1"])
        self.assertIn("busy", err)

    def test_unknown_model(self):
        rc, _, err = self.cs("up", "nope")
        self.assertEqual(rc, 1)
        self.assertIn("no such model", err)

    def test_background_alias_and_hold(self):
        self.assertEqual(self.cs("background", "stop")[0], 0)
        self.assertFalse(os.path.exists(self.f("bg0")) or os.path.exists(self.f("bg1")))
        self.assertEqual(self.cs("background", "start", "--card", "1")[0], 0)
        self.assertTrue(os.path.exists(self.f("bg1")))
        self.assertFalse(os.path.exists(self.f("bg0")))
        self.cs("hold", "on")
        self.cs("up", "small")
        self.cs("down")
        self.assertFalse(os.path.exists(self.f("bg1")))
        self.cs("hold", "off")
        self.cs("down")
        self.assertTrue(os.path.exists(self.f("bg0")) and os.path.exists(self.f("bg1")))

    def test_env_overrides_paths(self):
        alt = self.f("alt-state.json")
        rc, _, err = self.cs("up", "small", env={"CARDSHARE_STATE": alt})
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.exists(alt))
        self.assertFalse(os.path.exists(self.f("state.json")))

    def test_models_listing(self):
        rc, out, _ = self.cs("models", "--json")
        names = [r["name"] for r in json.loads(out)]
        self.assertEqual(names, ["big1", "big2", "small"])
        rc, out, _ = self.cs("presets")
        self.assertIn("small", out)

    def test_bad_config(self):
        with open(self.cfg_path, "w") as fh:
            fh.write('{"models": {"x": {"health_url": "http://h/health"}}}')
        rc, _, err = self.cs("status")
        self.assertEqual(rc, 2)
        self.assertIn("exactly one of unit, exec, start", err)


if __name__ == "__main__":
    unittest.main()
