"""Arbiter policy and lifecycle against FakeHost (no systemd, no GPUs, fake clock)."""

import json
import os
import unittest

from fakes import FakeHost, TmpDir, default_models, model, two_card_config

from cardshare.activity import write_lease
from cardshare.arbiter import Arbiter, ArbiterError, BudgetExhausted, NoRoom, StartFailed


class Base(unittest.TestCase):
    models = None
    over = {}

    def setUp(self):
        self.tmp = TmpDir()
        self.cfg = two_card_config(self.tmp.path, self.models, **self.over)
        self.host = FakeHost(self.cfg)
        self.msgs = []
        self.arb = Arbiter(self.cfg, self.host, say=self.msgs.append)
        # both background workloads run at the start
        self.host.units.update({"bg0.service": True, "bg1.service": True})

    def tearDown(self):
        self.tmp.cleanup()

    def state(self):
        return self.arb.load_state()

    def slots(self):
        with open(self.cfg.state_file) as fh:
            return json.load(fh)["slots"]

    def bg(self, card):
        return self.host.units.get("bg%s.service" % card)

    def lease(self, name, inflight, last=None):
        write_lease(self.cfg.run_dir, name, inflight, last or self.host.now(), self.host.now())


class Placement(Base):
    def test_small_goes_to_card_1_and_stops_only_its_background(self):
        r = self.arb.up("smallA")
        self.assertEqual(r["cards"], ["1"])
        self.assertEqual(self.slots(), {"0": None, "1": "smallA"})
        self.assertFalse(self.bg(1))
        self.assertTrue(self.bg(0))
        self.assertEqual(self.host.env_of["cardshare-smallA.service"]["CUDA_VISIBLE_DEVICES"], "1")

    def test_big_on_card_0_beside_small_on_card_1(self):
        self.arb.up("smallA")
        r = self.arb.up("bigA")
        self.assertEqual(r["cards"], ["0"])
        self.assertEqual(self.slots(), {"0": "bigA", "1": "smallA"})
        self.assertFalse(self.bg(0) or self.bg(1))

    def test_second_small_takes_the_free_card_0(self):
        self.arb.up("smallA")
        r = self.arb.up("smallB")
        self.assertEqual(r["cards"], ["0"])
        self.assertIn("cardshare-smallA.service", [u for k, u in self.host.calls if k == "start"])
        self.assertTrue(self.host.units["cardshare-smallA.service"])

    def test_explicit_card(self):
        r = self.arb.up("smallA", card="0")
        self.assertEqual(r["cards"], ["0"])
        with self.assertRaises(ArbiterError):
            self.arb.up("bigA", card="1")

    def test_already_up_is_a_no_op(self):
        self.arb.up("smallA")
        starts = len(self.host.calls)
        r = self.arb.up("smallA")
        self.assertTrue(r["already"])
        self.assertEqual(len(self.host.calls), starts)


class Eviction(Base):
    def test_big_evicts_idle_big_without_returning_the_background_between(self):
        self.arb.up("bigA")
        self.host.advance(60)
        r = self.arb.up("bigB")
        self.assertEqual(r["cards"], ["0"])
        self.assertEqual(r["evicted"], ["bigA"])
        self.assertFalse(self.host.units["cardshare-bigA.service"])
        self.assertNotIn(("start", "bg0.service"), self.host.calls)      # card 0 handed over directly
        self.assertEqual(self.slots()["0"], "bigB")

    def test_big_does_not_evict_a_big_serving_a_request_lease(self):
        self.arb.up("bigA")
        self.host.advance(60)
        self.lease("bigA", 1)
        with self.assertRaises(NoRoom) as cm:
            self.arb.up("bigB")
        self.assertEqual(cm.exception.blockers, ["bigA"])
        self.assertEqual(cm.exception.code, 75)
        self.assertTrue(self.host.units["cardshare-bigA.service"])
        self.lease("bigA", 0)
        self.assertEqual(self.arb.up("bigB")["cards"], ["0"])

    def test_busy_by_the_servers_own_slots(self):
        self.cfg.models["bigA"].busy_url = "http://127.0.0.1:9001/slots"
        self.arb.up("bigA")
        self.host.advance(60)
        self.host.busy_models.add("bigA")
        with self.assertRaises(NoRoom):
            self.arb.up("bigB")
        self.host.busy_models.clear()
        self.arb.up("bigB")

    def test_stale_lease_does_not_block(self):
        self.arb.up("bigA")
        self.lease("bigA", 3)
        self.host.advance(self.cfg.lease_stale_sec + 1)
        self.arb.up("bigB")

    def test_evict_grace(self):
        self.cfg.evict_grace_sec = 30
        self.arb.up("bigA")
        self.host.advance(10)
        with self.assertRaises(NoRoom):
            self.arb.up("bigB")
        self.host.advance(25)
        self.arb.up("bigB")

    def test_lru_between_two_idle_smalls(self):
        self.arb.up("smallA")                       # card 1
        self.arb.up("smallB")                       # card 0
        self.host.advance(100)
        self.lease("smallA", 0)                     # A used just now, B long ago
        self.host.advance(100)
        r = self.arb.up("smallC")
        self.assertEqual(r["cards"], ["0"])
        self.assertEqual(r["evicted"], ["smallB"])

    def test_busy_card_is_skipped_for_the_idle_one(self):
        self.arb.up("smallA")                       # card 1
        self.arb.up("smallB")                       # card 0
        self.host.advance(100)
        self.lease("smallA", 1)
        r = self.arb.up("smallC")
        self.assertEqual((r["cards"], r["evicted"]), (["0"], ["smallB"]))

    def test_big_evicts_idle_small_on_card_0(self):
        self.arb.up("smallA")
        self.arb.up("smallB")                       # card 0
        self.host.advance(100)
        r = self.arb.up("bigA")
        self.assertEqual((r["cards"], r["evicted"]), (["0"], ["smallB"]))
        self.assertTrue(self.host.units["cardshare-smallA.service"])

    def test_no_switch(self):
        self.arb.up("bigA")
        self.host.advance(100)
        with self.assertRaises(NoRoom):
            self.arb.up("bigB", no_switch=True)

    def test_preempt_evicts_busy(self):
        self.arb.up("bigA")
        self.lease("bigA", 1)
        self.arb.up("bigB", preempt=True)


class PortsMemoryAllCards(Base):
    models = dict(default_models(),
                  twinA=model(9100, gpus=[1]), twinB=model(9100, gpus=[0]),
                  wide=model(9200, "big", "all", host_mem_gib=100))

    def test_port_conflict_evicts_the_other_preset(self):
        self.arb.up("twinA")
        self.host.advance(100)
        r = self.arb.up("twinB")
        self.assertEqual(r["evicted"], ["twinA"])
        self.assertTrue(self.bg(1))                 # twinA's card went back to its background

    def test_model_on_all_cards(self):
        r = self.arb.up("wide")
        self.assertEqual(r["cards"], ["0", "1"])
        self.assertFalse(self.bg(0) or self.bg(1))
        self.assertEqual(self.host.env_of["cardshare-wide.service"]["CUDA_VISIBLE_DEVICES"], "0,1")
        self.arb.down("wide")
        self.assertTrue(self.bg(0) and self.bg(1))

    def test_host_memory_limit(self):
        self.cfg.host_mem_limit_gib = 150
        self.cfg.models["smallA"].host_mem_gib = 80
        self.cfg.models["bigA"].host_mem_gib = 100
        self.arb.up("smallA")                       # card 1, 80 GiB
        self.host.advance(100)
        r = self.arb.up("bigA")                     # card 0 is free, but 180 > 150
        self.assertEqual(r["evicted"], ["smallA"])
        self.assertTrue(self.bg(1))


class Lifecycle(Base):
    def test_gc_per_card_idle_returns_only_that_cards_background(self):
        self.cfg.models["smallA"].idle_minutes = 1
        self.cfg.models["bigA"].idle_minutes = 10
        self.arb.up("smallA")
        self.arb.up("bigA")
        self.host.advance(90)
        self.arb.gc()
        self.assertEqual(self.slots(), {"0": "bigA", "1": None})
        self.assertTrue(self.bg(1))
        self.assertFalse(self.bg(0))
        self.host.advance(600)
        self.arb.gc()
        self.assertTrue(self.bg(0))
        self.assertEqual(self.state()["models"], {})

    def test_gc_keeps_a_model_with_a_request_in_flight(self):
        self.cfg.models["smallA"].idle_minutes = 1
        self.arb.up("smallA")
        self.host.advance(120)
        self.lease("smallA", 1, last=self.host.now() - 120)
        self.arb.gc()
        self.assertIn("smallA", self.state()["models"])
        self.lease("smallA", 0)
        self.host.advance(61)
        self.arb.gc()
        self.assertNotIn("smallA", self.state()["models"])

    def test_client_class_idle_from_lease_overrides_model(self):
        self.cfg.models["smallA"].idle_minutes = 10
        self.arb.up("smallA")
        write_lease(self.cfg.run_dir, "smallA", 0, self.host.now(), self.host.now(), idle_minutes=2)
        self.host.advance(150)
        self.arb.gc()
        self.assertNotIn("smallA", self.state()["models"])

    def test_idle_zero_never_releases(self):
        self.cfg.models["smallA"].idle_minutes = 0
        self.arb.up("smallA")
        self.host.advance(10 ** 6)
        self.arb.gc()
        self.assertIn("smallA", self.state()["models"])

    def test_gc_cleans_up_a_dead_model(self):
        self.arb.up("smallA")
        self.host.units["cardshare-smallA.service"] = False
        self.arb.gc()
        self.assertEqual(self.state()["models"], {})
        self.assertTrue(self.bg(1))

    def test_gc_adopts_a_model_left_starting(self):
        self.arb.up("smallA")
        st = self.state()
        st["models"]["smallA"]["starting"] = True          # as if `up` was killed while waiting
        self.arb.save_state(st)
        self.arb.gc()
        self.assertFalse(self.state()["models"]["smallA"]["starting"])

    def test_gc_does_not_restart_a_background_stopped_by_hand(self):
        self.arb.background("stop", card="0")
        self.arb.gc()
        self.assertFalse(self.bg(0))

    def test_down_all_returns_every_background(self):
        self.arb.up("smallA")
        self.arb.up("bigA")
        self.arb.down()
        self.assertTrue(self.bg(0) and self.bg(1))
        self.assertEqual(self.slots(), {"0": None, "1": None})

    def test_down_one_returns_only_its_card(self):
        self.arb.up("smallA")
        self.arb.up("bigA")
        self.arb.down("smallA")
        self.assertTrue(self.bg(1))
        self.assertFalse(self.bg(0))

    def test_start_command_failure_rolls_back(self):
        self.host.start_fails.add("smallA")
        with self.assertRaises(StartFailed):
            self.arb.up("smallA")
        self.assertTrue(self.bg(1))
        self.assertEqual(self.state()["models"], {})

    def test_crash_while_loading_rolls_back(self):
        self.host.crash_on_load.add("bigA")
        with self.assertRaises(StartFailed) as cm:
            self.arb.up("bigA")
        self.assertIn("crashed", str(cm.exception))
        self.assertTrue(self.bg(0))
        self.assertEqual(self.state()["models"], {})

    def test_not_ready_in_time_rolls_back(self):
        self.cfg.models["smallA"].start_timeout = 30
        self.host.load_delay["smallA"] = 3600
        with self.assertRaises(StartFailed):
            self.arb.up("smallA")
        self.assertFalse(self.host.units["cardshare-smallA.service"])
        self.assertTrue(self.bg(1))

    def test_loading_takes_a_while(self):
        self.host.load_delay["smallA"] = 20
        t0 = self.host.now()
        self.arb.up("smallA")
        self.assertGreaterEqual(self.host.now() - t0, 20)

    def test_vram_not_freed_returns_background(self):
        self.host.stuck_vram.add("1")
        with self.assertRaises(ArbiterError) as cm:
            self.arb.up("smallA")
        self.assertIn("VRAM", str(cm.exception))
        self.assertTrue(self.bg(1))

    def test_no_gpu_query(self):
        self.cfg.gpu_query = None
        self.host.stuck_vram.add("1")
        self.arb.up("smallA")


class BudgetAndHold(Base):
    over = {"policy": {"evict_grace_sec": 0, "poll_sec": 1, "budget_minutes_per_hour": 5}}

    def test_budget(self):
        self.arb.up("smallA")
        self.host.advance(6 * 60)
        self.arb.down("smallA")
        with self.assertRaises(BudgetExhausted) as cm:
            self.arb.up("smallA")
        self.assertEqual(cm.exception.code, 76)
        self.arb.up("smallB", card="0")             # card 0's budget is untouched
        self.arb.up("smallA", force=True)
        self.arb.down("smallA")
        self.host.advance(3600)
        self.arb.up("smallA")                       # an hour later the budget is back

    def test_hold_keeps_freed_cards_empty(self):
        self.arb.up("smallA")
        self.arb.hold(True)
        self.arb.down("smallA")
        self.assertFalse(self.bg(1))
        self.assertTrue(self.arb.status()["cards"]["1"]["held"])
        self.arb.background("start")                # explicit start ignores the hold
        self.assertTrue(self.bg(1))
        self.arb.hold(False)
        self.assertFalse(self.arb.status()["hold"])

    def test_hold_one_card(self):
        self.arb.up("smallA")
        self.arb.up("bigA")
        self.arb.hold(True, card="1")
        self.arb.down()
        self.assertTrue(self.bg(0))
        self.assertFalse(self.bg(1))

    def test_gc_respects_hold(self):
        self.cfg.models["smallA"].idle_minutes = 1
        self.arb.up("smallA")
        self.arb.hold(True)
        self.host.advance(120)
        self.arb.gc()
        self.assertNotIn("smallA", self.state()["models"])
        self.assertFalse(self.bg(1))


class SharedBackground(Base):
    over = {"cards": {"0": {"background": {"unit": "bgall.service"}},
                      "1": {"background": {"unit": "bgall.service"}}}}

    def test_shared_background_returns_when_all_its_cards_are_free(self):
        self.host.units = {"bgall.service": True}
        self.arb.up("smallA")
        self.assertFalse(self.host.units["bgall.service"])
        self.arb.up("bigA")
        self.arb.down("smallA")
        self.assertFalse(self.host.units["bgall.service"])
        self.arb.down("bigA")
        self.assertTrue(self.host.units["bgall.service"])


class Sleep(Base):
    models = dict(default_models(), sleepy=model(9300, "big", [0], sleep=True, idle_minutes=1))

    def test_freed_sleeper_inserted_after_awake_model_is_not_gpu_owner(self):
        self.cfg.models["smallA"].gpus = ["0"]
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        self.arb.up("smallA")
        st = self.state()
        # Reproduce the misleading order: a retained sleeper is serialized last.
        st["models"]["sleepy"] = st["models"].pop("sleepy")
        self.arb.save_state(st)
        with open(self.cfg.state_file) as fh:
            saved = json.load(fh)
        self.assertEqual(saved["slots"]["0"], "smallA")
        self.assertEqual(saved["active"], "smallA")
        status = self.arb.status()
        self.assertEqual(status["cards"]["0"]["model"], "smallA")
        self.assertEqual(status["models"]["sleepy"]["state"], "asleep")

    def test_only_freeing_sleeper_has_no_gpu_owner_or_legacy_active(self):
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        with open(self.cfg.state_file) as fh:
            saved = json.load(fh)
        self.assertEqual(saved["slots"], {"0": None, "1": None})
        self.assertIsNone(saved["active"])
        status = self.arb.status()
        self.assertIsNone(status["cards"]["0"]["model"])
        self.assertEqual(status["cards"]["0"]["background"], "running")
        self.assertIn("sleepy", status["models"])

    def test_holding_sleeper_remains_gpu_owner_and_legacy_active(self):
        self.cfg.models["sleepy"].sleep["frees_card"] = False
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        with open(self.cfg.state_file) as fh:
            saved = json.load(fh)
        self.assertEqual(saved["slots"]["0"], "sleepy")
        self.assertEqual(saved["active"], "sleepy")
        status = self.arb.status()
        self.assertEqual(status["cards"]["0"]["model"], "sleepy")
        self.assertEqual(status["cards"]["0"]["background"], "stopped")

    def test_new_model_retains_sleeping_process_that_freed_its_card(self):
        self.cfg.models["smallA"].gpus = ["0"]
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        before = dict(self.state()["models"]["sleepy"])
        started = self.host.started_at["cardshare-sleepy.service"]
        calls = list(self.host.calls)
        result = self.arb.up("smallA")
        self.assertEqual(result["cards"], ["0"])
        self.assertEqual(result["evicted"], [])
        self.assertEqual(self.state()["models"]["sleepy"], before)
        self.assertTrue(self.host.units["cardshare-sleepy.service"])
        self.assertEqual(self.host.started_at["cardshare-sleepy.service"], started)
        self.assertNotIn(("stop", "cardshare-sleepy.service"), self.host.calls[len(calls):])

    def test_sleeping_model_that_holds_card_is_still_evicted(self):
        self.cfg.models["smallA"].gpus = ["0"]
        self.cfg.models["sleepy"].sleep["frees_card"] = False
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        result = self.arb.up("smallA")
        self.assertEqual(result["evicted"], ["sleepy"])
        self.assertNotIn("sleepy", self.state()["models"])
        self.assertFalse(self.host.units["cardshare-sleepy.service"])

    def test_sleeping_free_card_model_still_evicted_for_port_conflict(self):
        self.cfg.models["smallA"].gpus = ["0"]
        self.cfg.models["smallA"].ports = list(self.cfg.models["sleepy"].ports)
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        result = self.arb.up("smallA")
        self.assertEqual(result["evicted"], ["sleepy"])
        self.assertFalse(self.host.units["cardshare-sleepy.service"])

    def test_sleeping_free_card_model_still_counts_toward_host_memory_limit(self):
        self.cfg.host_mem_limit_gib = 15
        self.cfg.models["sleepy"].host_mem_gib = 10
        self.cfg.models["smallA"].host_mem_gib = 10
        self.cfg.models["smallA"].gpus = ["0"]
        self.arb.up("sleepy")
        self.arb.sleep_model("sleepy")
        result = self.arb.up("smallA")
        self.assertEqual(result["evicted"], ["sleepy"])
        self.assertFalse(self.host.units["cardshare-sleepy.service"])

    def test_manual_sleep_rejects_active_request_before_backend_call(self):
        self.arb.up("sleepy")
        self.lease("sleepy", 1)
        before = list(self.host.calls)
        with self.assertRaisesRegex(ArbiterError, "busy"):
            self.arb.sleep_model("sleepy")
        self.assertNotIn("http://127.0.0.1:9300/sleep", self.host.http_posts)
        self.assertEqual(self.host.calls, before)
        self.assertFalse(self.state()["models"]["sleepy"]["asleep"])
        self.assertTrue(self.host.units["cardshare-sleepy.service"])
        self.assertFalse(self.bg(0))

    def test_backend_busy_race_does_not_stop_model_or_start_background(self):
        self.arb.up("sleepy")
        original_http = self.host.http

        def raced_http(method, url, **kwargs):
            if method == "POST" and url.endswith("/sleep"):
                return 409, {"status": "busy"}
            return original_http(method, url, **kwargs)

        self.host.http = raced_http
        before = list(self.host.calls)
        with self.assertRaisesRegex(ArbiterError, "busy"):
            self.arb.sleep_model("sleepy")
        self.assertEqual(self.host.calls, before)
        self.assertTrue(self.host.units["cardshare-sleepy.service"])
        self.assertFalse(self.state()["models"]["sleepy"]["asleep"])
        self.assertFalse(self.bg(0))
        # The same admission race during idle GC must also defer sleep safely.
        self.host.advance(90)
        self.arb.gc()
        self.assertEqual(self.host.calls, before)
        self.assertTrue(self.host.units["cardshare-sleepy.service"])
        self.assertFalse(self.state()["models"]["sleepy"]["asleep"])
        self.assertFalse(self.bg(0))

    def test_idle_sleeps_then_wakes(self):
        self.arb.up("sleepy")
        self.assertFalse(self.bg(0))
        self.host.advance(90)
        self.arb.gc()
        rec = self.state()["models"]["sleepy"]
        self.assertTrue(rec["asleep"])
        self.assertTrue(self.bg(0))                 # the card went back to the background
        self.assertIn("http://127.0.0.1:9300/sleep", self.host.http_posts)
        self.assertTrue(self.host.units["cardshare-sleepy.service"])   # process kept
        r = self.arb.up("sleepy")
        self.assertTrue(r.get("woke"))
        self.assertFalse(self.bg(0))
        self.assertIn("http://127.0.0.1:9300/wake", self.host.http_posts)
        self.assertFalse(self.state()["models"]["sleepy"]["asleep"])

    def test_asleep_model_is_evicted_first(self):
        self.arb.up("sleepy")
        self.host.advance(90)
        self.arb.gc()
        r = self.arb.up("bigA")
        self.assertEqual(r["evicted"], ["sleepy"])
        self.assertFalse(self.host.units["cardshare-sleepy.service"])


class LegacyState(Base):
    def test_reads_single_slot_state(self):
        with open(self.cfg.state_file, "w") as fh:
            json.dump({"active": "smallA", "since": 900000.0, "idle_since": None,
                       "downtime": [[899000.0, None]]}, fh)
        self.host.units["bg1.service"] = False      # smallA ran where the background is stopped
        self.host.units["cardshare-smallA.service"] = True
        self.host.unit_cards["cardshare-smallA.service"] = ["1"]
        self.host.started_at["cardshare-smallA.service"] = 0
        st = self.state()
        self.assertEqual(st["models"]["smallA"]["cards"], ["1"])
        self.assertEqual(st["downtime"]["1"], [[899000.0, None]])
        self.arb.down("smallA")
        self.assertTrue(self.bg(1))
        with open(self.cfg.state_file) as fh:
            raw = json.load(fh)
        self.assertEqual(raw["version"], 2)
        self.assertIsNone(raw["active"])

    def test_missing_or_broken_state(self):
        self.assertEqual(self.state()["models"], {})
        with open(self.cfg.state_file, "w") as fh:
            fh.write("{broken")
        self.assertEqual(self.state()["models"], {})


class StatusAndFiles(Base):
    def test_status(self):
        self.arb.up("smallA")
        s = self.arb.status()
        self.assertEqual(s["cards"]["1"]["model"], "smallA")
        self.assertEqual(s["cards"]["1"]["background"], "stopped")
        self.assertEqual(s["cards"]["0"]["background"], "running")
        self.assertEqual(s["models"]["smallA"]["state"], "running")

    def test_state_file_mode(self):
        self.arb.up("smallA")
        self.assertEqual(os.stat(self.cfg.state_file).st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
