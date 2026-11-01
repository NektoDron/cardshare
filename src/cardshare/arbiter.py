"""The arbiter: the single owner of "which model is on which card".

Each card is a slot.  A slot holds one model (a model may also span several cards) or, when it is
free, the card's background workload (a miner, a batch job — anything that should use the card while
no model does).  The arbiter guarantees:

  * before a model starts, its cards are really free (VRAM checked, not assumed);
  * a model nobody uses gives its cards back by itself (per-model idle timeout, ``gc``);
  * on any failure on the way up the cards go back to their background workloads, never left empty;
  * an optional budget: background workloads lose at most N minutes per hour to models.

State lives in one JSON file, every changing command holds one lock.  All system access goes through
``Host`` (see host.py), so the logic here runs unchanged against fakes in the tests.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import tempfile

from .activity import read_lease
from .config import expand
from .host import Host

EX_NOROOM = 75          # EX_TEMPFAIL: the cards are busy with requests, retry later
EX_BUDGET = 76          # the background budget for this hour is used up


class ArbiterError(Exception):
    code = 1


class NoRoom(ArbiterError):
    code = EX_NOROOM

    def __init__(self, msg, blockers=()):
        super().__init__(msg)
        self.blockers = sorted(blockers)


class BudgetExhausted(ArbiterError):
    code = EX_BUDGET


class StartFailed(ArbiterError):
    code = 1


def hhmm(sec):
    sec = int(sec)
    if sec < 60:
        return "%d s" % sec
    return "%d:%02d h" % (sec // 3600, (sec % 3600) // 60) if sec >= 3600 else "%d min" % (sec // 60)


class Arbiter:
    def __init__(self, cfg, host=None, say=print):
        self.cfg = cfg
        self.host = host or Host(cfg)
        self.say = say
        self._cards = None

    # ── lock and state ───────────────────────────────────────────────────
    @contextlib.contextmanager
    def locked(self, wait=True):
        """One lock for every changing command, so two of them never read-modify-write the state at once.
        wait=False yields False when another command holds it (gc skips a round instead of queueing)."""
        d = os.path.dirname(self.cfg.lock_file)
        if d:
            os.makedirs(d, exist_ok=True)
        fd = os.open(self.cfg.lock_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not wait:
                    yield False
                    return
                self.say("waiting for another cardshare command to finish...")
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield True
        finally:
            os.close(fd)

    def load_state(self):
        try:
            with open(self.cfg.state_file, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            raw = {}
        return self._normalize(raw if isinstance(raw, dict) else {})

    def _normalize(self, raw):
        if raw.get("version") == 2:
            st = {"version": 2, "models": dict(raw.get("models") or {}),
                  "downtime": {str(k): list(v) for k, v in (raw.get("downtime") or {}).items()}}
            for rec in st["models"].values():
                rec["cards"] = [str(c) for c in rec.get("cards", [])]
            return st
        # Single-slot layout: {"active": name, "since", "idle_since", "downtime": [[start, end], ...]}.
        st = {"version": 2, "models": {}, "downtime": {}}
        name = raw.get("active")
        m = self.cfg.models.get(name) if name else None
        if m:
            cards = self._legacy_cards(m)
            since = raw.get("since") or self.host.now()
            st["models"][name] = {"cards": cards, "since": since,
                                  "last_used": raw.get("idle_since") or since, "asleep": False}
        legacy_dt = raw.get("downtime") or []
        if isinstance(legacy_dt, list) and legacy_dt:
            card = (st["models"][name]["cards"][0] if m else
                    next(iter(self.cfg.background), None) or (self.cards() or ["0"])[0])
            st["downtime"][card] = [list(iv) for iv in legacy_dt]
        return st

    def _legacy_cards(self, m):
        allowed = m.cards_for(self.cards())
        if m.gpus == "all":
            return allowed
        for c in allowed:                       # the card whose background is not running is the model's
            b = self.cfg.background.get(c)
            if b and not self.bg_running(b):
                return [c]
        return allowed[:1]

    def save_state(self, st):
        path = self.cfg.state_file
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            mode = 0o644                        # status is read by unprivileged users and the gateway
        out = dict(st)
        slots = {c: None for c in self.cards()}
        slots.update(self.occupied(st))
        out["slots"] = slots
        # read-only hint for older tools that knew a single "active" model
        out["active"] = next((slots[c] for c in sorted(slots) if slots[c]), None)
        fd, tmp = tempfile.mkstemp(dir=d, prefix="state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── cards ────────────────────────────────────────────────────────────
    def cards(self):
        if self._cards is None:
            if self.cfg.cards:
                self._cards = list(self.cfg.cards)
            else:
                used = self.host.gpu_used_mib()
                if not used:
                    raise ArbiterError("no cards in the config and the GPU query found none")
                self._cards = sorted(used, key=lambda c: (len(c), c))
        return self._cards

    def wait_free(self, cards, verbose=True):
        """Wait until every card in `cards` uses less than the threshold.  No GPU query: trust the stop."""
        if not self.cfg.gpu_query:
            return True
        deadline = self.host.now() + self.cfg.vram_wait_sec
        while True:
            used = self.host.gpu_used_mib()
            if used is None or any(c not in used for c in cards):
                if verbose:
                    self.say("GPU query failed or does not list cards %s" % ",".join(cards))
                return False
            busy = {c: used[c] for c in cards if used[c] >= self.cfg.vram_free_mib}
            if not busy:
                if verbose:
                    self.say("cards %s free: %s" % (",".join(cards), ", ".join(
                        "%s=%d MiB" % (c, used[c]) for c in cards)))
                return True
            if self.host.now() >= deadline:
                return False
            self.host.sleep(self.cfg.poll_sec)

    # ── background workloads ─────────────────────────────────────────────
    def bg_running(self, b):
        if b.unit:
            return self.host.unit_active(b.unit)
        return self.host.run(b.check, timeout=30)[0] == 0

    def held(self, card):
        h = self.cfg.hold_file
        return os.path.exists(h) or os.path.exists("%s.%s" % (h, card))

    def bg_stop(self, card, st):
        b = self.cfg.background.get(card)
        if not b or not self.bg_running(b):
            return True
        self.say("stopping background on card%s %s..." % ("s" if len(b.cards) > 1 else "", ",".join(b.cards)))
        if b.unit:
            self.host.unit_stop(b.unit, timeout=120)
        else:
            self.host.run(b.stop, timeout=120)
        now = self.host.now()
        for c in b.cards:
            ivs = st["downtime"].setdefault(c, [])
            if not ivs or ivs[-1][1] is not None:
                ivs.append([now, None])
        return not self.bg_running(b)

    def _close_downtime(self, st, cards):
        now = self.host.now()
        for c in cards:
            for iv in st["downtime"].get(c, []):
                if iv[1] is None:
                    iv[1] = now

    def occupied(self, st):
        """card -> model, for models that hold their card (running, starting, or asleep without freeing it)."""
        occ = {}
        for n, rec in st["models"].items():
            m = self.cfg.models.get(n)
            if rec.get("asleep") and (not m or not m.sleep or m.sleep["frees_card"]):
                continue
            for c in rec["cards"]:
                occ[c] = n
        return occ

    def sync_background(self, st, cards, exclude=(), force=False):
        """Return background workloads to those of `cards` that no model holds.  A held card (hold file)
        stays empty unless force.  Shared workloads start only when all their cards are free."""
        occ = self.occupied(st)
        done = set()
        ok = True
        for c in cards:
            b = self.cfg.background.get(c)
            if not b or b.id in done:
                continue
            done.add(b.id)
            if any(x in occ or x in exclude for x in b.cards):
                continue
            if not force and any(self.held(x) for x in b.cards):
                self.say("background on card %s is held (%s) — not returning it" % (
                    ",".join(b.cards), self.cfg.hold_file))
                continue
            self._close_downtime(st, b.cards)
            if self.bg_running(b):
                continue
            self.say("returning background to card%s %s..." % ("s" if len(b.cards) > 1 else "", ",".join(b.cards)))
            rc, out = (self.host.unit_start(b.unit) if b.unit else self.host.run(b.start, timeout=180))
            if rc != 0:
                self.say("COULD NOT start background on card %s: %s" % (",".join(b.cards), out[-300:]))
                ok = False
        return ok

    def downtime_last_hour(self, st, card):
        now = self.host.now()
        total = 0.0
        for start, end in st["downtime"].get(card, []):
            overlap = min(end or now, now) - max(start, now - 3600)
            if overlap > 0:
                total += overlap
        return total

    def prune_downtime(self, st):
        now = self.host.now()
        for c in list(st["downtime"]):
            st["downtime"][c] = [iv for iv in st["downtime"][c] if (iv[1] or now) > now - 7200][-200:]

    # ── one model's process ──────────────────────────────────────────────
    def urls(self, m, cards):
        return {k: expand(getattr(m, k), m.name, cards) if getattr(m, k) else None
                for k in ("health_url", "api_url", "busy_url")}

    def _unit(self, m, cards):
        if m.unit:
            return expand(m.unit, m.name, cards)
        if m.exec:
            return m.transient_unit(self.cfg.unit_prefix)
        return None

    def running(self, m, cards):
        """The model's process is alive (not necessarily ready)."""
        unit = self._unit(m, cards)
        if unit:
            return self.host.unit_active(unit)
        if m.check:
            return self.host.run(expand(m.check, m.name, cards), timeout=30,
                                 env=expand(m.env, m.name, cards))[0] == 0
        return self.ready(m, cards)

    def ready(self, m, cards):
        status, body = self.host.http("GET", self.urls(m, cards)["health_url"], timeout=3)
        if status != 200:
            return False
        return not isinstance(body, dict) or body.get("status", "ok") in ("ok", "ready")

    def start(self, m, cards):
        env = expand(m.env, m.name, cards)
        if m.unit:
            return self.host.unit_start(expand(m.unit, m.name, cards), timeout=60)
        if m.exec:
            return self.host.transient_start(m.transient_unit(self.cfg.unit_prefix),
                                             expand(m.exec, m.name, cards), env, m.user,
                                             expand(m.properties, m.name, cards))
        return self.host.run(expand(m.start, m.name, cards), timeout=m.start_timeout, env=env)

    def stop(self, m, cards):
        unit = self._unit(m, cards)
        if unit:
            return self.host.unit_stop(unit, timeout=m.stop_timeout)
        return self.host.run(expand(m.stop, m.name, cards), timeout=m.stop_timeout,
                             env=expand(m.env, m.name, cards))

    def last_used(self, name, rec):
        _, lease_last, _ = read_lease(self.cfg.run_dir, name, self.host.now(), self.cfg.lease_stale_sec)
        return max(rec.get("last_used") or 0, rec.get("since") or 0, lease_last)

    def idle_limit(self, name, m):
        """Seconds of idleness after which the model goes (0: never).  The client class of the last
        request through the gateway, when it has its own timeout, wins over the model's."""
        _, _, cls_idle = read_lease(self.cfg.run_dir, name, self.host.now(), float("inf"))
        return (cls_idle if cls_idle is not None else m.idle_minutes) * 60

    def busy(self, name, m, rec):
        """A request is being served: the gateway's lease, or the server's own /slots-like answer."""
        inflight, _, _ = read_lease(self.cfg.run_dir, name, self.host.now(), self.cfg.lease_stale_sec)
        if inflight > 0:
            return True
        url = self.urls(m, rec["cards"])["busy_url"] if m else None
        if not url or rec.get("asleep"):
            return False
        status, body = self.host.http("GET", url, timeout=3)
        if status != 200:
            return False
        if isinstance(body, list):              # llama.cpp /slots
            return any(isinstance(x, dict) and x.get("is_processing") for x in body)
        if isinstance(body, dict):
            return bool(body.get("is_processing") or body.get("busy") or body.get("slots_processing"))
        return False

    # ── placement ────────────────────────────────────────────────────────
    def plan(self, st, name, m, card=None, preempt=False, no_switch=False):
        """Cards for `name` and the models to evict for it.  Raises NoRoom if every option would evict
        a model that is serving a request (or was used within evict_grace_sec)."""
        cards = self.cards()
        allowed = m.cards_for(cards)
        bad = [c for c in allowed if c not in cards]
        if bad:
            raise ArbiterError("%s wants card %s, which is not in the config" % (name, ",".join(bad)))
        if m.gpus == "all":
            cands = [allowed]
            if card is not None and card not in allowed:
                raise ArbiterError("no card %s" % card)
        else:
            cands = [[c] for c in allowed]
            if card is not None:
                if card not in allowed:
                    raise ArbiterError("%s may run only on card %s" % (name, ",".join(allowed)))
                cands = [[card]]

        now = self.host.now()
        others = {n: r for n, r in st["models"].items() if n != name}
        occ = {c: n for c, n in self.occupied(st).items() if n != name}
        model_of = lambda n: self.cfg.models.get(n)
        last = {n: self.last_used(n, r) for n, r in others.items()}
        busy_cache = {}

        def evictable(n):
            if preempt:
                return True
            if n not in busy_cache:
                r = others[n]
                if r.get("starting"):
                    busy_cache[n] = False
                elif r.get("asleep"):
                    busy_cache[n] = True
                else:
                    busy_cache[n] = (not self.busy(n, model_of(n), r)
                                     and now - last[n] >= self.cfg.evict_grace_sec)
            return busy_cache[n]

        def lru(names):
            # asleep first, then the one used longest ago
            return sorted(names, key=lambda n: (not others[n].get("asleep"), last[n]))

        port_conflicts = {n for n in others if model_of(n) and set(model_of(n).ports) & set(m.ports)}
        bigs = [n for n in others if model_of(n) and model_of(n).klass == "big"]
        best, blockers = None, set()
        for i, cand in enumerate(cands):
            victims = {occ[c] for c in cand if c in occ} | port_conflicts
            blocked = False
            if m.klass == "big" and self.cfg.max_big > 0:
                rest = [b for b in bigs if b not in victims]
                excess = len(rest) - (self.cfg.max_big - 1)
                if excess > 0:
                    free_bigs = [b for b in lru(rest) if evictable(b)][:excess]
                    if len(free_bigs) < excess:
                        blockers.update(b for b in rest if not evictable(b))
                        blocked = True
                    victims |= set(free_bigs)
            limit = self.cfg.host_mem_limit_gib
            if limit and not blocked:
                keep = [n for n in others if n not in victims]
                total = m.host_mem_gib + sum(model_of(n).host_mem_gib for n in keep if model_of(n))
                for n in lru([n for n in keep if evictable(n)]):
                    if total <= limit:
                        break
                    victims.add(n)
                    total -= model_of(n).host_mem_gib if model_of(n) else 0
                if total > limit:
                    blockers.update(n for n in keep if n not in victims and not evictable(n))
                    blocked = True
            stuck = [v for v in victims if not evictable(v)]
            if stuck:
                blockers.update(stuck)
                blocked = True
            if victims and no_switch:
                blockers.update(victims)
                blocked = True
            if blocked:
                continue
            awake = [v for v in victims if not others[v].get("asleep")]
            key = (len(awake), len(victims), max([last[v] for v in victims] or [0]), i)
            if best is None or key < best[0]:
                best = (key, cand, victims)
        if best is None:
            where = ", ".join("%s (card %s)" % (b, ",".join(others[b]["cards"])) for b in sorted(blockers))
            raise NoRoom("no room for %s: %s %s" % (
                name, "would have to replace" if no_switch else "busy:", where or "nothing fits"), blockers)
        return best[1], lru(best[2])

    # ── operations ───────────────────────────────────────────────────────
    def _model(self, name):
        m = self.cfg.models.get(name)
        if not m:
            raise ArbiterError("no such model: %s. Known: %s" % (name, ", ".join(sorted(self.cfg.models)) or "-"))
        return m

    def _result(self, name, m, rec, **extra):
        out = {"ok": True, "model": name, "cards": rec["cards"], "asleep": bool(rec.get("asleep"))}
        out.update(self.urls(m, rec["cards"]))
        out.update(extra)
        return out

    def _check_budget(self, st, cards, force):
        budget = self.cfg.budget_min * 60
        if not budget or force:
            return
        for c in cards:
            b = self.cfg.background.get(c)
            if not b or not self.bg_running(b):
                continue
            spent = self.downtime_last_hour(st, c)
            if spent >= budget:
                raise BudgetExhausted(
                    "background budget for card %s is used up for this hour: %s of %d min. "
                    "Use --force if it is worth it." % (c, hhmm(spent), budget // 60))

    def release(self, st, name, exclude=(), verbose=True):
        """Stop a model, wait for its cards, forget it, give its cards back (except `exclude`)."""
        rec = st["models"].get(name)
        if not rec:
            return
        m = self.cfg.models.get(name)
        cards = rec["cards"]
        if m:
            if verbose:
                self.say("stopping %s..." % name)
            self.stop(m, cards)
            # an asleep model that freed its card has the background there already: nothing to wait for
            if not (rec.get("asleep") and m.sleep and m.sleep["frees_card"]):
                self.wait_free(cards, verbose=verbose)
        st["models"].pop(name, None)
        self.sync_background(st, cards, exclude=exclude)
        self.prune_downtime(st)

    def _fail(self, st, name, m, cards, why):
        self.say("%s: %s — rolling back" % (name, why))
        self.stop(m, cards)
        self.wait_free(cards, verbose=False)
        st["models"].pop(name, None)
        self.sync_background(st, cards)
        self.save_state(st)
        raise StartFailed("%s %s; background returned to card %s" % (name, why, ",".join(cards)))

    def _wait_ready(self, st, name, m, cards):
        deadline = self.host.now() + m.start_timeout
        while self.host.now() < deadline:
            if not self.running(m, cards):
                self._fail(st, name, m, cards, "crashed while loading")
            if self.ready(m, cards):
                return
            self.host.sleep(self.cfg.poll_sec)
        self._fail(st, name, m, cards, "was not ready within %d s" % m.start_timeout)

    def up(self, name, card=None, force=False, preempt=False, no_switch=False):
        m = self._model(name)
        card = None if card in (None, "auto") else str(card)
        with self.locked():
            st = self.load_state()
            rec = st["models"].get(name)
            if rec:
                if rec.get("starting") or not self.running(m, rec["cards"]):
                    self.say("%s is not running any more — cleaning up" % name)
                    self.release(st, name, verbose=False)
                    self.save_state(st)
                elif rec.get("asleep"):
                    return self._wake(st, name, m, rec, force)
                elif card is not None and card not in rec["cards"]:
                    self.say("%s is on card %s — moving it to card %s" % (name, ",".join(rec["cards"]), card))
                    self.release(st, name)
                    self.save_state(st)
                else:
                    rec["last_used"] = self.host.now()
                    self.save_state(st)
                    self.say("%s is already up on card %s" % (name, ",".join(rec["cards"])))
                    return self._result(name, m, rec, already=True)

            cards, victims = self.plan(st, name, m, card, preempt, no_switch)
            if self.running(m, cards):
                # a process left over outside the state (e.g. the arbiter was killed mid-way)
                self.say("%s is running outside the state — stopping it first" % name)
                self.stop(m, cards)
            self._check_budget(st, cards, force)
            for v in victims:
                self.say("evicting %s from card %s" % (v, ",".join(st["models"][v]["cards"])))
                self.release(st, v, exclude=cards)
                self.save_state(st)

            for c in cards:
                if not self.bg_stop(c, st):
                    self.sync_background(st, cards)
                    self.save_state(st)
                    raise ArbiterError("background on card %s did not stop — not starting %s" % (c, name))
            self.save_state(st)
            if not self.wait_free(cards):
                self.say("cards %s did not free up — returning background" % ",".join(cards))
                self.sync_background(st, cards)
                self.save_state(st)
                raise ArbiterError("VRAM on card %s did not free up" % ",".join(cards))

            now = self.host.now()
            st["models"][name] = {"cards": cards, "since": now, "last_used": now,
                                  "starting": True, "asleep": False}
            self.save_state(st)
            self.say("starting %s on card %s..." % (name, ",".join(cards)))
            rc, out = self.start(m, cards)
            if rc != 0:
                self._fail(st, name, m, cards, "did not start: %s" % out[-600:])
            self._wait_ready(st, name, m, cards)
            now = self.host.now()
            rec = st["models"][name]
            rec.update(starting=False, since=now, last_used=now)
            self.prune_downtime(st)
            self.save_state(st)
            self.say("%s ready on card %s (%s)" % (name, ",".join(cards), self.urls(m, cards)["api_url"]))
            return self._result(name, m, rec, already=False, evicted=victims)

    def _wake(self, st, name, m, rec, force):
        cards = rec["cards"]
        if m.sleep["frees_card"]:
            self._check_budget(st, cards, force)
            occ = self.occupied(st)
            clash = sorted({occ[c] for c in cards if c in occ and occ[c] != name})
            if clash:
                raise ArbiterError("%s cannot wake: card taken by %s" % (name, ", ".join(clash)))
            for c in cards:
                if not self.bg_stop(c, st):
                    self.sync_background(st, cards)
                    self.save_state(st)
                    raise ArbiterError("background on card %s did not stop" % c)
            self.save_state(st)
            if not self.wait_free(cards):
                self.sync_background(st, cards)
                self.save_state(st)
                raise ArbiterError("VRAM on card %s did not free up" % ",".join(cards))
        self.say("waking %s on card %s..." % (name, ",".join(cards)))
        rec["asleep"] = False
        rec["starting"] = True
        self.save_state(st)
        status, _ = self.host.http("POST", m.sleep["wake_url"], timeout=m.sleep["timeout_sec"], body={})
        if status != 200:
            self._fail(st, name, m, cards, "did not wake (HTTP %s)" % status)
        self._wait_ready(st, name, m, cards)
        now = self.host.now()
        rec.update(starting=False, last_used=now)
        rec.pop("slept_at", None)
        self.save_state(st)
        return self._result(name, m, rec, already=False, woke=True)

    def _sleep(self, st, name, m, rec):
        self.say("%s idle — putting it to sleep" % name)
        status, _ = self.host.http("POST", m.sleep["sleep_url"], timeout=m.sleep["timeout_sec"], body={})
        if status == 409:
            self.say("%s became busy — leaving it awake" % name)
            return False
        if status != 200:
            self.say("%s did not go to sleep (HTTP %s) — stopping it" % (name, status))
            return self.release(st, name)
        rec["asleep"] = True
        rec["slept_at"] = self.host.now()
        if m.sleep["frees_card"]:
            if not self.wait_free(rec["cards"]):
                self.say("%s asleep but its card is not free — stopping it" % name)
                return self.release(st, name)
            self.sync_background(st, rec["cards"])

    def down(self, name=None, card=None):
        with self.locked():
            st = self.load_state()
            if name:
                m = self._model(name)
                if name in st["models"]:
                    self.release(st, name)
                else:
                    cards = m.cards_for(self.cards())
                    if self.running(m, cards):
                        self.say("%s is running outside the state — stopping it" % name)
                        self.stop(m, cards)
                        self.wait_free(cards)
                        self.sync_background(st, cards)
                    else:
                        self.say("%s is not up" % name)
            else:
                targets = [n for n, r in st["models"].items() if card is None or str(card) in r["cards"]]
                if not targets:
                    self.say("no models on %s" % ("card " + str(card) if card is not None else "the cards"))
                for n in targets:
                    self.release(st, n)
                self.sync_background(st, [str(card)] if card is not None else self.cards())
            self.prune_downtime(st)
            self.save_state(st)
            return {"ok": True}

    def sleep_model(self, name):
        m = self._model(name)
        if not m.sleep:
            raise ArbiterError("%s has no sleep mode in the config" % name)
        with self.locked():
            st = self.load_state()
            rec = st["models"].get(name)
            if not rec or rec.get("asleep"):
                self.say("%s is %s" % (name, "asleep already" if rec else "not up"))
                return {"ok": True}
            if self.busy(name, m, rec):
                raise ArbiterError("%s is busy; cannot sleep while serving a request" % name)
            if self._sleep(st, name, m, rec) is False:
                raise ArbiterError("%s became busy; left awake" % name)
            self.save_state(st)
            return {"ok": True}

    def touch(self, name):
        """Count as use now: resets the idle timer (for clients that talk to the backend directly)."""
        self._model(name)
        with self.locked():
            st = self.load_state()
            rec = st["models"].get(name)
            if not rec:
                raise ArbiterError("%s is not up" % name)
            rec["last_used"] = self.host.now()
            self.save_state(st)
            return {"ok": True}

    def gc(self):
        """Idle models give their cards back.  Run by a timer (every minute) and/or by the gateway."""
        with self.locked(wait=False) as got:
            if not got:
                self.say("gc: another cardshare command is running — skipping")
                return {"ok": True, "skipped": True}
            st = self.load_state()
            now = self.host.now()
            for name, rec in list(st["models"].items()):
                m = self.cfg.models.get(name)
                if not m:
                    self.say("%s is not in the config any more — forgetting it" % name)
                    st["models"].pop(name)
                    self.sync_background(st, rec["cards"])
                    continue
                if not self.running(m, rec["cards"]):
                    self.say("%s is not running any more — cleaning up" % name)
                    self.release(st, name, verbose=False)
                    continue
                if rec.get("starting"):
                    # left by an `up` that was killed half-way: adopt the model if it came up after all
                    if self.ready(m, rec["cards"]):
                        self.say("%s came up after an interrupted start — adopting it" % name)
                        rec.update(starting=False, asleep=False, last_used=now)
                    continue
                if rec.get("asleep"):
                    rel = m.sleep["release_minutes"] if m.sleep else 0
                    if rel and now - rec.get("slept_at", now) >= rel * 60:
                        self.say("%s asleep for %s — stopping it" % (name, hhmm(now - rec["slept_at"])))
                        self.release(st, name)
                    continue
                if self.busy(name, m, rec):
                    rec["last_used"] = now
                    continue
                idle = now - self.last_used(name, rec)
                limit = self.idle_limit(name, m)
                if limit and idle >= limit:
                    if m.sleep:
                        self._sleep(st, name, m, rec)
                    else:
                        self.say("%s idle for %s — giving card %s back" % (name, hhmm(idle), ",".join(rec["cards"])))
                        self.release(st, name)
            self.prune_downtime(st)
            self.save_state(st)
            return {"ok": True}

    def background(self, action, card=None):
        """Direct control: stop/start background workloads (start ignores the hold file)."""
        with self.locked():
            st = self.load_state()
            cards = [str(card)] if card is not None else self.cards()
            ok = True
            if action == "stop":
                for c in cards:
                    ok = self.bg_stop(c, st) and ok
            else:
                occ = self.occupied(st)
                for c in cards:
                    if c in occ:
                        self.say("card %s holds %s — not starting its background" % (c, occ[c]))
                ok = self.sync_background(st, [c for c in cards if c not in occ], force=True)
            self.save_state(st)
            return {"ok": ok}

    def hold(self, on, card=None):
        """Hold file on: freed cards stay empty (no background) — for series of tests.  Off: nothing is
        restarted by itself; the next release (or `background start`) returns the workloads."""
        path = self.cfg.hold_file if card is None else "%s.%s" % (self.cfg.hold_file, card)
        if on:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass
            self.say("hold on: %s" % path)
        else:
            try:
                os.unlink(path)
                self.say("hold off: %s removed" % path)
            except FileNotFoundError:
                self.say("no hold at %s" % path)
        return {"ok": True}

    # ── reading ──────────────────────────────────────────────────────────
    def status(self, st=None, probe=True):
        """What is where.  probe=False reads only files (cheap; for the gateway's model list)."""
        st = st or self.load_state()
        now = self.host.now()
        cards = {}
        for c in self.cards():
            b = self.cfg.background.get(c)
            cards[c] = {"model": None, "background": None if not b else
                        (("running" if self.bg_running(b) else "stopped") if probe else "unknown"),
                        "held": self.held(c),
                        "background_downtime_last_hour_min": round(self.downtime_last_hour(st, c) / 60, 1)}
        models = {}
        for c, n in self.occupied(st).items():
            if c in cards:
                cards[c]["model"] = n
        for n, rec in st["models"].items():
            m = self.cfg.models.get(n)
            inflight, _, _ = read_lease(self.cfg.run_dir, n, now, self.cfg.lease_stale_sec)
            state = "starting" if rec.get("starting") else "asleep" if rec.get("asleep") else "running"
            models[n] = {"state": state, "cards": rec["cards"], "up_sec": int(now - (rec.get("since") or now)),
                         "idle_sec": int(now - self.last_used(n, rec)), "inflight": inflight,
                         "idle_minutes": self.idle_limit(n, m) / 60 if m else None}
        return {"cards": cards, "models": models, "budget_minutes_per_hour": self.cfg.budget_min,
                "hold": os.path.exists(self.cfg.hold_file)}
