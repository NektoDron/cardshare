"""Configuration: one file (JSON or TOML) shared by the arbiter and the gateway.

Lookup order for the file: ``--config`` flag, ``$CARDSHARE_CONFIG``, ``/etc/cardshare/config.json``,
``/etc/cardshare/config.toml``, ``$XDG_CONFIG_HOME/cardshare/config.json`` (``~/.config``).

Everything machine-specific (cards, background workloads, models, ports, units) lives in the file;
the code only knows the shapes.  Paths of the state, the lock and the runtime directory can also be
set from the environment (``CARDSHARE_STATE``, ``CARDSHARE_LOCK``, ``CARDSHARE_RUN_DIR``), which is
how the tests run the arbiter without root.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from urllib.parse import urlsplit

DEFAULT_STATE = "/var/lib/cardshare/state.json"
DEFAULT_LOCK = "/run/cardshare.lock"
DEFAULT_RUN_DIR = "/run/cardshare"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100

DEFAULT_GPU_QUERY = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]

# API families a backend may speak; the gateway refuses a route the model does not list.
API_KINDS = ("openai", "anthropic", "responses")


class ConfigError(Exception):
    pass


def candidate_paths(explicit=None):
    if explicit:
        return [explicit]
    if os.environ.get("CARDSHARE_CONFIG"):
        return [os.environ["CARDSHARE_CONFIG"]]
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return ["/etc/cardshare/config.json", "/etc/cardshare/config.toml",
            os.path.join(xdg, "cardshare", "config.json"), os.path.join(xdg, "cardshare", "config.toml")]


def find_config(explicit=None):
    paths = candidate_paths(explicit)
    for p in paths:
        if os.path.exists(p):
            return p
    raise ConfigError("no config file found (looked at: %s)" % ", ".join(paths))


def _read(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if path.endswith(".toml"):
        try:
            import tomllib
        except ImportError as exc:                 # Python < 3.11
            raise ConfigError("TOML config needs Python 3.11+ (tomllib); use JSON") from exc
        return tomllib.loads(raw.decode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def _env_float(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError as exc:
        raise ConfigError("%s=%r is not a number" % (name, v)) from exc


def _cmd(value, what):
    """A command is a list of strings (no shell); a string is split on whitespace for convenience."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list) and all(isinstance(x, (str, int, float)) for x in value):
        return [str(x) for x in value]
    raise ConfigError("%s: expected a command as a list of strings" % what)


def _port_of(url):
    try:
        u = urlsplit(url)
        return u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        return None


def expand(value, name, cards):
    """Fill {name}, {card} (the first card) and {cards} ("0,1") in a string, a list or a dict.

    Only these three tokens are replaced, so shell-like ${VAR} in commands stays as it is."""
    if isinstance(value, str):
        cards = [str(c) for c in (cards or [])]
        return (value.replace("{name}", name)
                .replace("{cards}", ",".join(cards))
                .replace("{card}", cards[0] if cards else ""))
    if isinstance(value, list):
        return [expand(v, name, cards) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, name, cards) for k, v in value.items()}
    return value


class Background:
    """Workload a card runs when no model holds it (a miner, a batch job...). Optional per card.

    Either a systemd unit or start/stop/check commands.  Cards that list the same workload share it:
    it runs only while all of them are free."""

    def __init__(self, card, raw):
        if not isinstance(raw, dict):
            raise ConfigError("cards.%s.background: expected an object" % card)
        self.unit = raw.get("unit")
        self.start = _cmd(raw.get("start"), "cards.%s.background.start" % card)
        self.stop = _cmd(raw.get("stop"), "cards.%s.background.stop" % card)
        self.check = _cmd(raw.get("check"), "cards.%s.background.check" % card)
        if not self.unit and not (self.start and self.stop and self.check):
            raise ConfigError("cards.%s.background: give a unit, or start, stop and check commands" % card)
        self.id = str(raw.get("id") or self.unit or json.dumps(self.start))
        self.cards = [card]


class Model:
    def __init__(self, name, raw, defaults):
        if not isinstance(raw, dict):
            raise ConfigError("models.%s: expected an object" % name)
        self.name = name
        self.raw = raw
        g = lambda k, d=None: raw.get(k, defaults.get(k, d))
        self.desc = str(raw.get("desc", ""))
        self.klass = str(g("class", "small"))
        if self.klass not in ("big", "small"):
            raise ConfigError("models.%s.class: 'big' or 'small'" % name)
        gpus = raw.get("gpus", defaults.get("gpus", "all"))
        if gpus in ("all", "both"):
            self.gpus = "all"
        elif isinstance(gpus, (list, tuple)) and gpus:
            self.gpus = [str(x) for x in gpus]
        elif isinstance(gpus, (int, str)) and str(gpus).isdigit():
            self.gpus = [str(gpus)]
        else:
            raise ConfigError("models.%s.gpus: a list of cards in order of preference, or \"all\"" % name)
        self.unit = raw.get("unit")
        self.exec = _cmd(raw.get("exec"), "models.%s.exec" % name)
        self.env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
        self.user = raw.get("user")
        self.properties = [str(x) for x in raw.get("properties", [])]
        self.start = _cmd(raw.get("start"), "models.%s.start" % name)
        self.stop = _cmd(raw.get("stop"), "models.%s.stop" % name)
        self.check = _cmd(raw.get("check"), "models.%s.check" % name)
        kinds = sum(bool(x) for x in (self.unit, self.exec, self.start))
        if kinds != 1:
            raise ConfigError("models.%s: give exactly one of unit, exec, start (+stop)" % name)
        if self.start and not self.stop:
            raise ConfigError("models.%s: start needs stop" % name)
        self.health_url = raw.get("health_url")
        self.api_url = raw.get("api_url")
        if not self.health_url:
            raise ConfigError("models.%s.health_url is required" % name)
        if not self.api_url:
            self.api_url = re.sub(r"/health/?$", "", self.health_url)
        self.busy_url = raw.get("busy_url")
        self.ports = sorted({int(p) for p in raw.get("ports", [])}
                            | {p for p in (_port_of(self.health_url), _port_of(self.api_url)) if p})
        self.idle_minutes = float(g("idle_minutes", 3))
        self.start_timeout = float(g("start_timeout_sec", 900))
        self.stop_timeout = float(g("stop_timeout_sec", 300))
        self.host_mem_gib = float(g("host_mem_gib", 0))
        self.aliases = [str(a) for a in raw.get("aliases", [])]
        self.upstream_model = raw.get("upstream_model")
        self.upstream_headers = {str(k): str(v) for k, v in (raw.get("upstream_headers") or {}).items()}
        api = raw.get("api", defaults.get("api", list(API_KINDS)))
        bad = [a for a in api if a not in API_KINDS]
        if bad:
            raise ConfigError("models.%s.api: unknown %s (known: %s)" % (name, bad, ", ".join(API_KINDS)))
        self.api = list(api)
        self.requires = [str(p) for p in raw.get("requires", [])]
        self.hidden = bool(raw.get("hidden", False))
        self.listed = bool(raw.get("listed", True))
        self.context_window = raw.get("context_window")
        if "context_window" in raw and (type(self.context_window) is not int or self.context_window <= 0):
            raise ConfigError("models.%s.context_window: expected a positive integer" % name)
        # Sleep instead of stop on idle: the server frees the card but keeps the process (and its
        # host memory), so waking is much faster than a cold start.  Off by default.
        sl = raw.get("sleep", defaults.get("sleep", False))
        if sl is True:
            sl = {}
        if sl and not isinstance(sl, dict):
            raise ConfigError("models.%s.sleep: true/false or an object" % name)
        self.sleep = None
        if sl is not False and sl is not None:
            base = self.api_url.rstrip("/")
            self.sleep = {
                "sleep_url": sl.get("sleep_url", base + "/sleep"),
                "wake_url": sl.get("wake_url", base + "/wake"),
                "timeout_sec": float(sl.get("timeout_sec", 120)),
                "frees_card": bool(sl.get("frees_card", True)),
                "release_minutes": float(sl.get("release_minutes", 0)),
            }

    def cards_for(self, all_cards):
        return list(all_cards) if self.gpus == "all" else list(self.gpus)

    def missing(self):
        return [p for p in self.requires if not os.path.exists(p)]

    def transient_unit(self, prefix):
        return prefix + re.sub(r"[^A-Za-z0-9:_.-]", "_", self.name) + ".service"


class Config:
    def __init__(self, raw, path=None):
        if not isinstance(raw, dict):
            raise ConfigError("config: expected an object at the top level")
        self.path = path
        self.raw = raw
        self.state_file = os.environ.get("CARDSHARE_STATE") or raw.get("state_file") or DEFAULT_STATE
        self.lock_file = os.environ.get("CARDSHARE_LOCK") or raw.get("lock_file") or DEFAULT_LOCK
        self.run_dir = os.environ.get("CARDSHARE_RUN_DIR") or raw.get("run_dir") or DEFAULT_RUN_DIR
        self.hold_file = raw.get("hold_file") or os.path.join(self.run_dir, "hold")

        cmds = raw.get("commands", {})
        self.systemctl = _cmd(cmds.get("systemctl", ["systemctl"]), "commands.systemctl")
        self.systemd_run = _cmd(cmds.get("systemd_run", ["systemd-run"]), "commands.systemd_run")
        self.unit_prefix = str(raw.get("unit_prefix", "cardshare-"))

        gpu = raw.get("gpu", {})
        q = gpu.get("query", DEFAULT_GPU_QUERY)
        self.gpu_query = _cmd(q, "gpu.query") if q else None       # null: no VRAM checks
        self.vram_free_mib = float(gpu.get("free_threshold_mib", 2000))
        self.vram_wait_sec = float(gpu.get("free_wait_sec", 120))

        pol = raw.get("policy", {})
        self.max_big = int(pol.get("max_big", 1))
        self.budget_min = float(pol.get("budget_minutes_per_hour", 0))
        self.host_mem_limit_gib = float(pol.get("host_mem_limit_gib", 0))
        self.evict_grace_sec = float(pol.get("evict_grace_sec", 15))
        self.lease_stale_sec = float(pol.get("lease_stale_sec", 90))
        self.poll_sec = float(pol.get("poll_sec", 2))

        defaults = dict(raw.get("defaults", {}))
        if os.environ.get("CARDSHARE_IDLE_MINUTES"):
            defaults["idle_minutes"] = _env_float("CARDSHARE_IDLE_MINUTES", 3)
        self.defaults = defaults

        cards = raw.get("cards")
        self.cards = []
        self.background = {}                       # card -> Background
        if isinstance(cards, dict):
            for c, cr in cards.items():
                c = str(c)
                self.cards.append(c)
                if cr and cr.get("background"):
                    self.background[c] = Background(c, cr["background"])
        elif isinstance(cards, list):
            self.cards = [str(c) for c in cards]
        elif cards is not None:
            raise ConfigError("cards: a list of card ids or an object card -> settings")
        self.cards_discover = not self.cards
        # cards that share one background workload
        by_id = {}
        for c, b in self.background.items():
            by_id.setdefault(b.id, []).append(c)
        for c, b in self.background.items():
            b.cards = by_id[b.id]

        self.models = {}
        for name, mr in (raw.get("models") or {}).items():
            self.models[str(name)] = Model(str(name), mr, defaults)

        self.aliases = {}                          # alias -> (model, upstream model or None)
        for name, m in self.models.items():
            for a in m.aliases:
                self._alias(a, name, None)
        for a, target in (raw.get("aliases") or {}).items():
            if isinstance(target, str):
                self._alias(a, target, None)
            elif isinstance(target, dict) and target.get("model"):
                self._alias(a, target["model"], target.get("upstream_model"))
            else:
                raise ConfigError("aliases.%s: a model name or {\"model\": ..., \"upstream_model\": ...}" % a)

        self.gateway = Gateway(raw.get("gateway", {}))

    def _alias(self, alias, model, upstream):
        if model not in self.models:
            raise ConfigError("alias %s points to unknown model %s" % (alias, model))
        if alias in self.models and alias != model:
            raise ConfigError("alias %s clashes with a model of that name" % alias)
        if alias in self.aliases and self.aliases[alias][0] != model:
            raise ConfigError("alias %s points to two models" % alias)
        self.aliases[alias] = (model, upstream)

    def resolve(self, requested):
        """Model name or alias -> (model name, upstream model name or None); None if unknown."""
        if requested in self.models:
            return requested, None
        return self.aliases.get(requested)

    def all_cards(self, discovered=None):
        if self.cards:
            return list(self.cards)
        return [str(c) for c in (discovered or [])]


class Gateway:
    def __init__(self, raw):
        # Address: --host/--port (applied in gateway.run) > $CARDSHARE_HOST/$CARDSHARE_PORT > config
        # "host"/"port" > 127.0.0.1:8100.  Loopback by default: exposing models is an explicit choice.
        self.host = str(os.environ.get("CARDSHARE_HOST") or raw.get("host") or DEFAULT_HOST).strip("[]")
        port = os.environ.get("CARDSHARE_PORT") or raw.get("port") or DEFAULT_PORT
        try:
            self.port = int(port)
        except (TypeError, ValueError) as exc:
            raise ConfigError("gateway port %r is not a number" % (port,)) from exc
        if not 0 < self.port < 65536:
            raise ConfigError("gateway port %d out of range" % self.port)
        ah = raw.get("allowed_hosts", [])
        ah = [ah] if isinstance(ah, str) else list(ah)
        env_ah = os.environ.get("CARDSHARE_ALLOWED_HOSTS", "")
        self.allowed_hosts = [x.strip().lower() for x in ah + env_ah.split(",") if x.strip()]
        co = raw.get("cors_origins", [])
        self.cors_origins = [co] if isinstance(co, str) else [str(x).rstrip("/") for x in co]
        self.api_key_file = os.environ.get("CARDSHARE_API_KEY_FILE") or raw.get("api_key_file")
        self.queue_timeout = _env_float("CARDSHARE_QUEUE_TIMEOUT", float(raw.get("queue_timeout_sec", 300)))
        self.keepalive_sec = float(raw.get("keepalive_sec", 10))
        self.retry_after_sec = int(raw.get("retry_after_sec", 30))
        self.upstream_read_timeout = float(raw.get("upstream_read_timeout_sec", 0)) or None
        self.list_aliases = bool(raw.get("list_aliases", False))
        self.gc_interval = float(raw.get("gc_interval_sec", 0))
        self.heartbeat_sec = float(raw.get("heartbeat_sec", 15))
        # JSONL request log (for tuning the idle timeouts later); rotated by size, N old files kept
        self.request_log = os.environ.get("CARDSHARE_REQUEST_LOG") or raw.get("request_log")
        self.request_log_max_bytes = int(float(raw.get("request_log_max_mb", 50)) * 1024 * 1024)
        self.request_log_keep = int(raw.get("request_log_keep", 5))
        # client classes: who is asking decides how long the model stays after the last request
        self.class_header = str(raw.get("class_header", "X-Cardshare-Class"))
        self.client_classes = []
        for i, c in enumerate(raw.get("client_classes", [])):
            if not isinstance(c, dict) or not c.get("name"):
                raise ConfigError("gateway.client_classes[%d]: needs a name" % i)
            match = c.get("match", {})
            if not isinstance(match, dict) or set(match) - {"header", "source", "key", "path"}:
                raise ConfigError("gateway.client_classes[%d].match: header, source, key, path" % i)
            try:
                nets = [ipaddress.ip_network(s, strict=False) for s in match.get("source", [])]
                hdrs = {k: re.compile(v, re.I) for k, v in (match.get("header") or {}).items()}
                path = re.compile(match["path"]) if match.get("path") else None
            except (ValueError, re.error) as exc:
                raise ConfigError("gateway.client_classes[%d]: %s" % (i, exc)) from exc
            keys = match.get("key", [])
            self.client_classes.append({
                "name": str(c["name"]),
                "idle_minutes": None if c.get("idle_minutes") is None else float(c["idle_minutes"]),
                "header": hdrs, "source": nets, "key": [keys] if isinstance(keys, str) else list(keys),
                "path": path})
        arb = raw.get("arbiter", "auto")
        if arb in ("auto", "inprocess"):
            self.arbiter = arb
        else:
            self.arbiter = _cmd(arb, "gateway.arbiter")


def load_config(explicit=None):
    path = find_config(explicit)
    try:
        raw = _read(path)
    except (OSError, ValueError) as exc:
        raise ConfigError("cannot read %s: %s" % (path, exc)) from exc
    return Config(raw, path)
