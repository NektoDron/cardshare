"""`cardshare` command line: the arbiter's commands plus `cardshare gateway`."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

from . import __version__
from .arbiter import Arbiter, ArbiterError, hhmm
from .config import ConfigError, load_config


def _parser():
    ap = argparse.ArgumentParser(prog="cardshare",
                                 description="share GPU cards between on-demand models and background workloads")
    ap.add_argument("--config", help="config file (default: $CARDSHARE_CONFIG, /etc/cardshare/config.json, ...)")
    ap.add_argument("--version", action="version", version="cardshare " + __version__)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status", help="what is on the cards").add_argument("--json", action="store_true")
    p = sub.add_parser("up", help="bring a model up (stops the background on its cards)")
    p.add_argument("name")
    p.add_argument("--card", default="auto", help="card id or auto (default: by policy)")
    p.add_argument("--force", action="store_true", help="ignore the background budget")
    p.add_argument("--preempt", action="store_true", help="may evict models that are serving requests")
    p.add_argument("--no-switch", action="store_true", help="fail rather than evict another model")
    p.add_argument("--json", action="store_true", help="print the result as JSON (messages go to stderr)")
    p = sub.add_parser("down", help="stop a model (or all) and return the background")
    p.add_argument("name", nargs="?")
    p.add_argument("--card", help="stop whatever is on this card")
    p.add_argument("--if-owned", action="store_true", help="compatibility no-op: down NAME only ever stops NAME")
    p.add_argument("--json", action="store_true")
    sub.add_parser("gc", help="idle models give their cards back (run by a timer)").add_argument(
        "--json", action="store_true")
    p = sub.add_parser("background", help="start/stop background workloads directly")
    p.add_argument("action", choices=["start", "stop"])
    p.add_argument("--card")
    p = sub.add_parser("hold", help="hold on: freed cards keep no background (test series); hold off")
    p.add_argument("action", choices=["on", "off"])
    p.add_argument("--card")
    sub.add_parser("sleep", help="put a model with a sleep mode to sleep now").add_argument("name")
    sub.add_parser("touch", help="reset a model's idle timer").add_argument("name")
    p = sub.add_parser("models", aliases=["presets"], help="configured models")
    p.add_argument("--all", action="store_true", help="also hidden ones and ones with missing files")
    p.add_argument("--json", action="store_true")
    sub.add_parser("show", help="one model in full").add_argument("name")
    p = sub.add_parser("gateway", help="run the OpenAI/Anthropic API gateway (needs aiohttp)")
    p.add_argument("--host", help="address to listen on (default: $CARDSHARE_HOST, gateway.host, 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default: $CARDSHARE_PORT, gateway.port, 8100)")
    p.add_argument("--config", dest="gw_config", help="config file (same as the global --config)")
    return ap


def _print_status(arb, data):
    print("time        %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    for c, info in data["cards"].items():
        model = info["model"]
        line = "card %-6s %s" % (c, model or "-")
        if model:
            mi = data["models"][model]
            line += " [%s, up %s, idle %s%s]" % (mi["state"], hhmm(mi["up_sec"]), hhmm(mi["idle_sec"]),
                                                 ", %d in flight" % mi["inflight"] if mi["inflight"] else "")
        if info["background"]:
            line += " · background %s" % info["background"]
        if info["background_downtime_last_hour_min"]:
            line += " (down %.1f min/h)" % info["background_downtime_last_hour_min"]
        if info["held"]:
            line += " · HELD"
        print(line)
    if data["hold"]:
        print("hold        %s — released cards keep no background" % arb.cfg.hold_file)
    b = data["budget_minutes_per_hour"]
    print("budget      %s" % ("%g min/h per card" % b if b else "none"))


def _models(arb, show_all, as_json):
    st = arb.load_state()
    rows = []
    for name in sorted(arb.cfg.models):
        m = arb.cfg.models[name]
        miss = m.missing()
        if (m.hidden or miss) and not show_all:
            continue
        rec = st["models"].get(name)
        rows.append({"name": name, "class": m.klass, "gpus": m.gpus, "idle_minutes": m.idle_minutes,
                     "sleep": bool(m.sleep), "aliases": m.aliases, "api": m.api, "desc": m.desc,
                     "missing": miss, "up_on": rec["cards"] if rec else None})
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    width = shutil.get_terminal_size((120, 24)).columns
    wn = max([len(r["name"]) for r in rows] + [4])
    for r in rows:
        gp = "all" if r["gpus"] == "all" else ",".join(r["gpus"])
        head = "%s %-*s %-5s cards %-4s idle %-4g " % ("*" if r["up_on"] else " ", wn, r["name"], r["class"],
                                                       gp, r["idle_minutes"])
        desc = " ".join(r["desc"].split())
        room = width - len(head)
        print((head + (desc[:room - 1] + "…" if len(desc) > room > 8 else desc if room > 8 else "")).rstrip())
        for p in r["missing"]:
            print("      missing %s" % p)
    return 0


def _show(arb, name):
    m = arb.cfg.models.get(name)
    if not m:
        raise ArbiterError("no such model: %s" % name)
    print(json.dumps(m.raw, ensure_ascii=False, indent=2))
    print("ports %s · aliases %s · missing files: %s" % (
        ",".join(map(str, m.ports)) or "-", ", ".join(m.aliases) or "-", ", ".join(m.missing()) or "none"))
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    if not args.cmd:
        _parser().print_help()
        return 2
    try:
        cfg = load_config(getattr(args, "gw_config", None) or args.config)
    except ConfigError as exc:
        print("cardshare: %s" % exc, file=sys.stderr)
        return 2
    if args.cmd == "gateway":
        try:
            from .gateway import run
        except ImportError as exc:
            print("cardshare: the gateway needs aiohttp (pip install 'cardshare[gateway]'): %s" % exc,
                  file=sys.stderr)
            return 2
        return run(cfg, host=args.host, port=args.port)

    as_json = getattr(args, "json", False)
    out = sys.stderr if as_json else sys.stdout
    arb = Arbiter(cfg, say=lambda msg: print(msg, file=out, flush=True))
    card = getattr(args, "card", None)
    try:
        if args.cmd == "status":
            data = arb.status()
            if as_json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                _print_status(arb, data)
            return 0
        if args.cmd in ("models", "presets"):
            return _models(arb, args.all, as_json)
        if args.cmd == "show":
            return _show(arb, args.name)
        if args.cmd == "up":
            res = arb.up(args.name, card=card, force=args.force, preempt=args.preempt, no_switch=args.no_switch)
        elif args.cmd == "down":
            res = arb.down(args.name, card=card)
        elif args.cmd == "gc":
            res = arb.gc()
        elif args.cmd == "background":
            res = arb.background(args.action, card=card)
        elif args.cmd == "hold":
            res = arb.hold(args.action == "on", card=card)
        elif args.cmd == "sleep":
            res = arb.sleep_model(args.name)
        elif args.cmd == "touch":
            res = arb.touch(args.name)
        else:
            return 2
    except ArbiterError as exc:
        if as_json:
            print(json.dumps({"ok": False, "error": str(exc), "code": exc.code,
                              "blockers": getattr(exc, "blockers", [])}, ensure_ascii=False))
        print("cardshare: %s" % exc, file=sys.stderr)
        return exc.code
    except PermissionError as exc:
        print("cardshare: %s (state, lock and units usually need root: sudo cardshare %s ...)"
              % (exc, args.cmd), file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("ok", True) else 1


def gateway_main(argv=None):
    """`cardshare-gateway` entry point = `cardshare gateway`."""
    return main(["gateway"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
