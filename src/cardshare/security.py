"""Who may talk to the gateway: Host allow-list (DNS rebinding), browser Origins, optional API keys.

The rules (the same as several local LLM servers use, so clients behave alike):

* Host: an IP address, ``localhost``/``*.localhost``, this machine's name (and ``.local``) and the
  configured ``allowed_hosts`` (``"*"`` = any, ``".example.com"`` = it and every name below it).  A web
  page whose DNS name points at this machine carries its own name in Host and is refused.
* Origin (browsers only): the request's own Host, a name this server answers to, or ``cors_origins``.
* With an API key configured both checks are off: every request must carry the key instead.
"""

from __future__ import annotations

import hmac
import ipaddress
import re
import socket

LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")


def host_name(value):
    """'Box.lan:8100' -> 'box.lan', '[::1]:8100' -> '::1'."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        return v[1:].split("]", 1)[0]
    if v.count(":") == 1:
        v = v.split(":", 1)[0]
    return v.rstrip(".")


def _is_ip(name):
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _name_in(name, names):
    return name in names or any(n.startswith(".") and (name.endswith(n) or name == n[1:]) for n in names)


def host_names(allowed_hosts):
    names = set(LOOPBACK_NAMES)
    try:
        pc = socket.gethostname().lower()
        names.update((pc, pc + ".local"))
    except OSError:
        pass
    names.update(h for h in allowed_hosts if h != "*")
    return names


def host_allowed(host, names, any_host=False):
    if any_host or not (host or "").strip():
        return True
    name = host_name(host)
    return bool(name) and (_is_ip(name) or name == "localhost" or name.endswith(".localhost")
                           or _name_in(name, names))


def origin_allowed(origin, host, names, cors_origins=()):
    origin = (origin or "").strip().rstrip("/")
    if origin in cors_origins or "*" in cors_origins:
        return True
    scheme, sep, rest = origin.lower().partition("://")
    if not sep or not scheme:
        return False                            # "null" and garbage: any site can send it
    if scheme not in ("http", "https"):
        return True                             # extensions, app:// — no web site can send these
    if host and rest == host.strip().lower():
        return True                             # the gateway's own page
    name = host_name(rest)
    return bool(name) and (name in LOOPBACK_NAMES or name.endswith(".localhost") or _name_in(name, names))


def load_keys(path):
    """Key file: one key per line, optionally named ``name:key`` (the name goes to the request log
    and can select a client class).  '#' starts a comment line."""
    keys = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mm = re.match(r"([A-Za-z0-9_.-]+):(.+)$", line)
            name, key = (mm.group(1), mm.group(2).strip()) if mm else ("key%d" % i, line)
            keys[key] = name
    if not keys:
        raise ValueError("%s has no keys" % path)
    return keys


def key_of_request(headers):
    auth = headers.get("Authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return (headers.get("x-api-key") or "").strip()


def match_key(given, keys):
    """Name of the matching key or None; compares against every key in constant time."""
    found = None
    g = given.encode("utf-8", "replace")
    for k, name in keys.items():
        if hmac.compare_digest(g, k.encode("utf-8", "replace")) and found is None:
            found = name
    return found
