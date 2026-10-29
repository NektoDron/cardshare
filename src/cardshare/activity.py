"""Activity leases: how the gateway tells the arbiter "this model is in use" without taking its lock.

The gateway (an unprivileged process) keeps one small JSON file per model in ``<run_dir>/activity/``:
``{"inflight": 2, "last": <unix time of the last byte>, "updated": <unix time>, "pid": 1234,
"idle_minutes": 2}``.  ``idle_minutes`` (optional) is the idle timeout of the client class that sent the
last request; it overrides the model's own timeout.
It rewrites the file when a request starts or ends and every few seconds while requests are in flight.
The arbiter only reads these files: a fresh file with ``inflight > 0`` means busy, ``last`` counts
as use for the idle timer.  A file nobody refreshed for ``lease_stale_sec`` (gateway killed) is ignored.
"""

from __future__ import annotations

import json
import os
import re
import tempfile


def lease_dir(run_dir):
    return os.path.join(run_dir, "activity")


def _path(run_dir, name):
    return os.path.join(lease_dir(run_dir), re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".json")


def write_lease(run_dir, name, inflight, last, now, idle_minutes=None):
    d = lease_dir(run_dir)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".lease.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            rec = {"inflight": inflight, "last": last, "updated": now, "pid": os.getpid()}
            if idle_minutes is not None:
                rec["idle_minutes"] = idle_minutes
            json.dump(rec, fh)
        os.chmod(tmp, 0o644)
        os.replace(tmp, _path(run_dir, name))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_lease(run_dir, name, now, stale_sec):
    """(inflight, last, idle_minutes or None) — inflight is 0 when the file is missing or stale."""
    try:
        with open(_path(run_dir, name), encoding="utf-8") as fh:
            d = json.load(fh)
        inflight = int(d.get("inflight", 0))
        last = float(d.get("last") or 0)
        updated = float(d.get("updated") or 0)
        idle = d.get("idle_minutes")
        idle = None if idle is None else float(idle)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0, 0.0, None
    if now - updated > stale_sec:
        inflight = 0
    return inflight, last, idle
