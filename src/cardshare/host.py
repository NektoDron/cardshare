"""Everything the arbiter does to the machine goes through one object, so tests can swap it.

``Host`` runs commands (systemctl, systemd-run, the GPU query, start/stop/check commands of models and
background workloads) and talks HTTP to model servers.  A test passes a subclass that keeps all of this
in memory; nothing else in the arbiter touches the system.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request


class Host:
    def __init__(self, cfg):
        self.cfg = cfg

    # ── time ──────────────────────────────────────────────────────────────
    def now(self):
        return time.time()

    def sleep(self, sec):
        time.sleep(sec)

    # ── commands ──────────────────────────────────────────────────────────
    def run(self, cmd, timeout=120, env=None):
        """(exit code, combined output).  Never raises: a missing program or a timeout is code 1."""
        full_env = None
        if env:
            full_env = dict(os.environ)
            full_env.update(env)
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, env=full_env)
            return p.returncode, p.stdout.decode("utf-8", "replace").strip()
        except (subprocess.TimeoutExpired, OSError) as exc:
            return 1, str(exc)

    def unit_active(self, unit):
        return self.run(self.cfg.systemctl + ["is-active", "--quiet", unit], timeout=15)[0] == 0

    def unit_start(self, unit, timeout=180):
        return self.run(self.cfg.systemctl + ["start", unit], timeout=timeout)

    def unit_stop(self, unit, timeout=180):
        return self.run(self.cfg.systemctl + ["stop", unit], timeout=timeout)

    def transient_start(self, unit, cmd, env, user=None, properties=()):
        """Run `cmd` as a transient service (systemd-run), so systemd owns and can kill the whole tree."""
        args = list(self.cfg.systemd_run) + ["--unit=" + unit, "--collect", "--quiet",
                                             "--property=KillMode=control-group"]
        if user:
            args.append("--property=User=" + user)
        for p in properties:
            args.append("--property=" + p)
        for k, v in sorted(env.items()):
            args.append("--setenv=%s=%s" % (k, v))
        return self.run(args + ["--"] + list(cmd), timeout=60)

    # ── GPUs ──────────────────────────────────────────────────────────────
    def gpu_used_mib(self):
        """{card: used MiB} from the configured query (CSV "index, used"), or None if unknown."""
        if not self.cfg.gpu_query:
            return None
        rc, out = self.run(self.cfg.gpu_query, timeout=30)
        if rc != 0:
            return None
        used = {}
        try:
            for line in out.splitlines():
                if not line.strip():
                    continue
                idx, val = [x.strip() for x in line.split(",")[:2]]
                used[str(idx)] = float(val)
        except ValueError:
            return None
        return used

    # ── HTTP to model servers ─────────────────────────────────────────────
    def http(self, method, url, timeout=3, body=None):
        """(status, parsed JSON or None).  status None: no answer at all."""
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read() if exc.fp else b""
        except (urllib.error.URLError, OSError, ValueError):
            return None, None
        try:
            return status, json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            return status, None
