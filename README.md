# cardshare

One OpenAI + Anthropic compatible endpoint for every model a machine can run. A model starts when a
request names it and goes away after it has been idle for a while; between models the GPU cards go back
to whatever background work they normally do.

> Early version (0.1). Linux + systemd. Tested with fakes only so far; see Limitations.

## Why

A box with a couple of big GPUs is usually either serving models or doing something else with the cards
(training, batch jobs, mining, rendering). Keeping every model loaded is impossible (one big MoE fills a
card and hundreds of GB of RAM), and starting them by hand means someone forgets to stop them.

cardshare puts one door in front of the cards:

* clients see a single API with every configured model listed in `/v1/models`;
* the first request for a model brings it up (and waits for it); concurrent requests share that start;
* a model nobody uses gives its cards back by itself, per-model idle timeout;
* each card has an optional **background workload** that runs whenever no model holds the card
  (a training run, a render queue, a miner, a batch job — a systemd unit or start/stop/check commands);
* nothing ever leaves a card empty after a failure: every rollback returns the background workload.

## How it works

Two parts, one config file:

* **`cardshare`** — the arbiter, a CLI (`up`, `down`, `gc`, `status`, `background`, `hold`, ...). It is
  the only thing that starts and stops models and background workloads. It keeps the state in one JSON
  file and takes one lock for every changing command.
* **`cardshare gateway`** (also `cardshare-gateway`) — an aiohttp server that proxies API requests and
  asks the arbiter to bring models up. It runs unprivileged and calls the arbiter through `sudo -n`
  (or in-process when it runs as root).

### Slots

Every card is a slot. A slot holds one model, or nothing (then its background workload runs). A model
declares the cards it may use, in order of preference (`"gpus": [1, 0]`), or `"all"` for a model that
spans every card. A model's class is `big` or `small`; at most `policy.max_big` big models (default 1)
run at once, whatever the cards.

`up <model>` picks the cards:

1. a free allowed card, in the model's order of preference;
2. otherwise a card whose model can be evicted: not serving a request (gateway lease or the server's own
   `/slots`), not used within `policy.evict_grace_sec`; asleep models first, then least recently used;
3. a big model also evicts the other big model(s) beyond `max_big`, models that share a port with it are
   evicted too (two presets of one server), and with `policy.host_mem_limit_gib` the least recently used
   idle models go until the host memory estimate fits;
4. if every option would evict a busy model, `up` fails with exit code 75 (the gateway queues and retries).

Taking a card stops only that card's background workload, waits until the card's VRAM is below
`gpu.free_threshold_mib`, starts the model and waits for its health URL. Releasing a card stops the model,
waits for the VRAM and returns that card's background workload (cards that share one workload get it back
when all of them are free).

### Idle timeouts

`gc` (a systemd timer, and optionally the gateway itself) stops a model once it has been idle for its
`idle_minutes` (default 3; `0` never). "Idle" counts from the last byte the gateway proxied or the last
time `gc` saw the server busy. A gateway **client class** may set its own timeout (agents 2 min, chat 3 min):
the class of the last request decides.

What a cycle costs on the machine this was written for: stopping the background workload ~1 s, starting it
again 3-5 s, a cold model start from 30 s to several minutes, a wake from sleep ~10 s. **Timeouts below one
minute are not worth it**: the background loses more to restarts than it gains.

### Sleep instead of stop

A model with `"sleep": true` (or an object with `sleep_url`, `wake_url`, `frees_card`, `release_minutes`)
is put to sleep when idle (`POST <api_url>/sleep`): the process stays, the card goes back to the background,
and the next request wakes it (`POST /wake`) instead of a cold start. A sleeping model is evicted (stopped)
first when its card is needed; with `release_minutes` it is stopped after sleeping that long. vLLM's sleep
mode (`--enable-sleep-mode`, `/sleep?level=1` and `/wake_up` with `VLLM_SERVER_DEV_MODE=1`) fits this; so
does any engine with such endpoints. If the card is not below `gpu.free_threshold_mib` after the sleep,
the model is stopped instead.

### Hold

`cardshare hold on [--card N]` creates a hold file: while it exists, released cards stay empty (no
background), which is what a series of tests on a card needs. `hold off` removes it; `background start`
starts the workloads explicitly (ignoring the hold).

### Budget

`policy.budget_minutes_per_hour` (default 0 = off): a model may take a card only while the card's
background workload has lost less than that many minutes in the last hour (`up --force` overrides).

## Configuration

JSON or TOML (Python 3.11+). Lookup: `--config`, `$CARDSHARE_CONFIG`, `/etc/cardshare/config.json`,
`/etc/cardshare/config.toml`, `$XDG_CONFIG_HOME/cardshare/config.json`. See `examples/single-gpu.json` and
`examples/two-gpu.json` (two cards with a background job each, big and small models, llama.cpp and vLLM
backends, a model on both cards, a model with sleep mode).

```jsonc
{
  "state_file": "/var/lib/cardshare/state.json",   // or $CARDSHARE_STATE
  "lock_file": "/run/cardshare.lock",              // or $CARDSHARE_LOCK
  "run_dir": "/run/cardshare",                     // or $CARDSHARE_RUN_DIR (activity leases, hold file)
  "hold_file": "/run/cardshare/hold",
  "commands": {"systemctl": ["systemctl"], "systemd_run": ["systemd-run"]},
  "unit_prefix": "cardshare-",                     // transient units: cardshare-<model>.service

  "gpu": {
    "query": ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    // any command printing "card, used MiB" lines; null = no VRAM checks (non-NVIDIA, CPU-only)
    "free_threshold_mib": 2000,
    "free_wait_sec": 120
  },

  "cards": {                                       // or a plain list ["0", "1"]; absent = from gpu.query
    "0": {"background": {"unit": "batch-0.service"}},
    "1": {"background": {"start": ["..."], "stop": ["..."], "check": ["..."]}}
  },

  "policy": {"max_big": 1, "budget_minutes_per_hour": 0, "evict_grace_sec": 15,
             "host_mem_limit_gib": 0, "lease_stale_sec": 90, "poll_sec": 2},

  "defaults": {"idle_minutes": 3, "start_timeout_sec": 900, "stop_timeout_sec": 300},  // $CARDSHARE_IDLE_MINUTES

  "models": {
    "coder": {
      "class": "small",                // big | small
      "gpus": [1, 0],                  // preference order, or "all"
      // one of three launch kinds:
      "exec": ["/usr/bin/llama-server", "--port", "8201", "-m", "/m/coder.gguf"],   // transient systemd unit
      "env": {"CUDA_VISIBLE_DEVICES": "{card}"}, "user": "llm", "properties": ["TimeoutStopSec=120"],
      // "unit": "my-model@{card}.service",                  // an existing (template) unit
      // "start": [...], "stop": [...], "check": [...],       // arbitrary commands
      "health_url": "http://127.0.0.1:8201/health",   // 200 (and status "ok" if JSON) = ready
      "api_url": "http://127.0.0.1:8201",             // default: health_url without /health
      "busy_url": "http://127.0.0.1:8201/slots",      // optional: llama.cpp-style /slots
      "api": ["openai", "anthropic", "responses"],   // what the backend speaks
      "aliases": ["qwen-coder"],
      "upstream_model": null,                         // rewrite "model" before proxying
      "upstream_headers": {},                         // e.g. a backend key
      "idle_minutes": 3, "sleep": false, "host_mem_gib": 0,
      "requires": ["/m/coder.gguf"],                  // hidden from the API while missing
      "hidden": false
    }
  },

  "aliases": {"fast": "coder", "coder-exact": {"model": "coder", "upstream_model": "coder-exact"}},

  "gateway": {
    "host": "127.0.0.1", "port": 8100,             // --host/--port > $CARDSHARE_HOST/$CARDSHARE_PORT > here
    "allowed_hosts": ["box.lan"],                  // + $CARDSHARE_ALLOWED_HOSTS; "*" = any
    "cors_origins": [],
    "api_key_file": null,                          // $CARDSHARE_API_KEY_FILE; lines "key" or "name:key"
    "arbiter": "auto",                             // auto | inprocess | ["sudo", "-n", "/opt/cardshare/bin/cardshare"]
    "queue_timeout_sec": 300, "keepalive_sec": 10, "retry_after_sec": 30,
    "upstream_read_timeout_sec": 0,                // 0 = none (long generations)
    "gc_interval_sec": 0,                          // >0: the gateway also runs gc
    "request_log": null,                           // $CARDSHARE_REQUEST_LOG; JSONL
    "request_log_max_mb": 50, "request_log_keep": 5,
    "class_header": "X-Cardshare-Class",
    "client_classes": [
      {"name": "agent", "match": {"header": {"User-Agent": "claude-cli|codex"}}, "idle_minutes": 2},
      {"name": "lan", "match": {"source": ["198.51.100.0/24"]}, "idle_minutes": 5},
      {"name": "bot", "match": {"key": ["bot"]}}
    ]
  }
}
```

Placeholders `{name}`, `{card}` (the first chosen card) and `{cards}` (`0,1`) are filled in unit names,
commands, `env` values and URLs. Nothing else is touched, so `${VAR}` in a command stays as it is.

On NVIDIA set `CUDA_DEVICE_ORDER=PCI_BUS_ID` next to `CUDA_VISIBLE_DEVICES={card}`, so the card numbers of
the CUDA runtime match `nvidia-smi`.

## Install (systemd)

```bash
sudo python3 -m venv /opt/cardshare
sudo /opt/cardshare/bin/pip install '/path/to/cardshare[gateway]'
sudo install -d /etc/cardshare && sudo cp examples/single-gpu.json /etc/cardshare/config.json   # then edit
sudo /opt/cardshare/bin/cardshare models --all       # checks the config, shows missing files

# arbiter timer (gc every 30 s)
sudo cp examples/systemd/cardshare-gc.{service,timer} /etc/systemd/system/
# gateway as an unprivileged user that may run the arbiter without a password
sudo useradd --system --no-create-home cardshare
echo 'cardshare ALL=(root) NOPASSWD: /opt/cardshare/bin/cardshare' | sudo tee /etc/sudoers.d/cardshare
sudo cp examples/systemd/cardshare-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cardshare-gc.timer cardshare-gateway.service
curl -s http://127.0.0.1:8100/v1/models
```

The gateway writes activity leases into `run_dir` (`RuntimeDirectory=cardshare` in the unit makes it
writable for the gateway user); the arbiter only reads them.

## API

| Method | Path | |
|---|---|---|
| GET | `/v1/models`, `/v1/models/{id}` | every visible model with `cardshare.state`: `running`, `starting`, `asleep`, `off`; Anthropic shape when the request has `anthropic-version` |
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | OpenAI, routed by `model` |
| POST | `/v1/responses` | OpenAI Responses, for backends with `"responses"` in `api` |
| POST | `/v1/messages`, `/v1/messages/count_tokens` | Anthropic, for backends with `"anthropic"` in `api` |
| GET | `/cardshare/status` | slots, models, in-flight requests (files only, no probing) |
| GET | `/health` | the gateway itself |

* Requests and answers pass through unchanged (`model` is rewritten only for `upstream_model`); SSE is
  streamed chunk by chunk.
* While a model is starting, a streaming request gets `200` with SSE comments (`: cardshare: loading ...`)
  every `keepalive_sec`; a non-streaming one just waits.
* No room (every option would evict a busy model): the request waits up to `queue_timeout_sec`, then `503`
  with `Retry-After`. Budget used up: `503`. The model failed to start: `502` (or an SSE `error` event).
* A backend that does not speak the requested API gets a clear `400` (`unsupported_api`), no translation.
* Errors use the dialect of the route: OpenAI `{"error": {...}}`, Anthropic `{"type": "error", ...}`.
* Access control: see Security.

## Security

* **Loopback by default.** The gateway listens on `127.0.0.1:8100`. Another address is an explicit choice:
  `cardshare gateway --host 0.0.0.0 --port 8100`, `$CARDSHARE_HOST`/`$CARDSHARE_PORT`, or `gateway.host`/
  `gateway.port` (priority in that order). Listening on anything but loopback without API keys works, but
  the gateway logs a loud warning at start: whoever reaches the port can use every model and make the
  machine start and stop them (and take cards from the background work).
* **API keys** (optional): `gateway.api_key_file` or `$CARDSHARE_API_KEY_FILE`, one key per line, `name:key`
  to give it a name (the name goes to the request log and can select a client class). Every request except
  `GET /health` then needs `Authorization: Bearer <key>` or `x-api-key: <key>`. Keys are compared in
  constant time. The client's key is **not** forwarded to the backend; a backend that needs its own key gets
  it from `upstream_headers`.
* **Host allow-list** (when no keys are set), against DNS rebinding: a request passes only if its Host is
  an IP address, `localhost`/`*.localhost`, this machine's name (`name`, `name.local`) or in
  `gateway.allowed_hosts` / `$CARDSHARE_ALLOWED_HOSTS` (`".example.com"` = it and every name below it,
  `"*"` = any). A web page whose DNS name points at the gateway carries its own name and gets 403.
* **Browser Origin check** (when no keys are set): a request with an `Origin` header passes only for the
  gateway's own page, a name it answers to, or `gateway.cors_origins`. CORS headers are sent only for
  `cors_origins`, on `/v1/*`.
* The gateway itself needs no root: it asks the arbiter through `sudo -n`, limited by a sudoers rule to the
  `cardshare` binary. Model servers should listen on 127.0.0.1 only; the gateway is the way in.

### Request log

One JSON line per proxied request: `ts`, `t`, `path`, `client`, `class`, `key` (name), `model` (as asked),
`preset`, `cards`, `stream`, `cold`, `wait_up_sec` (time spent bringing the model up), `duration_sec`,
`status`, `bytes`, `error`. Rotated by size (`request_log_max_mb`, `request_log_keep` old files). Meant for
tuning the idle timeouts from real gaps between requests.

## CLI

```
cardshare status [--json]
cardshare up NAME [--card N|auto] [--force] [--preempt] [--no-switch] [--json]
cardshare down [NAME] [--card N]
cardshare gc
cardshare background start|stop [--card N]
cardshare hold on|off [--card N]
cardshare sleep NAME · cardshare touch NAME
cardshare models [--all] [--json]                (alias: presets) · cardshare show NAME
cardshare gateway [--host ADDR] [--port N]       (also: cardshare-gateway)
```

Exit codes: 0 ok, 1 error, 2 usage/config, 75 no room (busy), 76 budget used up.

## Limitations

* Linux and systemd (transient units for `exec` models); start/stop/check commands work without systemd.
* One `up` at a time: the arbiter holds its lock while a model loads, so a small model waits for a big one
  that is loading on the other card.
* No format translation between OpenAI and Anthropic APIs: a backend gets only the APIs it lists.
* Host memory is an estimate you give (`host_mem_gib`), not measured.
* The sleep mode needs a server with sleep and wake endpoints (vLLM's sleep mode, or your own); servers without them are stopped and started instead.
* Tested against fakes (`tests/`): no run on real GPUs and real model servers yet.

## Development

```bash
python -m unittest discover -s tests        # or: pytest
```

The gateway tests need aiohttp and are skipped without it. `tests/fakes.py` has an in-memory `Host`
(systemd, VRAM, health) and a stub OpenAI/Anthropic server.

## License

MIT, see `LICENSE`.
