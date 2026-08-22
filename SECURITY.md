# Security

## Threat model

LTTS is a **local research tool**, not a hardened service. Two properties follow
from what it does and will not change:

1. **Analysis modules execute arbitrary Python inside the server process.** That
   is the plugin contract. Installing a module is equivalent to running its code
   on your machine.
2. **There is no authentication.** Anyone who can reach the port can load
   models, run inference, and attach modules.

Because of that, the server binds `127.0.0.1` by default. Set
`LTTS_HOST=0.0.0.0` only inside a container or behind a tunnel you control, and
never on a shared machine or a public network.

Two further switches are opt-in for the same reason:

- `LTTS_ALLOW_MODULE_INSTALL` — installing modules from a remote registry over
  the API. Off unless set.
- `trust_remote_code` in the model registry — running checkpoint-supplied code
  from Hugging Face. Per-model, off unless set.

## Reporting a vulnerability

Open a [security advisory](https://github.com/seruva19/LTTS/security/advisories/new)
rather than a public issue, and give it a few days.

Things that are **in** scope: anything that lets a *model file*, a *checkpoint*,
a *prompt*, or a *WebSocket message* execute code or read files outside the
intended paths; path traversal in the blueprint, settings, or module-install
paths; and injection into the UI from backend-supplied data.

Things that are **out** of scope: that a deliberately installed module can do
anything, and that an unauthenticated port bound to `0.0.0.0` on purpose is
reachable. Those are documented properties above, not bugs.
