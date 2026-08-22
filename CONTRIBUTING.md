# Contributing to LTTS

Analysis modules belong in LTTS-modules. Loader, API, model-adapter, and UI
changes belong in this repository.

## Where code goes

| What | Where |
|---|---|
| A new analysis technique | [LTTS-modules](https://github.com/seruva19/LTTS-modules) — one directory per module |
| Loader, model adapters, API, UI | this repository |
| The roadmap itself | `README.md` in this repository |

Modules load from a sibling `LTTS-modules` checkout by default.
`LTTS_MODULES_DIR` overrides that path. Registry installations are written to
the ignored local `ltts_modules/` directory.

## Writing a module

Read [docs/MODULE_AUTHORING.md](docs/MODULE_AUTHORING.md). It is the contract:
metadata, declared capabilities, target roles, supported topologies, the event
signature, and how results reach the UI.

The loader enforces these rules:

- `METADATA["name"]` must match the module's directory name.
- Declare every capability you use (`blocks`, `attention`, `gradients`,
  `tokenizer`, `unembedding`). The backend rejects incompatible attachments
  before inference.
- Declare optional third-party dependencies in the module's `requirements.txt`.
  They are installed into a per-module virtual environment.
- A module that cannot produce a result must say so. Silence is not success —
  the run diagnostics surface modules that emitted nothing.

Emit results through `scripts.core.module_data_sender.get_data_sender()` and
import it by its full path. Bare imports can create a separate registry
instance.

## Working on the backend or the UI

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements-cuda.txt   # NVIDIA GPU only
python -m pip install -r requirements-dev.txt
npm install
```

Before opening a PR:

```
python -m pytest -q
npm run build-client
npx vitest run
```

Module checks:

- `tests/test_module_contracts.py` checks metadata, emission channels, and path
  resolution with synthetic events.
- `tests/test_modules_on_real_model.py` builds a three-layer GPT-2 and BERT
  from a config — no download — attaches modules through the real attachment
  path and runs a forward pass.

`tests/manual_e2e_all_modules.py` attaches *every* module to a real checkpoint
and reports per-module health. It downloads a checkpoint and is not part of the
default suite.

## UI/backend field consistency

Fields used by layer cards must also be present in categorized layer entries.
Run these checks:

- `tests/test_ui_field_flow.py` — no layer property the UI reads may exist on
  the flat `layers` array while missing from `categorization`, which is what the
  cards are rendered from.
- `tests/test_ui_backend_attachment_agreement.py` — walks every module against
  every layer role and asserts the UI's verdict matches the backend's. Offering
  too much means a 422 at run time; offering too little hides a usable module.
- `tests/manual_audit_ui_fields.py` — the same sweep against the real
  checkpoints in the registry, where architecture-specific fields appear.

When adding a field the UI depends on, add it to the categorized layer entries
too (`scripts/models/model_struct.py`), and prefer fail-closed comparisons:
`capabilities[x] === true`, not `!== false`.

## Conventions

- **Logging** uses lazy `%s` formatting (`logger.debug("loaded %s", name)`),
  never f-strings, so disabled log levels cost nothing.
- **HTTP errors carry exception text.** LTTS is a single-user tool that binds to
  `127.0.0.1` by default, and the exception is usually the answer the user
  needs. Do not put credentials or file contents into an error message.
- **The server executes third-party module code in-process.** Anything that
  widens that surface — remote module installation, binding to `0.0.0.0`,
  loading remote-code checkpoints — must stay opt-in and be documented.

## Adding to the roadmap

New entries need a link to the paper and, where one exists, to a canonical
implementation. The roadmap describes *techniques*, so an entry is legitimate
before anything is wired up here — but it must be a distinct technique, not a
rename of an entry that already exists.
