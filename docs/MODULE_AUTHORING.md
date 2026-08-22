# LTTS Module Authoring Guide

A module contains `init.py` and `requirements.txt`. The loader discovers modules
at startup. If a module has a virtual environment, its `site-packages` directory
is added while loading it.

## Contract

`init.py` must define a `METADATA` dict (plain literals only — it is parsed
with `ast.literal_eval` by the registry scanner):

```python
METADATA = {
    "name": "my_module",                  # required, must match directory name
    "version": "0.1.0",                   # required (str)
    "description": "One-line summary",
    "author": "You",
    "event_types": ["layer_after"],       # required: layer_before / layer_after
    "dependencies": ["requirements.txt"],
    "methods": {
        "ctor": "initialize_module",      # called once at load: (context, **config)
        "dtor": "cleanup_module",         # called at unload: ()
        "ntor": "process_event",          # called per event: (ltts_event)
        "utor": "get_ui_schema",          # called for UI: () -> dict
    },
}
```

## Lifecycle methods

### `initialize_module(context, **config)`
Create a data sender and bind it to the runtime context:

```python
from scripts.core.module_data_sender import get_data_sender
_SENDER = None

def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized"}
```

### `process_event(ltts_event)`
Called on every subscribed event. The event provides:

- `ltts_event.event_type` — `"layer_before"`, `"layer_after"`, or the
  layer-specific `"layer_after_<path>"`. Handle `layer_after` and its
  layer-specific variants together to avoid double emission:
  `evt == "layer_after" or evt.startswith("layer_after_")`.
- `ltts_event.context` — `LTTSContext` with `inputs`, `outputs` (layer output;
  may be a tuple — take `[0]` for the hidden state `[batch, seq, hidden]`),
  `attention_weights`, `module_path` (layer path string), `module_class`,
  `layer_type` (`attention`/`mlp`/`moe`/`normalization`/`embedding`/`unknown`),
  `layer_state`, `model_state`, and `ltts_model` (the live `LTTSModel`; its
  `original_model` is the HuggingFace model, and `_tokenizer` /
  `_last_input_tokens` are available when inference set them).
- Miners, lazily: `ltts_event["attention"]`, `["embeddings"]`, `["ffn"]`,
  `["moe"]`, `["normalization"]` (or the same-named properties).
- `ltts_event.module_state` — persistent per-module dict; merge UI config from
  it plus `ltts_event.environment["module_config"]`.
- `ltts_event.context.analysis_results` — cross-module sharing.

Respect the emission controller so `emit_mode` (all / first / every_n /
final / manual) works:

```python
emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
if not emission.get("emit"):
    return {"status": "skipped", "reason": "emission_controller"}
...
if config.get("emit_mode") == "final":
    ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
```

Catch processing errors, log them, and emit an error signal:
`ltts_event.reactor.signal("error", {"message": ..., "module": ...})`.

### `get_ui_schema()`
Returns the parameter panel rendered by the frontend:

```python
{
    "parameters": [
        {"name": "top_k", "type": "number", "label": "Top K", "default": 5,
         "min": 1, "max": 20},
        {"name": "mode", "type": "select", "label": "Mode",
         "default": "table", "options": [
             {"value": "table", "label": "Table"},
             {"value": "chart", "label": "Chart"}]},
        {"name": "enabled", "type": "boolean", "label": "Enabled", "default": True},
        {"name": "label", "type": "text", "label": "Label", "default": "..."},
    ],
    "layout": {"type": "vertical"},
    "real_time": {"enabled": True, "interval": 200},
}
```

Supported parameter types: `boolean`, `number`, `text`, `select`,
`multiline_text`.

## Emitting data

All senders take `layer_path=...` (so the UI attaches the payload to the right
layer card) and an `emit_id` for deduplication:

| Sender | UI rendering |
|---|---|
| `send_scalar(value, label=...)` | big number + unit |
| `send_text(content, preformatted=...)` | text block |
| `send_table(headers, rows)` | table |
| `send_chart({"series": [{"name", "points": [{"x","y"},...]}], "x_label", "y_label"}, "line")` | line/bar/histogram chart |
| `send_image(base64_str)` | `<img>` (use a data URI) |
| `send_heatmap(values, dimensions)` | heatmap |
| `send_visualization({"type": "json", "data": ...}, "json")` | JSON tree |

Render Matplotlib figures as PNG data URIs and call `plt.close()` (see
`attn_visualizer/visuals.py` in LTTS-modules).

## Stateful modules (buttons + custom routes)

For run-on-demand analyses (multi-pass, training, dataset jobs), subclass
`LTTSModuleBase` (`scripts/core/ltts_module.py`) in `init.py`. The loader
instantiates it automatically and exposes buttons in the UI:

```python
from scripts.core.ltts_module import LTTSModuleBase

class MyModule(LTTSModuleBase):
    def _register_module_routes(self):
        self.add_stateful_button(
            {"id": "run", "label": "Run analysis", "variant": "primary"},
            "handle_run",
        )

    async def handle_run(self, event_data, module_state):
        ...  # POST /api/modules/<module_name>/events/run
```

## Interventions (modifying activations)

Modules can causally intervene, not just observe. During `process_event`, set:

- `ltts_event.context.analysis_results["modified_outputs"] = new_outputs` on a
  `layer_after` event to replace the layer's outputs, or
- `ltts_event.context.analysis_results["modified_inputs"] = new_inputs` on a
  `layer_before` event to replace its inputs.

The replacement must match the original structure exactly — if the layer
returned a tuple `(hidden, attn, ...)`, return a tuple of the same arity with
your modified hidden state at index 0 (rebuild it with
`(new_hidden,) + tuple(outputs[1:])`). Shape and dtype must match. This is how
ablation, patching, and steering modules are built (`neuron_ablation`,
`activation_patching`).

## Optional heavy dependencies

Declare them in the module's `requirements.txt` and import defensively with a
clear fallback message (see `model_control_vector`'s `repeng` handling) so the
module degrades visibly, not silently.
