# Research engine

`ltts_engine` is the UI-independent layer for repeatable transformer research.
It builds on the same event and module contracts as the LTTS application but
does not import the web router, FastAPI, Socket.IO, or frontend code.

The public modules have intentionally narrow responsibilities:

- `runtime.py` instruments one PyTorch model and manages module lifecycle.
- `experiments.py` runs isolated baseline, control, and intervention trials.
- `graphs.py` defines versioned attribution nodes, edges, and graph builders.
- `inspection.py` captures components and projects changes into vocabulary space.
- `datasets.py` scans activation or SAE features with bounded memory.
- `features.py` applies encoder/edit/decoder feature interventions.
- `diffing.py` aligns and compares two models or checkpoints.

## Lifecycle

Create one `HeadlessSession` per instrumented model. Reuse it for trials and
close it when finished:

```python
from ltts_engine import HeadlessEngine

with HeadlessEngine().instrument(model) as session:
    output = session.run(**model_inputs)
```

`reset_modules()` detaches attachments, calls module cleanup functions, and
clears artifacts without re-registering model hooks. `close()` additionally
removes all hooks owned by the session. Experiment runners call
`reset_modules()` between trials, so attachment state cannot leak into the next
case.

## Comparable experiments

```python
from ltts_engine import ExperimentCase, ExperimentRunner, ModuleAttachment

result = ExperimentRunner(session).run(
    scorer=lambda output: output.logits[0, -1, target_id].item(),
    kwargs=model_inputs,
    seed=7,
    cases=(
        ExperimentCase(name="empty-control", kind="control"),
        ExperimentCase(
            name="intervention",
            attachments=(
                ModuleAttachment(module=module, layer=session.layers("block")[0]),
            ),
        ),
    ),
)
```

The result contains the baseline, signed deltas, timings, artifacts, seed, and
metadata. It deliberately does not retain model outputs, which may be very
large. `causal_sweep()` applies one intervention independently at each selected
layer and can be converted to an `AttributionGraph`.

## Component and feature research

`ComponentInspector` captures exact module paths during one forward pass. Its
vocabulary comparison reports the logits promoted or suppressed between a
reference component and a selected component. Callers may supply the model's
final normalizer when the architecture requires one.

`ActivationDatasetScanner` accepts a streaming iterable and a feature extractor
such as `sae.encode`. It retains running moments, activation frequency, maxima,
and a bounded top-K heap per selected feature; it never stores the complete
activation dataset.

`build_feature_intervention()` edits selected encoded features and adds only the
decoded feature delta to the original hidden state. This prevents SAE
reconstruction error from being mistaken for the causal intervention.

## Model and generation comparisons

`ModelDiffer` aligns exact paths first and then unique semantic descriptors
(`role`, `stage`, `layer_index`, and attention kind). It reports unmatched paths
rather than silently comparing unrelated components.

Attachments may include a generation schedule:

```python
session.attach(
    module,
    layer=layer,
    schedule={"generation_steps": [1, 3]},
)
```

Schedules are optional. Existing modules without one retain their previous
behavior. Decode steps are zero-based. Generation step information is also
available on `event.context` and `event.environment`.

## Extension rules

- Keep web orchestration out of `ltts_engine`.
- Prefer immutable, versioned result dataclasses with `to_dict()`.
- Keep tensors out of durable result objects unless an API explicitly promises
  tensor ownership.
- Validate layer alignment, shape, dtype, and device before comparison.
- Preserve module lifecycle isolation and add a regression test for every new
  research primitive.
- UI integration is a separate layer; do not make engine behavior depend on a
  particular visualization.
