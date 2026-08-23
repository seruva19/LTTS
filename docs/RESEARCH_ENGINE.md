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
- `autointerp.py` keeps feature explanation providers separate from evaluation.
- `features.py` applies encoder/edit/decoder feature interventions.
- `diffing.py` aligns and compares two models or checkpoints.
- `architectures.py` exposes capability-gated SSM and multimodal primitives.
- `research_io.py` stores versioned research results with integrity checks.

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

`decompose_attention_heads()` accepts the concatenated head tensor immediately
before the attention output projection. `decompose_mlp_neurons()` accepts the
post-activation neuron tensor immediately before the MLP down projection. Both
return signed vocabulary projections and can be converted into attribution
graphs. They require these explicit boundaries because exact head or neuron
effects cannot generally be recovered from an already mixed residual output.
Use `ComponentInspector.capture_inputs()` to capture those boundaries. Standard
`nn.Linear` projections are detected automatically; modules that store weights
as `[input, output]` (including GPT-2's Conv1D projection) use
`weight_layout="in_out"`.

`ActivationDatasetScanner` accepts a streaming iterable and a feature extractor
such as `sae.encode`. It retains running moments, activation frequency, maxima,
and a bounded top-K heap per selected feature; it never stores the complete
activation dataset.

`build_feature_intervention()` edits selected encoded features and adds only the
decoded feature delta to the original hidden state. This prevents SAE
reconstruction error from being mistaken for the causal intervention.

`FeatureAutointerpreter` accepts a provider implemented by the caller. Provider
training evidence and scoring evidence must be disjoint. A separate predictor
scores the generated explanation against held-out activations, reporting
correlation, normalized absolute error, and top-versus-bottom separation. LTTS
does not require a specific LLM service and does not treat generated labels as
validated explanations without this independent score.

## Attribution graph adapters

Graph builders accept structured research results rather than rendered images.
`graph_from_dla_artifact()` and `graph_from_sae_artifact()` consume the actual
table envelopes emitted by the corresponding LTTS modules and retain artifact
row/module provenance. `graph_from_attention_matrix()` converts a numeric
`[query, key]` matrix into directed token edges with layer and head metadata.
Activation-patching sweeps already convert through `graph_from_causal_sweep()`.
The resulting graphs can be combined with `fuse_attribution_graphs()`; shared
token, logit, and metric targets are merged while source components remain
namespaced.

## SSM and multimodal research

`discover_research_adapter()` probes executable tensor boundaries instead of
matching model family names. It reports SSM state mixers, vision patch
boundaries, and fusion modules only when their required attributes or forward
arguments are actually present. Call `require()` before running a workflow that
depends on one of these capabilities.

`ssm_effective_influence()` calculates source-to-target influence for a supplied
diagonal recurrence and deliberately does not label it transformer attention.
`intervene_ssm_state()` performs explicit channel/position edits when an adapter
actually exposes a recurrent-state tensor. A standard Transformers Mamba mixer
only exposes its mixer output to ordinary hooks, so discovery reports mixer
capture/intervention but does not claim raw-state access. Vision patch scores
and cross-modal attention matrices have graph builders that retain patch
coordinates, modality, layer, head, and direction.

## Durable results

`bundle_result()` accepts any result with `to_dict()` (or a mapping).
`save_research_bundle()` writes an atomic UTF-8 JSON file with schema version,
timestamp, metadata, and SHA-256 checksum. `load_research_bundle()` verifies the
checksum and rejects unknown schema versions or malformed payloads. Model id,
revision, prompt/dataset id, seed, and module versions belong in bundle metadata
so a stored measurement retains its reproduction context.

## Model and generation comparisons

`ModelDiffer` aligns exact paths first and then unique semantic descriptors
(`role`, `stage`, `layer_index`, and attention kind). It reports unmatched paths
rather than silently comparing unrelated components. When both calls include
`input_ids`, the result also records token tensor shapes and up to 100 mismatch
positions, so differently tokenized prompts are not presented as a clean paired
comparison.

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

`GenerationExperimentRunner` repeats baseline and intervention generation with
the same RNG seed. Its durable result records complete and generated-only token
trajectories, changed decode positions, optional decoded text, renderable
artifacts, and the exact attachment schedules used by each trial.

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
