# Research engine

`ltts_engine` is the UI-independent layer for repeatable transformer research.
It builds on the same event and module contracts as the LTTS application but
does not import the web router, FastAPI, Socket.IO, or frontend code.

The public modules have intentionally narrow responsibilities:

- `runtime.py` instruments one PyTorch model and manages module lifecycle.
- `experiments.py` runs isolated baseline, control, and intervention trials.
- `graphs.py` defines versioned attribution nodes, edges, and graph builders.
- `attributions.py` normalizes token attribution artifacts and score semantics.
- `inspection.py` captures components and projects changes into vocabulary space.
- `datasets.py` scans activation or SAE features with bounded memory.
- `autointerp.py` keeps feature explanation providers separate from evaluation.
- `features.py` applies encoder/edit/decoder feature interventions.
- `diffing.py` aligns and compares two models or checkpoints.
- `architectures.py` exposes capability-gated SSM and multimodal primitives.
- `research_io.py` stores versioned research results with integrity checks.
- `studies.py` composes result stages into a validated dependency graph.

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

Vocabulary drill-down also carries a `ResearchSelectionContext` containing the
run, output token, layer, component kind/index, and token position. Graph nodes
derived from selections, sweeps, DLA, SAE, heads, or neurons expose an
`InterventionTargetRef`, so a consumer can construct a follow-up experiment
without parsing display labels.

`ActivationDatasetScanner` accepts a streaming iterable and a feature extractor
such as `sae.encode`. It retains running moments, activation frequency, maxima,
and a bounded top-K heap per selected feature; it never stores the complete
activation dataset.

`build_dataset_manifest()` records sorted example ids, source revision,
deterministic train/validation/test assignments, and a SHA-256 fingerprint. The
split of an id depends only on that id and the seed, not iterable order.
`examples_for_split()` rejects examples absent from the manifest, making dataset
drift visible before SAE scanning or held-out autointerp evaluation. Input
tensors are deliberately not serialized in the manifest.

`build_feature_intervention()` edits selected encoded features and adds only the
decoded feature delta to the original hidden state. This prevents SAE
reconstruction error from being mistaken for the causal intervention.
`decompose_sae_features()` decodes one-hot feature directions, subtracts decoder
bias using the decoded zero vector, and projects the resulting directions into
vocabulary space. Its graph nodes remain `sae_feature` nodes and retain feature
ids for follow-up ablation or steering.

`FeatureAutointerpreter` accepts a provider implemented by the caller. Provider
training evidence and scoring evidence must be disjoint. A separate predictor
scores the generated explanation against held-out activations, reporting
correlation, normalized absolute error, and top-versus-bottom separation. LTTS
does not require a specific LLM service and does not treat generated labels as
validated explanations without this independent score.
`partition_feature_evidence()` selects training and held-out examples from the
manifest and rejects duplicate, unknown, identical, or undersized splits.

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

Occlusion, gradient-importance, and token-contributor artifacts first normalize
into `TokenAttributionResult`. The result states whether scores are signed
effects, absolute importance, or normalized importance and retains the exact
target, layer, method, and module provenance. `graph_from_token_attribution()`
then creates directed input-token-to-target edges without treating unlike score
semantics as interchangeable.

Gradient neuron tables normalize separately into
`ComponentAttributionResult`; component kind and index are preserved when
building neuron/head/feature-to-target graph edges. This prevents a neuron index
from being mistaken for a token position merely because both are table rows.

## SSM and multimodal research

`discover_research_adapter()` probes executable tensor boundaries instead of
matching model family names. It reports SSM state mixers, vision patch
boundaries, and fusion modules only when their required attributes or forward
arguments are actually present. Call `require()` before running a workflow that
depends on one of these capabilities.

`research_workflow_matrix()` produces an explicit supported/missing report for
each SSM, vision, or cross-modal workflow, including the discovered capture
paths. In particular, a Transformers Mamba mixer supports mixer-output
intervention but is not reported as supporting raw recurrent-state intervention.
Cross-modal discovery prefers modules that explicitly declare
`is_cross_attention` and removes parent forwarding proxies. This has been tested
with an actual tiny `VisionEncoderDecoderModel`, where the reported boundary is
the decoder cross-attention module rather than its decoder/block parents.

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

`ResearchStudyBuilder` combines manifests, scans, explanations, interventions,
and attribution graphs as named stages. Dependencies must already exist when a
stage is added, stage names are unique, cycles are rejected, and tensor outputs
cannot be retained accidentally. The resulting `ResearchStudyResult` can itself
be stored in an integrity-checked research bundle.
`ResearchStudyRunner` executes `ResearchStudyPlanStep` callbacks only after their
declared dependencies, passes dependency results by name, rejects unknown or
cyclic plans, and immediately converts every stage output to a mapping so model
tensors are not retained across a long study.

Two common plans are concrete rather than callback-only:

- `CausalStudyRunner` executes an isolated layer sweep and immediately converts
  it into a provenance-preserving attribution graph.
- `FeatureStudyRunner` verifies that examples exactly match a manifest, scans
  only its training split, partitions feature evidence through the same
  manifest, and independently scores the explanation on the test split.

Both return an ordinary `ResearchStudyResult`; neither depends on the UI or web
application.

### Compact circuit discovery

`discover_circuit()` reduces an `AttributionGraph` to a node-limited subgraph
connected to a chosen metric or output node. It expands backwards from that
target, prioritizes absolute signed edge strength, preserves provenance and
actionable `InterventionTargetRef` coordinates, and reports the estimated share
of candidate edge mass retained.

An optional validator receives the full and selected intervention-coordinate
sets. Its measured full/circuit effects and fidelity are stored separately from
the graph-based estimate. This keeps circuit extraction useful headlessly
without treating attribution weights alone as causal validation.

### Training-free recirculation

`RecirculationRunner` implements a reference sequential prefill for decoder-only
causal language models. A `RecirculationSpec` names a deep source and shallower
destination residual boundary, mixing coefficient, optional explicit beta,
L2/no normalization, and an optional early-token ramp. Each token is first
processed normally; its mixed destination state is then exposed while later
tokens are processed. Baseline and recirculated next-token likelihoods are
paired on the same input and returned as finite per-token NLL and perplexity.
`discover_recirculation_layers()` locates the model's longest ordered decoder
block list and returns explicit compatible layer paths; unsupported structures
receive a reason instead of failing later during an experiment.

The implementation follows Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny
Sawyer, Sunny Sanyal, and Rosanne Liu, ["Recirculation"
(2026)](https://arxiv.org/abs/2608.17981). LTTS provides an independent
PyTorch/Hugging Face reference implementation; it does not include or claim the
authors' source code.

`sweep()` evaluates an explicit, unique source/destination/coefficient grid and
selects the lowest measured perplexity while checking that every trial retained
the identical baseline. The result is versioned and JSON-safe. This reference
executor disables KV caching and recomputes prefixes for clarity and broad
Hugging Face compatibility; an optimized architecture adapter can preserve the
same result contract without changing research records.

`evaluate_dataset()` aggregates NLL by predicted-token count rather than giving
short and long documents equal weight. `sweep_dataset()` applies the same
explicit search space to all examples and retains each per-example result for
audit and stability analysis.
`recirculation_sweep_heatmap()` converts a complete fixed-parameter
source-by-destination grid into the existing labeled, zero-centered heatmap
payload. It rejects ambiguous or incomplete grids instead of inventing cells.

`generate()` performs paired deterministic greedy decoding and stores both token
trajectories plus every changed generated position. It uses the same sequential
state rule as likelihood evaluation, so generation can be tested without the
web application or an artifact renderer.

### Portable study artifacts and comparisons

`StudyArtifactRecord` is the versioned numeric observation contract shared by
headless studies and the web Research Study service. It records the metric and
value together with module, subtype, layer/component/token location, example,
seed, stage, run id, and JSON metadata. `StudyArtifactCollection` groups unique
records and optionally binds them to a dataset-manifest fingerprint.

`archive_research_study()` combines a `ResearchStudyResult`, its normalized
artifact collection, manifest, trial exports, and attribution graphs into a
`ResearchStudyArchive`. `bundle_study_archive()` stores that archive through the
existing integrity-checked `ResearchRunBundle`; `study_archive_from_bundle()`
restores and validates the typed archive after loading.

`compare_study_artifacts()` aligns observations by their semantic location and
method rather than transient artifact ids. It reports matched, appeared, and
disappeared effects, means and population standard deviations for both runs,
overall deltas, and paired delta mean/std for identical `(example_id, seed)`
trials. Manifest compatibility is explicit: `true`, `false`, or unknown when
either fingerprint is absent.

## Research Study web API

The additive `POST /api/research/studies/run` endpoint connects the browser UI
to reproducible inference trials without changing `/api/inference`. A request
contains a manifest identity, explicit examples, unique seeds, the currently
attached module specifications, and optional study metadata. Limits are 32
examples and 16 seeds per request.

Each example/seed pair runs as an isolated normal inference with its seed in the
run export. The service subscribes to that trial's private message-bus token,
captures only real visualization messages, and aggregates finite numeric
`score`, `effect`, `delta`, `attribution`, or `value` fields by layer. Results
include every trial export, artifact provenance, mean and population standard
deviation, and validated dataset/scan stages. Missing, textual, or non-finite
measurements never become candidates.

The response also includes the same versioned `StudyArtifactCollection` used by
the headless API, so a browser-created study can be archived or compared without
reverse-engineering its visualization envelopes.

Three additive JSON endpoints expose the portable workflow without server-side
path access:

- `POST /api/research/studies/archive` validates a `ResearchStudyArchive` and
  returns the standard checksummed bundle document.
- `POST /api/research/studies/restore` verifies a bundle checksum and returns
  the validated archive payload.
- `POST /api/research/studies/compare` compares two artifact collections and
  returns the headless comparison schema.

Clients may download or store the returned JSON themselves. These endpoints do
not read or write caller-selected server filesystem paths.

The aggregate is also emitted as one table artifact on the caller's original
client token, so the existing artifact panel renders it without a second output
system.

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

Schedules may additionally use `token_positions` (positions present in the
current layer input) and `output_token_positions` (the absolute token position
being predicted). During cached decoding, the first model call exposes all
prompt positions; later one-token calls expose their absolute generated-input
position. Resolved positions and the original attachment schedule are available
to the module through its event environment.

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
