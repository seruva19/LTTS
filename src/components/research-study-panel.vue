<script setup>
import { computed, ref, watch } from 'vue'
import axios from 'axios'
import { useLttsState } from '@/state'
import { useResearchStore } from '@/state/research-store'
import { scrollToLayerByName } from '@/composables/useLayerNavigation'
import { collectInferenceModules, MODEL_MODULES_KEY } from '@/utils/module-collection'
import { baseUrl } from '@/constants'

const state = useLttsState()
const research = useResearchStore()
const emit = defineEmits(['close'])
const isStudyRunning = ref(false)
const runError = ref('')

const layers = computed(() => state.blueprints.loadedBlueprint?.layers || [])
const maximumLayer = computed(() => Math.max(0, layers.value.length - 1))
const attachedModules = computed(() => collectInferenceModules({
  blueprint: state.blueprints.loadedBlueprint,
  moduleStore: state.modules
}))
const scopedModules = computed(() => {
  const requestedStart = Number(research.layerStart)
  const requestedEnd = Number(research.layerEnd)
  const first = Number.isFinite(requestedStart) ? requestedStart : 0
  const last = Number.isFinite(requestedEnd) ? requestedEnd : maximumLayer.value
  const start = Math.max(0, Math.min(first, last))
  const end = Math.min(maximumLayer.value, Math.max(first, last))
  const allowed = new Set(layers.value.slice(start, end + 1).flatMap(layer => [
    layer.fullPath,
    layer.name,
    layer.path
  ].filter(Boolean)))
  return Object.fromEntries(Object.entries(attachedModules.value).filter(([path]) => (
    path === MODEL_MODULES_KEY || allowed.has(path)
  )))
})
const hasAttachedAnalysis = computed(() => Object.values(scopedModules.value).some(items => items.length > 0))
const canRun = computed(() => (
  research.hasConfiguration &&
  layers.value.length > 0 &&
  hasAttachedAnalysis.value &&
  !state.ws.isInferenceRunning &&
  !isStudyRunning.value
))
const stageLabels = {
  dataset: 'Dataset',
  scan: 'Scan',
  inspect: 'Inspect',
  intervene: 'Intervene',
  compare: 'Compare'
}

watch(maximumLayer, (value) => {
  if (research.layerEnd === null || research.layerEnd > value) research.layerEnd = value
}, { immediate: true })

watch(
  () => state.ws.visualizations,
  (artifacts) => research.ingestArtifacts(artifacts, layers.value),
  { deep: true, immediate: true }
)

const runStudyStage = async () => {
  if (!canRun.value) return
  isStudyRunning.value = true
  runError.value = ''
  research.resetResults()
  research.setStage('scan')
  try {
    const seedCount = Math.max(1, Math.min(16, Number(research.seeds) || 1))
    const response = await axios.post(`${baseUrl}/api/research/studies/run`, {
      name: `${research.manifest} study`,
      model_id: state.blueprints.loadedBlueprint.source?.id,
      manifest: { name: research.manifest, split: research.split },
      examples: [{ id: 'current-prompt', prompt: state.ui.userPrompt }],
      seeds: Array.from({ length: seedCount }, (_, index) => index),
      parameters: { force_eager_attention: true },
      modules: scopedModules.value,
      client_token: state.ws.clientToken,
      metadata: {
        target: research.targetToken,
        baseline: research.baseline,
        layer_range: [research.layerStart, research.layerEnd]
      }
    })
    research.ingestStudyResult(response.data)
    research.setStage(research.candidates.length > 0 ? 'inspect' : 'scan')
  } catch (error) {
    runError.value = error?.response?.data?.detail || error?.message || String(error)
  } finally {
    isStudyRunning.value = false
  }
}

const inspectCandidate = (candidate) => {
  research.selectLayer(candidate.layerName)
  research.setStage('inspect')
  state.ui.setSelectedLayer(layers.value.find(layer => layer.name === candidate.layerName) || null)
  scrollToLayerByName(candidate.layerName)
}
</script>

<template>
  <section class="research-study" aria-label="Research Study controls">
    <header class="research-header">
      <div>
        <strong>Research Study</strong>
        <small>Guide a reproducible run through the model stack</small>
      </div>
      <button type="button" class="research-close" title="Close Research Study" @click="emit('close')">
        <VaIcon name="close" size="small" />
      </button>
    </header>

    <nav class="research-stages" aria-label="Study stages">
      <button
        v-for="(stage, index) in research.stages"
        :key="stage"
        type="button"
        :class="{ current: research.activeStage === stage, complete: index < research.activeStageIndex }"
        :title="stageLabels[stage]"
        @click="research.setStage(stage)"
      >
        <span>{{ String(index + 1).padStart(2, '0') }}</span>
      </button>
    </nav>

    <div class="research-section">
      <h3>Dataset</h3>
      <label>
        Manifest
        <input v-model.trim="research.manifest" type="text" placeholder="factual-recall-v1">
      </label>
      <label>
        Split
        <select v-model="research.split">
          <option value="train">Train</option>
          <option value="validation">Validation</option>
          <option value="test">Test</option>
        </select>
      </label>
    </div>

    <div class="research-section">
      <h3>Scan</h3>
      <label>
        Target token or label
        <input v-model.trim="research.targetToken" type="text" placeholder="Paris → France">
      </label>
      <div class="research-row">
        <label>
          First layer
          <input v-model.number="research.layerStart" type="number" min="0" :max="maximumLayer">
        </label>
        <label>
          Last layer
          <input v-model.number="research.layerEnd" type="number" min="0" :max="maximumLayer">
        </label>
      </div>
      <label>
        Seeds
        <input v-model.number="research.seeds" type="number" min="1" max="64">
      </label>
      <label>
        Baseline
        <select v-model="research.baseline">
          <option value="clean-corrupt">Clean / corrupted pairs</option>
          <option value="zero">Zero activation</option>
          <option value="shuffle">Shuffled control</option>
        </select>
      </label>
    </div>

    <button type="button" class="research-run" :disabled="!canRun" @click="runStudyStage">
      <VaIcon name="manage_search" size="small" />
      {{ isStudyRunning ? 'Scanning…' : 'Run attached analysis' }}
    </button>
    <p class="research-help">
      <template v-if="hasAttachedAnalysis">
        Runs the attached modules for every selected seed. Only artifacts with a layer path and numeric effect become candidates.
      </template>
      <template v-else>
        Attach an analysis module to at least one layer in this range before running the study.
      </template>
    </p>
    <p v-if="runError" class="research-error" role="alert">{{ runError }}</p>

    <div v-if="research.candidates.length > 0" class="research-section research-results">
      <h3>Candidates</h3>
      <button
        v-for="candidate in research.candidates"
        :key="candidate.layerName"
        type="button"
        :class="{ selected: research.selectedLayerName === candidate.layerName }"
        @click="inspectCandidate(candidate)"
      >
        <span>{{ candidate.layerName }}</span>
        <strong>{{ candidate.score >= 0 ? '+' : '' }}{{ candidate.score.toFixed(3) }}</strong>
      </button>
    </div>

    <div v-if="research.activeStage === 'compare'" class="research-section research-runs">
      <h3>Recorded runs</h3>
      <p v-if="research.runs.length === 0">Complete a scan to create a comparable run.</p>
      <div v-for="run in research.runs" :key="run.id" class="research-run-row">
        <span>{{ run.manifest }} · {{ run.split }}</span>
        <strong v-if="run.candidates[0]">
          {{ run.candidates[0].score >= 0 ? '+' : '' }}{{ run.candidates[0].score.toFixed(3) }}
        </strong>
      </div>
    </div>
  </section>
</template>

<style scoped>
.research-study { padding: 0 0.7rem 1rem; overflow-y: auto; height: calc(100vh - 118px); color: var(--va-text-primary); }
.research-header { display: flex; align-items: center; gap: 0.5rem; padding: 0.8rem 0; border-bottom: 1px solid var(--va-background-border); }
.research-header > div { display: flex; flex-direction: column; min-width: 0; }
.research-header strong { font-size: 0.82rem; font-weight: 600; }
.research-header small { color: var(--va-text-secondary); font-size: 0.68rem; line-height: 1.35; }
.research-close { margin-left: auto; border: 0; border-radius: 2px; background: transparent; color: var(--va-text-primary); padding: 0.25rem; cursor: pointer; }
.research-stages { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.25rem; padding: 0.75rem 0; }
.research-stages button { border: 0; border-top: 2px solid var(--va-background-border); border-radius: 0; background: transparent; color: var(--va-text-secondary); padding: 0.35rem 0 0; cursor: pointer; text-align: left; }
.research-stages button.complete { border-top-color: var(--va-success); }
.research-stages button.current { border-top-color: var(--va-primary); color: var(--va-primary); }
.research-section { border-top: 1px solid var(--va-background-border); padding: 0.75rem 0; }
.research-section h3 { font-size: 0.75rem; font-weight: 600; margin: 0 0 0.55rem; }
.research-section label { display: grid; gap: 0.25rem; margin-top: 0.55rem; color: var(--va-text-secondary); font-size: 0.68rem; }
.research-section input, .research-section select { width: 100%; border: 1px solid var(--va-background-border); border-radius: 2px; background: var(--va-background-secondary); color: var(--va-text-primary); padding: 0.4rem 0.45rem; font: inherit; }
.research-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; }
.research-run { display: flex; align-items: center; justify-content: center; gap: 0.35rem; width: 100%; border: 1px solid var(--va-primary); border-radius: 2px; background: var(--va-primary); color: var(--va-on-primary, white); padding: 0.5rem; cursor: pointer; }
.research-run:disabled { opacity: 0.5; cursor: not-allowed; }
.research-help { color: var(--va-text-secondary); font-size: 0.65rem; line-height: 1.4; margin: 0.45rem 0 0; }
.research-error { color: var(--va-danger); font-size: 0.65rem; line-height: 1.4; margin: 0.45rem 0 0; overflow-wrap: anywhere; }
.research-results button { width: 100%; display: flex; align-items: center; gap: 0.4rem; border: 1px solid var(--va-background-border); border-radius: 2px; background: var(--va-background-primary); color: var(--va-text-primary); padding: 0.45rem; margin-top: 0.35rem; cursor: pointer; text-align: left; }
.research-results button.selected { border-color: var(--va-primary); background: color-mix(in srgb, var(--va-primary) 7%, var(--va-background-primary)); }
.research-results button span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.research-results button strong { margin-left: auto; color: var(--va-primary); font-size: 0.7rem; }
.research-runs p { color: var(--va-text-secondary); font-size: 0.68rem; line-height: 1.4; }
.research-run-row { display: flex; align-items: center; gap: 0.4rem; padding: 0.45rem 0; border-top: 1px solid var(--va-background-border); font-size: 0.68rem; }
.research-run-row span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.research-run-row strong { margin-left: auto; color: var(--va-primary); }
</style>
