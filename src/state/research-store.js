import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

const STAGES = ['dataset', 'scan', 'inspect', 'intervene', 'compare']

const finiteNumber = (value) => {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

const artifactScore = (artifact) => {
  const payload = artifact?.data?.visualization || artifact?.data || {}
  const candidates = [
    payload.score,
    payload.effect,
    payload.delta,
    payload.attribution,
    payload.value,
    artifact?.stats?.mean
  ]
  for (const candidate of candidates) {
    const score = finiteNumber(candidate)
    if (score !== null) return score
  }
  return null
}

const normalizePath = (value) => String(value || '').trim().toLowerCase()
const cloneSerializable = (value) => value == null ? value : JSON.parse(JSON.stringify(value))
const pathsMatch = (left, right) => {
  const first = normalizePath(left)
  const second = normalizePath(right)
  return Boolean(first && second && (
    first === second ||
    first.endsWith(`.${second}`) ||
    second.endsWith(`.${first}`) ||
    first.includes(`.${second}.`) ||
    second.includes(`.${first}.`)
  ))
}

export const useResearchStore = defineStore('research', () => {
  const isOpen = ref(false)
  const activeStage = ref('dataset')
  const manifest = ref('')
  const split = ref('validation')
  const seeds = ref(8)
  const baseline = ref('clean-corrupt')
  const targetToken = ref('')
  const layerStart = ref(0)
  const layerEnd = ref(null)
  const selectedLayerName = ref(null)
  const candidates = ref([])
  const lastRunAt = ref(null)
  const runs = ref([])
  const baselineRunId = ref(null)
  const candidateRunId = ref(null)
  const comparison = ref(null)

  const activeStageIndex = computed(() => STAGES.indexOf(activeStage.value))
  const hasConfiguration = computed(() => manifest.value.trim().length > 0)

  function toggle() {
    isOpen.value = !isOpen.value
  }

  function open() {
    isOpen.value = true
  }

  function close() {
    isOpen.value = false
  }

  function setStage(stage) {
    if (STAGES.includes(stage)) activeStage.value = stage
  }

  function selectLayer(layerName) {
    selectedLayerName.value = layerName || null
  }

  function ingestArtifacts(artifacts, layers) {
    const layerList = Array.isArray(layers) ? layers : []
    const byLayer = new Map()
    for (const artifact of artifacts || []) {
      const path = normalizePath(
        artifact?.normalizedLayerPath || artifact?.modulePath || artifact?.layerPath
      )
      const layer = layerList.find((item) => {
        const name = normalizePath(item?.name)
        return name && (
          path === name ||
          path.endsWith(`.${name}`) ||
          path.startsWith(`${name}.`) ||
          path.includes(`.${name}.`)
        )
      })
      const score = artifactScore(artifact)
      if (!layer || score === null) continue
      const previous = byLayer.get(layer.name)
      if (!previous || Math.abs(score) > Math.abs(previous.score)) {
        byLayer.set(layer.name, {
          layerName: layer.name,
          layerId: layer.id || null,
          score,
          artifactId: artifact.id || null,
          artifactType: artifact.type || 'visualization'
        })
      }
    }
    candidates.value = [...byLayer.values()].sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
    if (candidates.value.length > 0 && !selectedLayerName.value) {
      selectedLayerName.value = candidates.value[0].layerName
    }
    lastRunAt.value = candidates.value.length > 0 ? new Date().toISOString() : lastRunAt.value
    return candidates.value
  }

  function candidateForLayer(layerName) {
    return candidates.value.find(candidate => candidate.layerName === layerName) || null
  }

  function comparisonForLayer(layerName) {
    return comparison.value?.points?.find(point => pathsMatch(point.layer_path, layerName)) || null
  }

  function resetResults() {
    candidates.value = []
    selectedLayerName.value = null
    lastRunAt.value = null
    comparison.value = null
  }

  function recordRun() {
    if (candidates.value.length === 0) return null
    const run = {
      id: `study-${Date.now()}-${runs.value.length + 1}`,
      createdAt: new Date().toISOString(),
      manifest: manifest.value,
      split: split.value,
      seeds: seeds.value,
      baseline: baseline.value,
      targetToken: targetToken.value,
      layerRange: [layerStart.value, layerEnd.value],
      candidates: candidates.value.map(candidate => ({ ...candidate }))
    }
    runs.value.unshift(run)
    runs.value = runs.value.slice(0, 12)
    return run
  }

  function ingestStudyResult(result) {
    const rows = Array.isArray(result?.candidates) ? result.candidates : []
    candidates.value = rows
      .map(row => ({
        layerName: String(row.layer || ''),
        layerId: null,
        score: finiteNumber(row.mean_effect),
        standardDeviation: finiteNumber(row.std_effect) ?? 0,
        observations: Number(row.observations) || 0,
        artifactId: result.study_id || null,
        artifactType: 'research_study'
      }))
      .filter(candidate => candidate.layerName && candidate.score !== null)
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
    if (candidates.value.length > 0) selectedLayerName.value = candidates.value[0].layerName
    lastRunAt.value = result?.created_at || null
    const run = {
      id: result?.study_id || `study-${Date.now()}`,
      createdAt: result?.created_at || new Date().toISOString(),
      manifest: result?.manifest?.name || manifest.value,
      split: result?.manifest?.split || split.value,
      seeds: Array.isArray(result?.seeds) ? [...result.seeds] : [],
      candidates: candidates.value.map(candidate => ({ ...candidate })),
      trialCount: Array.isArray(result?.trials) ? result.trials.length : 0,
      artifacts: cloneSerializable(result?.artifacts || null)
    }
    runs.value.unshift(run)
    runs.value = runs.value.slice(0, 12)
    candidateRunId.value = run.id
    if (runs.value.length > 1) baselineRunId.value = runs.value[1].id
    return run
  }

  function ingestComparison(result) {
    const points = Array.isArray(result?.points) ? result.points
      .map(point => ({
        ...cloneSerializable(point),
        delta: finiteNumber(point?.delta),
        baseline_mean: finiteNumber(point?.baseline_mean),
        candidate_mean: finiteNumber(point?.candidate_mean),
        paired_delta_std: finiteNumber(point?.paired_delta_std)
      }))
      .filter(point => point.layer_path)
      : []
    points.sort((left, right) => Math.abs(right.delta ?? 0) - Math.abs(left.delta ?? 0))
    comparison.value = {
      ...cloneSerializable(result || {}),
      points
    }
    const first = points[0]
    if (first?.layer_path) selectedLayerName.value = first.layer_path
    activeStage.value = 'compare'
    return comparison.value
  }

  function clearComparison() {
    comparison.value = null
    if (activeStage.value === 'compare') activeStage.value = candidates.value.length ? 'inspect' : 'scan'
  }

  function runById(runId) {
    return runs.value.find(run => run.id === runId) || null
  }

  return {
    stages: STAGES,
    isOpen,
    activeStage,
    activeStageIndex,
    manifest,
    split,
    seeds,
    baseline,
    targetToken,
    layerStart,
    layerEnd,
    selectedLayerName,
    candidates,
    lastRunAt,
    runs,
    baselineRunId,
    candidateRunId,
    comparison,
    hasConfiguration,
    toggle,
    open,
    close,
    setStage,
    selectLayer,
    ingestArtifacts,
    candidateForLayer,
    comparisonForLayer,
    resetResults,
    recordRun,
    ingestStudyResult,
    ingestComparison,
    clearComparison,
    runById
  }
})
