<script setup>
import { useLttsState } from '@/state'
import { computed, onMounted, ref } from 'vue'
import axios from 'axios'
import { baseUrl } from '@/constants'

const state = useLttsState()

// Module registry summary (fetched once on mount)
const registryStatus = ref(null)

onMounted(async () => {
  try {
    const response = await axios.get(`${baseUrl}/api/ltts/registry/status`, { timeout: 10000 })
    if (response.data?.registry_count !== undefined) {
      registryStatus.value = response.data
    }
  } catch (err) {
    console.debug('Registry status unavailable:', err?.message || err)
  }
})

const registrySummary = computed(() => {
  if (!registryStatus.value) return null
  const { registry_count, installed_count, updates_available } = registryStatus.value
  const parts = [`${registry_count} in registry`, `${installed_count} installed`]
  if (updates_available > 0) {
    parts.push(`${updates_available} update${updates_available > 1 ? 's' : ''} available`)
  }
  return parts.join(' · ')
})

const layersCount = computed(() => {
  return (state.blueprints.loadedBlueprint?.layers?.length) || 0
})

// Layer-level attachments (pre + post)
const preModulesCount = computed(() => {
  const layers = state.blueprints.loadedBlueprint?.layers || []
  return layers.reduce((sum, layer) => sum + (layer.preModules?.length || 0), 0)
})

const postModulesCount = computed(() => {
  const layers = state.blueprints.loadedBlueprint?.layers || []
  return layers.reduce((sum, layer) => sum + (layer.postModules?.length || 0), 0)
})

const layerAttachmentsCount = computed(() => preModulesCount.value + postModulesCount.value)

const layersWithModulesCount = computed(() => {
  const layers = state.blueprints.loadedBlueprint?.layers || []
  return layers.reduce((count, layer) => {
    const hasAny = (layer.preModules?.length || 0) + (layer.postModules?.length || 0) > 0
    return count + (hasAny ? 1 : 0)
  }, 0)
})

const globalModulesCount = computed(() => {
  return (state.modules.attachedModules || []).length
})

const modelModulesCount = computed(() => {
  return (state.blueprints.loadedBlueprint?.modules?.length) || 0
})

const visualizationsCount = computed(() => {
  return (state.ws.visualizations || []).length
})

const statusEmoji = computed(() => {
  return state.ws.isInferenceRunning ? '🚀' : '🧪'
})

// Layer-by-layer inference progress
const currentProcessingLayer = computed(() => state.ws.currentProcessingLayer)
const showLayerProgress = computed(() =>
  state.ws.isInferenceRunning &&
  currentProcessingLayer.value >= 0 &&
  layersCount.value > 0
)
// Display as 1-based "X / N"
const currentLayerDisplay = computed(() =>
  Math.min(currentProcessingLayer.value + 1, layersCount.value)
)
const layerProgressPercent = computed(() => {
  if (!layersCount.value) return 0
  return Math.round((currentLayerDisplay.value / layersCount.value) * 100)
})

// Selected layer info
const selectedLayerName = computed(() => state.ui.selectedLayerName || null)
const selectedLayerId = computed(() => state.ui.selectedLayerId || null)
</script>

<template>
  <div class="status-bar">
    <div class="stat" v-if="registrySummary" :title="registryStatus.registry_url">
      <span class="emoji">📦</span>
      <span class="label">Registry</span>
      <span class="value">{{ registrySummary }}</span>
    </div>
    <div class="stat">
      <span class="emoji">🧱</span>
      <span class="label">Layers</span>
      <span class="value">{{ layersCount }}</span>
    </div>
    <div class="stat">
      <span class="emoji">🔗</span>
      <span class="label">Layer Attachments</span>
      <span class="value">{{ layerAttachmentsCount }}</span>
    </div>
    <div class="stat">
      <span class="emoji">🔬</span>
      <span class="label">Layers w/ Modules</span>
      <span class="value">{{ layersWithModulesCount }}</span>
    </div>
    <div class="stat">
      <span class="emoji">📊</span>
      <span class="label">All Visualizations</span>
      <span class="value">{{ visualizationsCount }}</span>
    </div>

    <div class="stat" v-if="selectedLayerName && selectedLayerId">
      <span class="emoji">🔎</span>
      <span class="label">Selected Layer</span>
      <span class="value">{{ selectedLayerName }}</span>
    </div>

    <div class="stat layer-progress" v-if="showLayerProgress">
      <span class="emoji">⏳</span>
      <span class="label">Layer</span>
      <span class="value">{{ currentLayerDisplay }} / {{ layersCount }}</span>
      <VaProgressBar
        class="layer-progress-bar"
        :model-value="layerProgressPercent"
        size="small"
      />
    </div>

    <div class="stat state">
      <span class="emoji">{{ statusEmoji }}</span>
      <span class="label">{{ state.ws.isInferenceRunning ? 'Running' : 'Idle' }}</span>
    </div>
  </div>

</template>

<style scoped>
.status-bar {
  position: fixed;
  bottom: 0;
  left: 90px;
  width: calc(100% - 155px);
  height: 26px;
  padding: 3px 8px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  background: var(--va-background-element);
  z-index: 1000;
}

.stat {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--va-text-secondary);
}

.stat .value {
  font-weight: 600;
  color: var(--va-text-primary);
}

.stat.state .label {
  letter-spacing: -0.2px;
}

.emoji {
  font-size: 12px;
  line-height: 1;
}

.layer-progress-bar {
  width: 80px;
}
</style>
