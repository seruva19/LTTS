<script setup>
import { useLttsState } from '@/state'
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'

const state = useLttsState()

// Timeline state
const outputEvents = ref([])
const hoveredEvent = ref(null)
const maxEvents = 1000 // Maximum events to keep in memory
const isCollapsed = ref(true)
const isTimelineVisible = ref(true)
const activeMode = ref('artifacts')
const panelWidth = ref(300)
const selectedEvent = ref(null)
let resizing = false

// Window dimensions for positioning
const windowWidth = ref(1200)
const windowHeight = ref(800)

// Computed properties
const isInferenceRunning = computed(() => state.ws.isInferenceRunning)
const hasEvents = computed(() => outputEvents.value.length > 0)
const logEvents = computed(() => (state.ws.visualizationEvents || []).map(toOutputEvent))
const displayedEvents = computed(() => activeMode.value === 'artifacts' ? outputEvents.value : logEvents.value)
const autoScrollToTarget = computed(() => (
  state.settings?.settings?.controls?.auto_scroll_to_target !== false
))

// Listen for timeline visibility changes
const handleTimelineVisibilityToggle = (event) => {
  isTimelineVisible.value = event.detail.isVisible
}

// Event processing
const addOutputEvent = (event) => {
  const outputEvent = {
    id: Date.now() + Math.random(),
    timestamp: new Date(),
    layerName: event.layer_name || 'unknown',
    eventType: event.event_type || 'layer_output',
    dataType: event.data_type || 'tensor',
    shape: event.shape || [],
    stats: event.stats || {},
    metadata: event.metadata || {},
    preview: event.preview || null,
    moduleId: event.module_id || null
  }

  outputEvents.value.unshift(outputEvent) // Add to beginning

  // Limit events to prevent memory issues
  if (outputEvents.value.length > maxEvents) {
    outputEvents.value = outputEvents.value.slice(0, maxEvents)
  }
}

const clearTimeline = () => {
  outputEvents.value = []
}

const formatTimestamp = (timestamp) => {
  return timestamp.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3
  })
}

const formatShape = (shape) => {
  if (!shape || shape.length === 0) return 'scalar'
  return `[${shape.join(', ')}]`
}

const formatPreview = (event) => {
  if (event.preview) return event.preview
  if (event.shape && event.shape.length > 0) return formatShape(event.shape)
  return event.eventType
}

const formatStats = (stats) => {
  if (!stats || Object.keys(stats).length === 0) return ''

  const parts = []
  if (stats.mean !== undefined) parts.push(`μ=${stats.mean.toFixed(3)}`)
  if (stats.std !== undefined) parts.push(`σ=${stats.std.toFixed(3)}`)
  if (stats.min !== undefined) parts.push(`min=${stats.min.toFixed(3)}`)
  if (stats.max !== undefined) parts.push(`max=${stats.max.toFixed(3)}`)

  return parts.join(', ')
}

const getEventColor = (eventType) => {
  const colors = {
    'layer_output': 'var(--va-primary)',
    'module_output': 'var(--va-success)',
    'attention_output': 'var(--va-info)',
    'embedding_output': 'var(--va-warning)',
    'final_output': 'var(--va-danger)'
  }
  return colors[eventType] || 'var(--va-primary)'
}

const getDataTypeIcon = (dataType) => {
  const icons = {
    'tensor': 'grid_4x4',
    'scalar': 'tag',
    'vector': 'trending_up',
    'matrix': 'view_comfy',
    'image': 'image',
    'text': 'text_fields'
  }
  return icons[dataType] || 'data_object'
}

// Convert visualization data to output timeline events
function toOutputEvent(visualization) {
  return {
    id: visualization.id,
    timestamp: visualization.timestamp,
    layerName: extractLayerName(visualization.modulePath) || visualization.module || 'unknown',
    layerPath: visualization.modulePath || null,
    eventType: visualization.type || 'visualization',
    dataType: getDataTypeFromVisualization(visualization),
    shape: extractShapeFromVisualization(visualization),
    stats: extractStatsFromVisualization(visualization),
    metadata: {
      module: visualization.module,
      modulePath: visualization.modulePath,
      eventType: visualization.eventType,
      params: visualization.params,
      updateCount: visualization.updateCount || 1
    },
    data: visualization.data,
    preview: createPreviewFromVisualization(visualization),
    moduleId: visualization.module
  }
}

const processVisualizationEvent = (visualization) => {
  const outputEvent = toOutputEvent(visualization)
  outputEvents.value.unshift(outputEvent)

  // Limit events to prevent memory issues
  if (outputEvents.value.length > maxEvents) {
    outputEvents.value = outputEvents.value.slice(0, maxEvents)
  }
}

const toggleCollapsed = () => {
  isCollapsed.value = !isCollapsed.value
  document.dispatchEvent(new CustomEvent('artifact-panel-resize', {
    detail: { width: isCollapsed.value ? 50 : panelWidth.value }
  }))
}

const startResize = (event) => {
  if (isCollapsed.value) return
  resizing = true
  event.preventDefault()
}
const resizePanel = (event) => {
  if (!resizing) return
  panelWidth.value = Math.min(520, Math.max(240, window.innerWidth - event.clientX))
  document.dispatchEvent(new CustomEvent('artifact-panel-resize', {
    detail: { width: isCollapsed.value ? 50 : panelWidth.value }
  }))
}
const stopResize = () => { resizing = false }

const resolveLayerTarget = (event) => {
  if (!event) return null
  const metadata = event.metadata || {}
  return (
    event.layerPath ||
    metadata.modulePath ||
    metadata.layer_path ||
    metadata.layerPath ||
    event.layerName ||
    null
  )
}

const scrollToEventLayer = (event) => {
  if (!autoScrollToTarget.value) {
    return
  }
  const targetLayer = resolveLayerTarget(event)
  if (!targetLayer) {
    return
  }
  const candidates = [
    targetLayer,
    targetLayer.replace(/^original_model\.model\./, ''),
    targetLayer.replace(/^original_model\./, ''),
    targetLayer.replace(/^model\./, '')
  ]
  let layerElement = null
  for (const candidate of candidates) {
    layerElement = document.querySelector(`[data-layer-name="${candidate}"]`)
    if (layerElement) {
      break
    }
  }
  if (!layerElement) {
    return
  }

  let container = state.ui.scrollableRef
  container = container || document.querySelector('.layers-container') ||
    document.querySelector('.workspace') ||
    document.querySelector('[data-scrollable]') ||
    document.querySelector('.main-content') ||
    document.documentElement

  if (!container) {
    return
  }

  const containerRect = container.getBoundingClientRect()
  const elementRect = layerElement.getBoundingClientRect()
  const scrollTop = container.scrollTop + elementRect.top - containerRect.top - 20

  container.scrollTo({
    top: Math.max(0, scrollTop),
    behavior: 'smooth'
  })
}

// Helper functions
const extractLayerName = (modulePath) => {
  if (!modulePath) return null
  // Extract layer name from module path (e.g., "model.layers.0.self_attn" -> "self_attn")
  const parts = modulePath.split('.')
  return parts[parts.length - 1]
}

const getDataTypeFromVisualization = (visualization) => {
  const type = visualization.type?.toLowerCase() || ''
  if (type.includes('image') || type.includes('heatmap')) return 'image'
  if (type.includes('attention') || type.includes('attn')) return 'tensor'
  if (type.includes('embedding')) return 'vector'
  if (type.includes('scalar')) return 'scalar'
  if (type.includes('table')) return 'table'
  if (type.includes('histogram')) return 'histogram'
  return 'tensor'
}

const artifactPayload = (event) => event?.data?.visualization || event?.data || null

const previewImageSource = (event) => {
  const payload = artifactPayload(event)
  if (typeof payload === 'string') {
    return payload.startsWith('data:image') || payload.startsWith('http') ? payload : ''
  }
  if (!payload || typeof payload !== 'object') return ''
  const raw = payload.base64 ?? payload.url ?? payload.src ?? payload.image ??
    (payload.type === 'image' ? payload.data : null)
  if (typeof raw !== 'string') return ''
  return raw.startsWith('data:') || raw.startsWith('http') ? raw : `data:image/png;base64,${raw}`
}

const previewTable = (event) => {
  const payload = artifactPayload(event)
  return payload?.type === 'table' && Array.isArray(payload.rows) ? payload : null
}

const previewHeatmap = (event) => {
  const payload = artifactPayload(event)
  return payload?.type === 'heatmap' && Array.isArray(payload.values) ? payload : null
}

const previewText = (event) => {
  const payload = artifactPayload(event)
  return payload?.type === 'text' ? (payload.content ?? '') : null
}

const heatmapGridStyle = (data) => ({
  gridTemplateColumns: `repeat(${Math.min(data?.dimensions?.cols || data?.values?.[0]?.length || 8, 32)}, 1fr)`
})

const heatmapColor = (value, values) => {
  if (typeof value !== 'number' || !Array.isArray(values)) return 'var(--va-background-element)'
  const flat = values.flat().filter(Number.isFinite)
  if (!flat.length) return 'var(--va-background-element)'
  const min = Math.min(...flat)
  const range = Math.max(...flat) - min || 1
  const normalized = (value - min) / range
  const red = Math.round(255 * normalized)
  const blue = Math.round(255 * (1 - normalized))
  return `rgb(${red}, 80, ${blue})`
}

const extractShapeFromVisualization = (visualization) => {
  const data = visualization.data
  if (!data) return []

  // Try to extract shape from different data structures
  if (data.shape) return data.shape
  if (data.data?.shape) return data.data.shape
  if (data.layer_info?.output_shape) return data.layer_info.output_shape
  if (data.output_shape) return data.output_shape
  if (data.type === 'table' && Array.isArray(data.rows)) {
    return [data.rows.length, Array.isArray(data.headers) ? data.headers.length : (data.rows[0]?.length || 0)]
  }

  // For 2D visualizations, infer shape from data
  if (data.data && Array.isArray(data.data)) {
    if (Array.isArray(data.data[0])) {
      return [data.data.length, data.data[0].length]
    }
    return [data.data.length]
  }

  return []
}

const extractStatsFromVisualization = (visualization) => {
  const data = visualization.data
  if (!data) return {}

  // Try to extract statistics from different data structures
  const stats = {}

  if (data.stats) {
    Object.assign(stats, data.stats)
  } else if (data.data?.stats) {
    Object.assign(stats, data.data.stats)
  }

  // Extract basic stats from data arrays
  if (!stats.mean && data.data && Array.isArray(data.data)) {
    const flatArray = data.data.flat(100) // Use reasonable depth instead of Infinity
    if (flatArray.length > 0 && typeof flatArray[0] === 'number') {
      stats.mean = flatArray.reduce((a, b) => a + b, 0) / flatArray.length
      stats.min = Math.min(...flatArray)
      stats.max = Math.max(...flatArray)

      // Calculate standard deviation
      const variance = flatArray.reduce((a, b) => a + Math.pow(b - stats.mean, 2), 0) / flatArray.length
      stats.std = Math.sqrt(variance)
    }
  }

  return stats
}

const createPreviewFromVisualization = (visualization) => {
  const data = visualization.data
  if (!data) return null

  // Create a text preview based on visualization type
  const type = visualization.type?.toLowerCase() || ''

  if (type.includes('scalar') && data.value !== undefined) {
    return `Value: ${data.value.toFixed(6)}`
  }

  if (data.stats) {
    const parts = []
    if (data.stats.mean !== undefined) parts.push(`μ=${data.stats.mean.toFixed(3)}`)
    if (data.stats.std !== undefined) parts.push(`σ=${data.stats.std.toFixed(3)}`)
    return parts.join(', ')
  }

  // Generic preview for unknown types
  if (type) {
    return `${type} data`
  }

  return 'Visualization data'
}

const syncVisualizationEvents = () => {
  outputEvents.value = []
  state.ws.visualizations.slice().reverse().forEach(viz => processVisualizationEvent(viz))
}

// Keep the artifact rail synchronized when an existing live artifact is
// replaced as well as when a new subtype is added.
const setupVisualizationWatcher = () => {
  return watch(
    () => state.ws.visualizations.map(viz => viz.id).join('|'),
    syncVisualizationEvents,
  )
}

// Window resize handler
const updateWindowDimensions = () => {
  if (typeof window !== 'undefined') {
    windowWidth.value = window.innerWidth
    windowHeight.value = window.innerHeight
  }
}

// Lifecycle
let stopVizWatcher
let handleInferenceStart = null
onMounted(() => {
  // Initialize window dimensions
  updateWindowDimensions()

  // Initialize with existing visualizations
  syncVisualizationEvents()

  // Set up watcher for new visualizations
  stopVizWatcher = setupVisualizationWatcher()

  // Clear timeline when inference starts
  handleInferenceStart = () => {
    clearTimeline()
  }

  // Listen to inference events
  state.ws.addEventListener('inference_start', handleInferenceStart)

  // Listen for timeline visibility changes
  document.addEventListener('timeline-visibility-toggle', handleTimelineVisibilityToggle)

  // Listen for window resize
  window.addEventListener('resize', updateWindowDimensions)
  window.addEventListener('mousemove', resizePanel)
  window.addEventListener('mouseup', stopResize)
  document.dispatchEvent(new CustomEvent('artifact-panel-resize', { detail: { width: panelWidth.value } }))
})

onUnmounted(() => {
  if (stopVizWatcher) {
    stopVizWatcher()
  }
  if (handleInferenceStart) {
    state.ws.removeEventListener('inference_start', handleInferenceStart)
  }
  document.removeEventListener('timeline-visibility-toggle', handleTimelineVisibilityToggle)
  if (typeof window !== 'undefined') {
    window.removeEventListener('resize', updateWindowDimensions)
    window.removeEventListener('mousemove', resizePanel)
    window.removeEventListener('mouseup', stopResize)
  }
})
</script>

<template>
  <aside
    v-show="isTimelineVisible"
    class="output-timeline artifact-panel"
    :class="{ collapsed: isCollapsed }"
    :style="{ width: `${isCollapsed ? 50 : panelWidth}px` }"
    aria-label="Output artifacts"
  >
    <div class="resize-handle" @mousedown="startResize" />
    <header class="artifact-header">
      <button class="collapse-toggle" :title="isCollapsed ? 'Expand artifacts panel' : 'Collapse artifacts panel'" @click="toggleCollapsed">
        <VaIcon :name="isCollapsed ? 'chevron_left' : 'chevron_right'" />
      </button>
      <template v-if="!isCollapsed">
        <div>
          <strong>Outputs</strong>
          <small>{{ outputEvents.length }} artifacts</small>
        </div>
      </template>
    </header>

    <div v-if="!isCollapsed" class="mode-tabs" role="tablist" aria-label="Output view">
      <button :class="{ active: activeMode === 'artifacts' }" @click="activeMode = 'artifacts'">Artifacts</button>
      <button :class="{ active: activeMode === 'events' }" @click="activeMode = 'events'">Event log <span>{{ logEvents.length }}</span></button>
    </div>

    <div v-if="displayedEvents.length === 0" class="empty-state">
      <VaIcon :name="isInferenceRunning ? 'hourglass_top' : 'auto_awesome'" size="large" class="empty-icon" />
      <div v-if="!isCollapsed" class="empty-text">{{ isInferenceRunning ? 'Waiting for module outputs…' : 'Run inference to generate artifacts' }}</div>
    </div>

    <div v-else class="timeline-events artifact-list">
      <button
        v-for="event in displayedEvents"
        :key="event.id"
        class="timeline-event artifact-card"
        :class="{ selected: selectedEvent?.id === event.id }"
        :title="`${event.metadata.module || event.layerName} · ${event.eventType}`"
        @click="selectedEvent = event; scrollToEventLayer(event)"
      >
        <div class="event-indicator" :style="{ backgroundColor: getEventColor(event.eventType) }">
          <VaIcon :name="getDataTypeIcon(event.dataType)" size="14px" color="textInverted" />
        </div>
        <div v-if="!isCollapsed" class="artifact-copy">
          <strong>{{ event.metadata.module || event.layerName }}</strong>
          <span>{{ event.eventType }}</span>
          <small>{{ event.layerPath || 'model output' }}</small>
        </div>
        <span v-if="!isCollapsed && event.metadata.updateCount > 1" class="update-badge">updated {{ event.metadata.updateCount }}×</span>
      </button>
    </div>

    <section v-if="!isCollapsed && selectedEvent" class="artifact-preview">
      <header>
        <strong>Preview</strong>
        <button title="Close preview" @click="selectedEvent = null"><VaIcon name="close" /></button>
      </header>
      <div class="preview-content">
        <img
          v-if="previewImageSource(selectedEvent)"
          :src="previewImageSource(selectedEvent)"
          :alt="`${selectedEvent.metadata.module || 'Module'} ${selectedEvent.eventType}`"
          class="artifact-image"
        />
        <table v-else-if="previewTable(selectedEvent)" class="artifact-table">
          <thead><tr><th v-for="header in previewTable(selectedEvent).headers" :key="header">{{ header }}</th></tr></thead>
          <tbody><tr v-for="(row, rowIndex) in previewTable(selectedEvent).rows" :key="rowIndex"><td v-for="(cell, cellIndex) in row" :key="cellIndex">{{ cell }}</td></tr></tbody>
        </table>
        <div v-else-if="previewHeatmap(selectedEvent)" class="artifact-heatmap" :style="heatmapGridStyle(previewHeatmap(selectedEvent))">
          <template v-for="(row, rowIndex) in previewHeatmap(selectedEvent).values.slice(0, 32)" :key="rowIndex">
            <span
              v-for="(cell, cellIndex) in row.slice(0, 32)"
              :key="cellIndex"
              :style="{ backgroundColor: heatmapColor(cell, previewHeatmap(selectedEvent).values) }"
              :title="`[${rowIndex},${cellIndex}]: ${cell}`"
            />
          </template>
        </div>
        <pre v-else-if="previewText(selectedEvent) !== null" class="artifact-text">{{ previewText(selectedEvent) }}</pre>
        <template v-else>{{ formatPreview(selectedEvent) }}</template>
      </div>
      <div class="detail-row"><span>Type</span><code>{{ selectedEvent.eventType }}</code></div>
      <div class="detail-row"><span>Shape</span><code>{{ formatShape(selectedEvent.shape) }}</code></div>
      <div class="detail-row"><span>Updated</span><code>{{ selectedEvent.metadata.updateCount || 1 }}×</code></div>
    </section>
  </aside>
</template>

<style scoped>
/* Keyboard focus must be visible on the clickable containers below. */
[role='button']:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.output-timeline {
  width: 50px; /* Match transformer controls width */
  background: var(--va-background-element);
  border-left: 1px solid var(--va-background-border);
  height: calc(100vh - 40px);
  overflow: hidden;
  position: fixed;
  right: 0;
  top: 40px;
  z-index: 40;
  display: flex;
  flex-direction: column;
}

.artifact-image {
  display: block;
  width: 100%;
  max-height: 42vh;
  object-fit: contain;
  background: #fff;
  border-radius: 6px;
}
.artifact-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.75rem;
}
.artifact-table th,
.artifact-table td {
  padding: 0.3rem;
  border: 1px solid var(--va-background-border);
  text-align: left;
}
.artifact-heatmap {
  display: grid;
  gap: 1px;
  min-height: 140px;
}
.artifact-heatmap span { min-width: 3px; min-height: 3px; }
.artifact-text {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.header-icon {
  color: var(--va-text-primary);
}

.clear-btn {
  display: none; /* Hide clear button in narrow mode */
}

.panel-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1rem 0.25rem; /* Reduced padding */
  color: var(--va-text-secondary);
  text-align: center;
}

.empty-icon {
  margin-bottom: 0.5rem;
  opacity: 0.5;
}

.empty-text {
  font-size: 10px; /* Smaller text */
  font-weight: 600;
  margin-bottom: 0.25rem;
  display: none; /* Hide text in narrow mode */
}

.empty-subtitle {
  display: none; /* Hide subtitle in narrow mode */
}

.timeline-events {
  flex: 1;
  overflow-y: auto;
  padding: 4px 2px; /* Reduced padding */
}

.timeline-event {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2px;
  margin-bottom: 2px;
  background: var(--va-background-secondary);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid transparent;
  gap: 1px;
}

.timeline-event:hover {
  background: var(--va-background-primary);
  border-color: var(--va-primary);
  transform: translateX(-2px);
}

.timeline-event.hovered {
  background: var(--va-background-primary);
  border-color: var(--va-primary);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.event-indicator {
  width: 16px; /* Smaller indicator */
  height: 16px;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.event-preview {
  font-size: 6px; /* Very small text */
  color: var(--va-text-primary);
  font-family: monospace;
  text-align: center;
  line-height: 1.1;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  display: none; /* Hide preview text in narrow mode */
}

.collapsed-content {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.collapsed-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  color: var(--va-text-secondary);
}

.event-count {
  font-size: 10px;
  font-weight: 600;
}

.hover-details {
  position: fixed;
  width: 320px;
  background: var(--va-background-primary);
  border: 1px solid var(--va-primary);
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  max-height: 80vh;
  overflow-y: auto;
}

.details-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--va-background-border);
  background: var(--va-background-secondary);
}

.details-icon {
  color: var(--va-primary);
}

.details-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--va-text-primary);
  flex: 1;
}

.close-btn {
  color: var(--va-text-secondary) !important;
}

.details-content {
  padding: 12px;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  border-bottom: 1px solid var(--va-background-border);
}

.detail-row:last-child {
  border-bottom: none;
}

.detail-label {
  font-size: 12px;
  color: var(--va-text-secondary);
  font-weight: 600;
}

.detail-value {
  font-size: 12px;
  color: var(--va-text-primary);
  font-family: monospace;
  text-align: right;
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
}

.detail-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--va-background-border);
}

.section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--va-text-primary);
  margin-bottom: 8px;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-label {
  font-size: 11px;
  color: var(--va-text-secondary);
}

.stat-value {
  font-size: 11px;
  color: var(--va-text-primary);
  font-family: monospace;
}

.preview-content {
  font-size: 11px;
  color: var(--va-text-primary);
  background: var(--va-background-secondary);
  padding: 8px;
  border-radius: 4px;
  font-family: monospace;
  white-space: pre-wrap;
  max-height: 100px;
  overflow-y: auto;
}

.metadata-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metadata-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metadata-key {
  font-size: 11px;
  color: var(--va-text-secondary);
  font-weight: 600;
}

.metadata-value {
  font-size: 11px;
  color: var(--va-text-primary);
  text-align: right;
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Scrollbar styling */
.timeline-events::-webkit-scrollbar {
  width: 6px;
}

.timeline-events::-webkit-scrollbar-track {
  background: var(--va-background-secondary);
}

.timeline-events::-webkit-scrollbar-thumb {
  background: var(--va-background-border);
  border-radius: 3px;
}

.timeline-events::-webkit-scrollbar-thumb:hover {
  background: var(--va-primary);
}

.hover-details::-webkit-scrollbar {
  width: 6px;
}

.hover-details::-webkit-scrollbar-track {
  background: var(--va-background-secondary);
}

.hover-details::-webkit-scrollbar-thumb {
  background: var(--va-background-border);
  border-radius: 3px;
}

.hover-details::-webkit-scrollbar-thumb:hover {
  background: var(--va-primary);
}

/* Artifact-first panel */
.artifact-panel {
  width: 300px;
  min-width: 50px;
  box-shadow: -8px 0 24px rgba(15, 23, 42, 0.08);
  transition: width 160ms ease;
}
.resize-handle { position: absolute; left: -3px; top: 0; bottom: 0; width: 6px; cursor: ew-resize; z-index: 2; }
.resize-handle:hover { background: color-mix(in srgb, var(--va-primary) 35%, transparent); }
.artifact-header { min-height: 58px; padding: 10px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--va-background-border); }
.artifact-header > div { display: flex; flex-direction: column; min-width: 0; }
.artifact-header strong { font-size: 14px; }
.artifact-header small { color: var(--va-text-secondary); font-size: 11px; }
.collapse-toggle, .artifact-preview button { border: 0; background: transparent; color: var(--va-text-primary); cursor: pointer; padding: 5px; border-radius: 6px; }
.collapse-toggle:hover, .artifact-preview button:hover { background: var(--va-background-secondary); }
.mode-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 8px; border-bottom: 1px solid var(--va-background-border); }
.mode-tabs button { border: 0; border-radius: 7px; padding: 7px 5px; background: transparent; color: var(--va-text-secondary); cursor: pointer; font-size: 12px; }
.mode-tabs button.active { background: color-mix(in srgb, var(--va-primary) 14%, transparent); color: var(--va-primary); font-weight: 700; }
.mode-tabs span { margin-left: 3px; opacity: .7; }
.artifact-list { padding: 8px; }
.artifact-card { width: 100%; display: grid; grid-template-columns: 30px minmax(0, 1fr) auto; align-items: center; gap: 8px; padding: 9px; margin-bottom: 7px; border: 1px solid var(--va-background-border); border-radius: 9px; background: var(--va-background-primary); text-align: left; color: var(--va-text-primary); }
.artifact-card:hover, .artifact-card.selected { transform: none; border-color: var(--va-primary); background: color-mix(in srgb, var(--va-primary) 6%, var(--va-background-primary)); }
.artifact-card .event-indicator { width: 28px; height: 28px; border-radius: 7px; }
.artifact-copy { min-width: 0; display: flex; flex-direction: column; }
.artifact-copy strong, .artifact-copy span, .artifact-copy small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.artifact-copy strong { font-size: 12px; }
.artifact-copy span { font-size: 11px; color: var(--va-text-primary); }
.artifact-copy small { font-size: 10px; color: var(--va-text-secondary); }
.update-badge { border-radius: 999px; padding: 3px 6px; background: var(--va-background-secondary); color: var(--va-text-secondary); font-size: 9px; white-space: nowrap; }
.artifact-preview { border-top: 1px solid var(--va-background-border); padding: 10px; max-height: 34vh; overflow: auto; background: var(--va-background-primary); }
.artifact-preview header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.artifact-preview .detail-row { font-size: 11px; }
.artifact-preview code { max-width: 65%; overflow: hidden; text-overflow: ellipsis; }
.artifact-panel.collapsed .artifact-header { justify-content: center; padding: 8px 3px; }
.artifact-panel.collapsed .artifact-list { padding: 5px; }
.artifact-panel.collapsed .artifact-card { display: flex; justify-content: center; padding: 6px; }
.artifact-panel.collapsed .resize-handle { display: none; }
</style>
