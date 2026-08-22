<template>
  <div class="ltts-module" :class="{ muted: isMuted }">
    <div class="module-header">
      <div class="left">
        <VaBadge :text="positionLabel" :color="positionColor" class="pos-badge" />
        <h5 class="module-title">{{ moduleAttachment.name }}</h5>
        <VaChip v-if="moduleInfo?.metadata?.version" size="small" color="info">v{{ moduleInfo.metadata.version }}</VaChip>
      </div>
      <div class="right">
        <VaButton
          size="small"
          preset="plain"
          :icon="isEnabled ? 'biotech' : 'block'"
          :color="isEnabled ? 'primary' : 'gray'"
          class="header-toggle"
          :aria-label="isEnabled ? 'Enabled' : 'Disabled'"
          title="Toggle module"
          @click="setEnabled(!isEnabled)"
        />
        <div class="sampling-container">
          <div class="sampling-panel" ref="samplingPanelRef">
            <div class="mode-group">
              <button
                class="mode-btn"
                :class="{ active: selectedEmitMode === 'all' }"
                :disabled="!canEditEmission"
                @click="setMode('all')"
              >All</button>
              <button
                class="mode-btn"
                :class="{ active: selectedEmitMode === 'first' }"
                :disabled="!canEditEmission"
                @click="setMode('first')"
              >First</button>
              <button
                class="mode-btn"
                :class="{ active: selectedEmitMode === 'every_n' }"
                :disabled="!canEditEmission"
                @click="setMode('every_n')"
              >N=</button>
              <button
                class="mode-btn"
                :class="{ active: selectedEmitMode === 'final' }"
                :disabled="!canEditEmission"
                @click="setMode('final')"
              >Final</button>
              <button
                class="mode-btn"
                :class="{ active: selectedEmitMode === 'manual' }"
                :disabled="!canEditEmission"
                @click="setMode('manual')"
              >Manual</button>
            </div>
            <div class="stride-inline" :class="{ inactive: selectedEmitMode !== 'every_n' }">
              <span class="stride-prefix">N=</span>
              <input
                v-model.number="selectedEmitStride"
                type="number"
                min="1"
                class="sampling-number"
                :disabled="!canEditEmission || selectedEmitMode !== 'every_n'"
                placeholder="5"
              />
            </div>
          </div>
        </div>
        <VaButton preset="plain" size="small" icon="close" @click="$emit('remove')" aria-label="Remove module" />
      </div>
    </div>

    <div v-if="hasSchema" class="module-body">
      <DynamicForm
        :schema="filteredSchema"
        :initial-data="moduleAttachment.parameters || {}"
        :show-actions="false"
        @fieldChange="onFieldChange"
        @update="onFormUpdate"
      />
    </div>

    <div v-else class="no-schema">
      <VaAlert color="secondary" border="left" size="small">
        This module has no configurable UI.
      </VaAlert>
    </div>

    <div class="visualization-panel">
      <div
        class="viz-header"
        role="button"
        tabindex="0"
        :aria-expanded="showVisualizations"
        @click="showVisualizations = !showVisualizations"
        @keydown.enter.prevent="showVisualizations = !showVisualizations"
        @keydown.space.prevent="showVisualizations = !showVisualizations"
      >
        <span class="viz-title">Visualizations</span>
        <span class="viz-count">({{ moduleVisualizations.length }})</span>
        <VaButton
          preset="plain"
          size="small"
          :icon="showVisualizations ? 'expand_less' : 'expand_more'"
          :disabled="true"
          title="No direct action"
        />
      </div>
      <div v-if="showVisualizations" class="viz-content">
        <div v-if="moduleVisualizations.length === 0" class="no-viz-message subtle">
          No outputs yet.
        </div>

        <div v-for="viz in moduleVisualizations" :key="viz.id" class="viz-item">
          <div v-if="viz.data && viz.data.visualization" class="control-vector-viz">
            <div v-if="viz.data.visualization.type === 'image'" class="image-viz">
              <img :src="viz.data.visualization.data" :alt="viz.data.visualization.description" class="viz-image" />
              <p class="viz-desc">{{ viz.data.visualization.description }}</p>
            </div>
            <div v-else-if="viz.data.visualization.type === 'scalar'" class="scalar-viz">
              <div class="scalar-value">{{ viz.data.visualization.value.toFixed(4) }}</div>
              <div class="scalar-unit">{{ viz.data.visualization.unit }}</div>
              <p class="viz-desc">{{ viz.data.visualization.description }}</p>
            </div>
          </div>

          <!-- Layer Inspector Visualization - display as JSON -->
          <div v-else-if="viz.type === 'layer_inspector'" class="layer-inspector-viz">
            <VaCollapse header="Layer Inspector Data" size="small">
              <div class="layer-inspector-content">
                <pre class="json-display">{{ JSON.stringify(viz.data, null, 2) }}</pre>
              </div>
            </VaCollapse>
          </div>

          <!-- Standard Visualization -->
          <div v-else-if="viz.data && typeof viz.data === 'string' && viz.data.startsWith('data:image')" class="image-viz">
            <img :src="viz.data" :alt="`${viz.module} ${viz.type}`" class="viz-image" />
          </div>

          <!-- Typed Scalar Visualization -->
          <div v-else-if="viz.data && viz.data.type === 'scalar'" class="scalar-viz">
            <div class="scalar-value">{{ formatNumber(viz.data.value) }}</div>
            <div class="scalar-unit">{{ viz.data.label || viz.data.unit || '' }}</div>
          </div>

          <!-- Typed Table Visualization -->
          <div v-else-if="viz.data && viz.data.type === 'table'" class="table-viz">
            <table class="viz-table">
              <thead>
                <tr>
                  <th v-for="header in (viz.data.headers || [])" :key="header">{{ header }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, rowIdx) in (viz.data.rows || [])" :key="rowIdx">
                  <td v-for="(cell, cellIdx) in row" :key="cellIdx">{{ cell }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Typed Chart Visualization (histogram/bar/line) -->
          <div v-else-if="viz.data && viz.data.type === 'chart'" class="chart-viz">
            <div class="chart-header">{{ viz.data.chart_type || 'chart' }}</div>
            <div v-if="viz.data.data && viz.data.data.histogram" class="histogram-bars">
              <div
                v-for="(val, idx) in viz.data.data.histogram"
                :key="idx"
                class="histogram-bar"
                :style="{ height: histogramBarHeight(val, viz.data.data.histogram) + '%' }"
                :title="`Bin ${idx}: ${val.toFixed(4)}`"
              />
            </div>
            <div v-else-if="viz.data.data && viz.data.data.importance" class="importance-bars">
              <div
                v-for="(val, idx) in viz.data.data.importance"
                :key="idx"
                class="importance-bar"
                :style="{ height: histogramBarHeight(val, viz.data.data.importance) + '%' }"
                :title="`Position ${idx}: ${val.toFixed(4)}`"
              />
            </div>
            <div v-else-if="seriesChart(viz.data)" class="series-chart">
              <svg :viewBox="`0 0 ${SERIES_W} ${SERIES_H}`" preserveAspectRatio="none" class="series-svg">
                <template v-for="(s, si) in seriesChart(viz.data).series" :key="s.name">
                  <polyline
                    v-if="seriesChart(viz.data).kind === 'line'"
                    :points="s.polyline"
                    :stroke="seriesColor(si)"
                    fill="none"
                    stroke-width="1.5"
                    vector-effect="non-scaling-stroke"
                  />
                  <rect
                    v-else
                    v-for="(bar, bi) in s.bars"
                    :key="bi"
                    :x="bar.x"
                    :y="bar.y"
                    :width="bar.w"
                    :height="bar.h"
                    :fill="seriesColor(si)"
                  >
                    <title>{{ bar.label }}</title>
                  </rect>
                </template>
              </svg>
              <div class="series-legend">
                <span v-for="(s, si) in seriesChart(viz.data).series" :key="s.name" class="series-key">
                  <i class="series-swatch" :style="{ backgroundColor: seriesColor(si) }" />
                  {{ s.name }}
                </span>
                <span class="series-range">
                  {{ seriesChart(viz.data).xLabel }}:
                  {{ seriesChart(viz.data).xMin }}..{{ seriesChart(viz.data).xMax }} |
                  {{ seriesChart(viz.data).yLabel }}:
                  {{ seriesChart(viz.data).yMin }}..{{ seriesChart(viz.data).yMax }}
                </span>
              </div>
            </div>
            <div v-else class="chart-fallback">
              <pre class="json-content">{{ JSON.stringify(viz.data.data, null, 2) }}</pre>
            </div>
          </div>

          <!-- Typed Heatmap Visualization -->
          <div v-else-if="viz.data && viz.data.type === 'heatmap'" class="heatmap-viz">
            <div class="heatmap-grid" :style="heatmapGridStyle(viz.data)">
              <div
                v-for="(row, rowIdx) in (viz.data.values || []).slice(0, 32)"
                :key="rowIdx"
                class="heatmap-row"
              >
                <div
                  v-for="(cell, colIdx) in row.slice(0, 32)"
                  :key="colIdx"
                  class="heatmap-cell"
                  :style="{ backgroundColor: heatmapColor(cell, viz.data.values) }"
                  :title="`[${rowIdx},${colIdx}]: ${typeof cell === 'number' ? cell.toFixed(4) : cell}`"
                />
              </div>
            </div>
            <div v-if="viz.data.dimensions" class="heatmap-dims">
              {{ viz.data.dimensions.rows }} x {{ viz.data.dimensions.cols }}
            </div>
          </div>

          <!-- Typed Image Visualization (ModuleDataSender.send_image) -->
          <div v-else-if="viz.data && viz.data.type === 'image'" class="image-viz">
            <img :src="imageSource(viz.data)" :alt="`${viz.module} image`" class="viz-image" />
          </div>

          <!-- Typed Text Visualization -->
          <div v-else-if="viz.data && viz.data.type === 'text'" class="text-viz">
            <pre v-if="viz.data.preformatted" class="preformatted-text">{{ viz.data.content }}</pre>
            <p v-else>{{ viz.data.content }}</p>
          </div>

          <!-- JSON Data (generic fallback for objects) -->
          <div v-else-if="viz.data && typeof viz.data === 'object'" class="json-viz">
            <VaCollapse header="View Data" size="small">
              <pre class="json-content">{{ JSON.stringify(viz.data, null, 2) }}</pre>
            </VaCollapse>
          </div>

          <!-- Text Data (generic fallback for strings) -->
          <div v-else-if="viz.data" class="text-viz">
            <p>{{ viz.data }}</p>
          </div>

          <div class="viz-meta">
            <span class="viz-time">{{ formatTime(viz.timestamp) }}</span>
            <VaChip size="small" :color="getVizTypeColor(viz.type)">{{ viz.type }}</VaChip>
            <span v-if="viz.forwardPass !== null" class="viz-pass">Pass {{ viz.forwardPass }}</span>
          </div>
        </div>

      </div>
    </div>

    <!-- Cross-Attention Anchors (for attention modules) -->
    <div v-if="isAttentionModule && hasAttentionAnchors" class="attention-anchors">
      <div class="anchors-header">
        <VaIcon name="anchor" size="small" />
        <span>Cross-Attention Anchors</span>
      </div>
      <div class="anchors-content">
        <div v-for="(anchor, index) in attentionAnchors" :key="index" class="anchor-item">
          <VaBadge :text="`H${anchor.head}`" color="primary" size="small" />
          <span class="anchor-token">{{ anchor.token }}</span>
          <span class="anchor-weight">{{ (anchor.weight * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import { useLttsState } from '@/state'
import DynamicForm from '@/components/ui-renderer/dynamic-form.vue'
const props = defineProps({
  moduleAttachment: { type: Object, required: true },
  layer: { type: Object, required: true },
  position: { type: String, default: 'post' } // 'pre' | 'post'
})

const emit = defineEmits(['remove'])

const state = useLttsState()

// Visualization panel state
const showVisualizations = ref(true)
const hasNewVisualizations = ref(false)
const lastVisualizationCount = ref(0)
const samplingPanelRef = ref(null)

const moduleInfo = computed(() => {
  const name = props.moduleAttachment?.name
  if (!name) return null
  return state.modules.modules?.[name] || null
})

// Mute state for this attached module
const isMuted = computed(() => {
  const name = props.moduleAttachment?.name
  if (!name) return false
  return state.modules.isMuted(name)
})

// Enabled is inverse of muted
const isEnabled = computed(() => {
  const name = props.moduleAttachment?.name
  if (!name) return true
  return !state.modules.isMuted(name)
})

const setEnabled = (val) => {
  const name = props.moduleAttachment?.name
  if (!name) return
  const currentlyEnabled = !state.modules.isMuted(name)
  if (val !== currentlyEnabled) {
    state.modules.toggleMute(name)
  }
}

const emissionModeOptions = [
  { text: 'All passes', value: 'all' },
  { text: 'First pass', value: 'first' },
  { text: 'Every N', value: 'every_n' },
  { text: 'Final pass', value: 'final' },
  { text: 'Manual', value: 'manual' }
]

const canEditEmission = computed(() => !!props.moduleAttachment)

const updateAttachmentParameters = (updates) => {
  const current = { ...(props.moduleAttachment?.parameters || {}) }
  const merged = { ...current, ...updates }
  state.blueprints.updateModuleParameters(props.layer, props.moduleAttachment.id, merged)
}

const selectedEmitStride = computed({
  get: () => {
    const stride = props.moduleAttachment?.parameters?.emit_every_n
    return Number.isFinite(stride) && stride > 0 ? stride : 5
  },
  set: (value) => {
    const numericValue = Number(value)
    const sanitized = Number.isFinite(numericValue) && numericValue > 0
      ? Math.floor(numericValue)
      : 1
    updateAttachmentParameters({ emit_every_n: sanitized })
  }
})

const selectedEmitMode = computed({
  get: () => props.moduleAttachment?.parameters?.emit_mode || 'final',
  set: (value) => {
    const stride = selectedEmitStride.value || 5
    updateAttachmentParameters({
      emit_mode: value,
      emit_every_n: value === 'every_n' ? Math.max(1, stride) : stride
    })
  }
})

const samplingLabel = computed(() => {
  switch (selectedEmitMode.value) {
    case 'all':
      return 'All'
    case 'first':
      return 'First'
    case 'every_n':
      return 'N='
    case 'final':
      return 'Final'
    case 'manual':
      return 'Manual'
    default:
      return 'All'
  }
})

const samplingDetail = computed(() => {
  if (selectedEmitMode.value === 'every_n') {
    return `N=${selectedEmitStride.value}`
  }
  return null
})

const setMode = (mode) => {
  if (!canEditEmission.value) return
  selectedEmitMode.value = mode
}

const filteredSchema = computed(() => {
  const schema = moduleInfo.value?.ui_schema
  if (!schema) return null

  const cloned = JSON.parse(JSON.stringify(schema))
  if (Array.isArray(cloned.parameters)) {
    cloned.parameters = cloned.parameters.filter(param => !['emit_mode', 'emit_every_n'].includes(param.name))
  }
  return cloned
})

const hasSchema = computed(() => {
  return !!filteredSchema.value?.parameters?.length
})

// Get module visualizations - use layer.name for filtering
const layerIdentifier = computed(() => {
  if (props.moduleAttachment?.layerPath) {
    return props.moduleAttachment.layerPath
  }

  if (props.layer && typeof props.layer.fullPath === 'string') {
    return props.layer.fullPath
  }

  if (props.layer && typeof props.layer.originalPath === 'string') {
    return props.layer.originalPath
  }

  if (props.layer && typeof props.layer.path === 'string') {
    return props.layer.path.startsWith('original_model.')
      ? props.layer.path
      : `original_model.model.${props.layer.path.replace(/^model\./, '')}`
  }

  if (props.layer && typeof props.layer.name === 'string') {
    return props.layer.name.startsWith('original_model.')
      ? props.layer.name
      : `original_model.model.${props.layer.name.replace(/^model\./, '')}`
  }

  if (typeof props.layer === 'string') {
    return props.layer.startsWith('original_model.')
      ? props.layer
      : `original_model.model.${props.layer.replace(/^model\./, '')}`
  }

  return null
})

const moduleVisualizations = computed(() => {
  const layerKey = layerIdentifier.value
  const layerVizs = state.ws.getRecentVisualizationsByModule(
    props.moduleAttachment.name,
    layerKey
  )


  return layerVizs
})

const hasVisualizations = computed(() => {
  const hasViz = moduleVisualizations.value.length > 0
    return hasViz
})

// Check if this is an attention module
const isAttentionModule = computed(() => {
  const name = props.moduleAttachment.name.toLowerCase()
  return name.includes('attention') || name.includes('attn')
})

// Attention anchors derived from actual visualization data (no mock data)
const attentionAnchors = computed(() => {
  if (!isAttentionModule.value) return []

  // Extract anchors from actual attention visualization data if available
  for (const viz of moduleVisualizations.value) {
    const anchors = viz.data?.attention_anchors || viz.data?.data?.attention_anchors
    if (Array.isArray(anchors) && anchors.length > 0) {
      return anchors
    }
  }
  return []
})

const hasAttentionAnchors = computed(() => attentionAnchors.value.length > 0)

const positionLabel = computed(() => (props.position === 'pre' ? 'PRE' : 'POST'))
const positionColor = computed(() => (props.position === 'pre' ? 'info' : 'warning'))

// Check for new visualizations
const checkForNewVisualizations = () => {
  const currentCount = moduleVisualizations.value.length
  if (currentCount > lastVisualizationCount.value) {
    hasNewVisualizations.value = true
    setTimeout(() => {
      hasNewVisualizations.value = false
    }, 3000) // Clear "NEW" badge after 3 seconds
  }
  lastVisualizationCount.value = currentCount
}

// Utility functions
const formatTime = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString()
}

const formatNumber = (num) => {
  if (num === undefined || num === null) return 'N/A'
  if (typeof num !== 'number') return num.toString()
  return num.toFixed(6)
}

const formatMemory = (bytes) => {
  if (!bytes || bytes === 0) return 'N/A'
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`
  return `${bytes} B`
}

const getVizTypeColor = (type) => {
  const typeMap = {
    'attention': 'primary',
    'control_vector': 'success',
    'gradient': 'warning',
    'activation': 'info',
    'default': 'secondary'
  }
  return typeMap[type?.toLowerCase()] || typeMap.default
}

// Histogram bar height as percentage of max value
const SERIES_W = 320
const SERIES_H = 120
const SERIES_COLORS = ['#3b82f6', '#f97316', '#10b981', '#a855f7', '#ef4444', '#64748b']

const seriesColor = (index) => SERIES_COLORS[index % SERIES_COLORS.length]

const imageSource = (data) => {
  // send_image(payload, image_type) stores the payload under the image_type key.
  const raw = data.base64 ?? data.url ?? data.src ?? data.data
  if (typeof raw !== 'string') return ''
  return raw.startsWith('data:') || raw.startsWith('http') ? raw : `data:image/png;base64,${raw}`
}

// Project {"series": [{"name", "points": [{x, y}]}]} onto the SVG viewBox.
const seriesChart = (data) => {
  const series = data?.data?.series
  if (!Array.isArray(series) || series.length === 0) return null

  const points = series.map((s) => (Array.isArray(s.points) ? s.points : [])
    .map((p, i) => ({ x: Number(p.x ?? i), y: Number(p.y) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)))
  const flat = points.flat()
  if (flat.length === 0) return null

  const xs = flat.map((p) => p.x)
  const ys = flat.map((p) => p.y)
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const xSpan = xMax - xMin || 1
  const ySpan = yMax - yMin || 1

  const px = (x) => ((x - xMin) / xSpan) * SERIES_W
  // SVG y grows downwards, so a larger value must sit closer to the top.
  const py = (y) => SERIES_H - ((y - yMin) / ySpan) * SERIES_H

  const kind = (data.chart_type === 'line' || data.chart_type === undefined) ? 'line' : 'bar'
  const barWidth = Math.max(1, SERIES_W / Math.max(1, flat.length))

  return {
    kind,
    xLabel: data?.data?.x_label || 'x',
    yLabel: data?.data?.y_label || 'y',
    xMin: round2(xMin),
    xMax: round2(xMax),
    yMin: round2(yMin),
    yMax: round2(yMax),
    series: series.map((s, si) => ({
      name: s.name || `series ${si + 1}`,
      polyline: points[si].map((p) => `${px(p.x)},${py(p.y)}`).join(' '),
      bars: points[si].map((p) => {
        const top = py(p.y)
        const base = py(Math.max(yMin, 0))
        return {
          x: px(p.x) - barWidth / 2,
          y: Math.min(top, base),
          w: barWidth * 0.8,
          h: Math.max(1, Math.abs(base - top)),
          label: `${p.x}: ${round2(p.y)}`
        }
      })
    }))
  }
}

const round2 = (value) => Math.round(value * 1000) / 1000

const histogramBarHeight = (value, allValues) => {
  if (!allValues || allValues.length === 0) return 0
  const maxVal = Math.max(...allValues)
  if (maxVal === 0) return 0
  return Math.max(1, (value / maxVal) * 100)
}

// Heatmap grid style based on dimensions
const heatmapGridStyle = (data) => {
  const cols = Math.min(data?.dimensions?.cols || data?.values?.[0]?.length || 8, 32)
  return {
    display: 'grid',
    gridTemplateColumns: `repeat(${cols}, 1fr)`,
    gap: '1px',
  }
}

// Heatmap cell color (blue-white-red diverging scale)
const heatmapColor = (value, allValues) => {
  if (typeof value !== 'number' || !allValues) return 'var(--va-background-element)'
  const flat = allValues.flat()
  const minVal = Math.min(...flat)
  const maxVal = Math.max(...flat)
  const range = maxVal - minVal
  if (range === 0) return 'rgb(255, 255, 255)'
  const normalized = (value - minVal) / range // 0..1
  // Blue (low) -> White (mid) -> Red (high)
  if (normalized <= 0.5) {
    const t = normalized * 2 // 0..1
    const r = Math.round(t * 255)
    const g = Math.round(t * 255)
    return `rgb(${r}, ${g}, 255)`
  } else {
    const t = (normalized - 0.5) * 2 // 0..1
    const g = Math.round((1 - t) * 255)
    const b = Math.round((1 - t) * 255)
    return `rgb(255, ${g}, ${b})`
  }
}


// Refresh this module when the visualization store changes.
const handleWebSocketMessage = (message) => {
  checkForNewVisualizations()
}

const onFieldChange = ({ field, value, formData }) => {
  try {
    state.blueprints.updateModuleParameters(props.layer, props.moduleAttachment.id, formData)
  } catch (error) {
    console.error('Error in onFieldChange:', error, { field, value, layer: props.layer, moduleId: props.moduleAttachment.id })
  }
}

const onFormUpdate = (formData) => {
  try {
    state.blueprints.updateModuleParameters(props.layer, props.moduleAttachment.id, formData)
  } catch (error) {
    console.error('Error in onFormUpdate:', error, { formData, layer: props.layer, moduleId: props.moduleAttachment.id })
  }
}

onMounted(() => {
  // Initial check
  lastVisualizationCount.value = moduleVisualizations.value.length

  // Watch for changes in visualizations store
  const stopWatching = watch(
    () => state.ws.visualizations,
    () => {
      checkForNewVisualizations()
    },
    { deep: true }
  )

  // Cleanup watcher on unmount
  onUnmounted(() => {
    stopWatching()
  })
})
</script>

<style scoped>
.ltts-module {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.75rem;
  background: var(--va-background-element);
  border-radius: 8px;
  border: 1px solid var(--va-background-border);
}

.module-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
}

.module-header .left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.module-header .right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.ltts-module.muted {
  opacity: 0.6;
}

.sampling-trigger {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.1rem 0.3rem;
  border-radius: 4px;
  border: 1px solid var(--va-primary);
  background: rgba(var(--va-primary-rgb), 0.08);
  color: var(--va-primary);
  font-size: 0.7rem;
  cursor: pointer;
  transition: background 0.2s ease, border-color 0.2s ease;
  height: 22px;
}

.sampling-trigger:disabled {
  opacity: 0.5;
  cursor: default;
}

.sampling-trigger:hover:not(:disabled) {
  background: rgba(var(--va-primary-rgb), 0.14);
}

.sampling-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 500;
}

.sampling-detail {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
}

.sampling-container {
  display: inline-flex;
  align-items: center;
}

.sampling-panel {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  background: transparent;
  border: none;
  padding: 0;
}

.mode-group {
  display: inline-flex;
  gap: 0.2rem;
}

.mode-btn {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  line-height: 1;
  padding: 0.1rem 0.25rem;
  border: 1px solid rgba(52, 152, 219, 0.3);
  border-radius: 3px;
  background: rgba(52, 152, 219, 0.08);
  color: var(--va-primary);
  cursor: pointer;
}

.mode-btn.active {
  background: rgba(52, 152, 219, 0.2);
  border-color: var(--va-primary);
}

.mode-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.stride-inline {
  display: inline-flex;
  align-items: center;
  gap: 0.15rem;
  background: transparent;
  border: none;
}

.stride-inline.inactive {
  opacity: 0.5;
}

.stride-prefix {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--va-primary);
  font-weight: 600;
}

.sampling-number {
  width: 36px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  padding: 0.1rem 0.2rem;
  border: 1px solid rgba(52, 152, 219, 0.3);
  border-radius: 2px;
  background: transparent;
  color: inherit;
}

.pos-badge {
  font-size: 0.625rem;
}

.module-title {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
}

.module-body {
  padding-top: 0.25rem;
}

.no-schema {
  padding-top: 0.25rem;
}

/* Visualization Panel Styles */
.visualization-panel {
  border: 1px solid var(--va-primary);
  border-radius: 6px;
  background: var(--va-background-element);
  margin-top: 0.5rem;
}

.viz-header:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.viz-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: linear-gradient(135deg, var(--va-primary) 0%, var(--va-primary-dark) 100%);
  color: white;
  border-bottom: 1px solid var(--va-background-border);
  cursor: pointer;
  user-select: none;
  gap: 0.5rem;
}

.viz-header:hover {
  background: linear-gradient(135deg, var(--va-primary-dark) 0%, var(--va-primary) 100%);
}

.viz-header span {
  font-size: 0.875rem;
  font-weight: 500;
  flex: 1;
}

.viz-content {
  padding: 0.75rem;
  overflow-y: auto;
}

.viz-item {
  border: 1px solid var(--va-background-border);
  border-radius: 4px;
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  background: var(--va-background-secondary);
}

.viz-item:last-child {
  margin-bottom: 0;
}

.viz-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--va-background-border);
}

.viz-time {
  font-size: 0.75rem;
  color: var(--va-text-secondary);
}

/* Visualization Type Styles */
.image-viz {
  text-align: center;
}

.viz-image {
  max-width: 100%;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.viz-desc {
  font-size: 0.875rem;
  color: var(--va-text-secondary);
  font-style: italic;
  margin: 0.5rem 0 0 0;
}

.scalar-viz {
  text-align: center;
  padding: 1rem;
}

.scalar-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--va-primary);
  margin-bottom: 0.25rem;
}

.scalar-unit {
  font-size: 0.75rem;
  color: var(--va-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.json-content {
  font-family: 'Courier New', monospace;
  font-size: 0.75rem;
  background: var(--va-background-element);
  padding: 0.5rem;
  border-radius: 4px;
  overflow-x: auto;
  max-height: 150px;
  overflow-y: auto;
  margin: 0;
}

.text-viz p {
  margin: 0;
  padding: 0.5rem;
  background: var(--va-background-element);
  border-radius: 4px;
  font-size: 0.875rem;
}

.more-viz {
  text-align: center;
  padding-top: 0.5rem;
  border-top: 1px solid var(--va-background-border);
}

/* Cross-Attention Anchors Styles */
.attention-anchors {
  border: 1px solid var(--va-warning);
  border-radius: 6px;
  background: var(--va-background-element);
  margin-top: 0.5rem;
}

.anchors-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: linear-gradient(135deg, var(--va-warning) 0%, var(--va-warning-dark) 100%);
  color: white;
  border-bottom: 1px solid var(--va-background-border);
}

.anchors-header span {
  font-size: 0.875rem;
  font-weight: 500;
}

.anchors-content {
  padding: 0.75rem;
}

.anchor-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0;
  border-bottom: 1px solid var(--va-background-border);
}

.anchor-item:last-child {
  border-bottom: none;
}

.anchor-token {
  font-family: 'Courier New', monospace;
  font-weight: 500;
  background: var(--va-background-secondary);
  padding: 0.125rem 0.25rem;
  border-radius: 3px;
  flex: 1;
}

.anchor-weight {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--va-primary);
}

.no-viz-message {
  text-align: center;
  padding: 1rem;
}

/* Visualization panel */
.visualization-panel {
  border: 1px solid var(--va-background-border);
  border-radius: 6px;
  background: var(--va-background-element);
  margin-top: 0.5rem;
}
.viz-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  color: var(--va-text-primary);
}
.viz-title {
  font-weight: 600;
  font-size: 0.875rem;
}
.viz-count {
  font-size: 0.75rem;
  color: var(--va-text-secondary);
}
.viz-content {
  padding: 0.75rem;
}
.no-viz-message.subtle {
  padding: 0.5rem;
  font-size: 0.8rem;
  color: var(--va-text-secondary);
}

/* Layer Inspector Visualization Styles */
.layer-inspector-viz {
  border: 1px solid var(--va-info);
  border-radius: 6px;
  background: var(--va-background-element);
  margin-top: 0.5rem;
}

.layer-inspector-content {
  padding: 0.75rem;
}

.inspector-section {
  margin-bottom: 1rem;
}

.inspector-section:last-child {
  margin-bottom: 0;
}

.inspector-section h6 {
  margin: 0 0 0.5rem 0;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--va-info);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.shape-item, .dtype-item {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  padding: 0.25rem 0;
  color: var(--va-text-primary);
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0;
  border-bottom: 1px solid var(--va-background-border);
}

.stat-item:last-child {
  border-bottom: none;
}

.stat-label {
  font-size: 0.8rem;
  color: var(--va-text-secondary);
}

.stat-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--va-text-primary);
}

.memory-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0;
}

.memory-label {
  font-size: 0.8rem;
  color: var(--va-text-secondary);
}

.memory-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--va-success);
}

/* Table Visualization */
.table-viz {
  overflow-x: auto;
}

.viz-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
  font-family: 'JetBrains Mono', monospace;
}

.viz-table th {
  background: var(--va-background-element);
  padding: 0.35rem 0.5rem;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid var(--va-background-border);
  color: var(--va-primary);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.viz-table td {
  padding: 0.25rem 0.5rem;
  border-bottom: 1px solid var(--va-background-border);
  color: var(--va-text-primary);
}

.viz-table tbody tr:hover {
  background: rgba(var(--va-primary-rgb), 0.04);
}

/* Chart Visualization */
.chart-viz {
  padding: 0.25rem;
}

.chart-header {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--va-text-secondary);
  margin-bottom: 0.35rem;
}

.histogram-bars, .importance-bars {
  display: flex;
  align-items: flex-end;
  gap: 1px;
  height: 80px;
  background: var(--va-background-element);
  border-radius: 4px;
  padding: 2px;
}

.histogram-bar {
  flex: 1;
  min-width: 2px;
  background: var(--va-primary);
  border-radius: 1px 1px 0 0;
  transition: opacity 0.15s;
  opacity: 0.75;
}

.histogram-bar:hover {
  opacity: 1;
}

.importance-bar {
  flex: 1;
  min-width: 2px;
  background: var(--va-warning);
  border-radius: 1px 1px 0 0;
  transition: opacity 0.15s;
  opacity: 0.75;
}

.importance-bar:hover {
  opacity: 1;
}

.series-chart {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.series-svg {
  width: 100%;
  height: 120px;
  background: var(--va-background-element, rgba(0, 0, 0, 0.03));
  border-radius: 4px;
}

.series-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  font-size: 11px;
  opacity: 0.8;
}

.series-key {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.series-swatch {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  display: inline-block;
}

.series-range {
  margin-left: auto;
  opacity: 0.7;
}

.chart-fallback {
  max-height: 120px;
  overflow-y: auto;
}

/* Heatmap Visualization */
.heatmap-viz {
  padding: 0.25rem;
}

.heatmap-grid {
  border-radius: 4px;
  overflow: hidden;
}

.heatmap-row {
  display: contents;
}

.heatmap-cell {
  aspect-ratio: 1;
  min-width: 4px;
  min-height: 4px;
  transition: opacity 0.15s;
}

.heatmap-cell:hover {
  opacity: 0.8;
  outline: 1px solid var(--va-text-primary);
  z-index: 1;
}

.heatmap-dims {
  font-size: 0.7rem;
  color: var(--va-text-secondary);
  text-align: center;
  margin-top: 0.25rem;
}

/* Preformatted text */
.preformatted-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  padding: 0.5rem;
  background: var(--va-background-element);
  border-radius: 4px;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}

/* Layer Inspector JSON display */
.json-display {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  background: var(--va-background-element);
  padding: 0.75rem;
  border-radius: 4px;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  border: 1px solid var(--va-background-border);
}
</style>
