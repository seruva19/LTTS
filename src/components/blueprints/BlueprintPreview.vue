<template>
  <svg
    :width="width"
    :height="height"
    :viewBox="`0 0 ${width} ${height}`"
    class="blueprint-preview"
  >
    <defs>
      <!-- Gradient for layers -->
      <linearGradient id="layerGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" :style="{ stopColor: 'var(--va-primary)', stopOpacity: 0.8 }" />
        <stop offset="100%" :style="{ stopColor: 'var(--va-primary)', stopOpacity: 0.3 }" />
      </linearGradient>

      <!-- Gradient for active layers (with modules) -->
      <linearGradient id="activeLayerGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" :style="{ stopColor: 'var(--va-success)', stopOpacity: 1 }" />
        <stop offset="100%" :style="{ stopColor: 'var(--va-success)', stopOpacity: 0.6 }" />
      </linearGradient>
    </defs>

    <!-- Background -->
    <rect
      x="0"
      y="0"
      :width="width"
      :height="height"
      fill="transparent"
      rx="4"
    />

    <!-- Layer Stack Visualization -->
    <g v-if="displayMode === 'stack'">
      <!-- Input Layer -->
      <rect
        :x="padding"
        :y="padding"
        :width="layerWidth"
        :height="layerHeight"
        fill="url(#layerGradient)"
        rx="2"
        class="layer-rect"
      />
      <text
        :x="padding + layerWidth / 2"
        :y="padding + layerHeight / 2"
        text-anchor="middle"
        dominant-baseline="middle"
        :style="{ fill: 'var(--va-text-inverted)', fontSize: '8px', fontWeight: '600' }"
      >
        Input
      </text>

      <!-- Middle Layers (simplified) -->
      <g v-for="(layer, index) in visibleLayers" :key="index">
        <rect
          :x="padding"
          :y="padding + layerHeight + layerGap + index * (layerHeight + layerGap)"
          :width="layerWidth"
          :height="layerHeight"
          :fill="hasModules(index) ? 'url(#activeLayerGradient)' : 'url(#layerGradient)'"
          rx="2"
          class="layer-rect"
        />

        <!-- Module indicator dots -->
        <g v-if="hasModules(index)">
          <circle
            v-for="dot in Math.min(getModuleCount(index), 3)"
            :key="dot"
            :cx="padding + layerWidth - 8 - (dot - 1) * 6"
            :cy="padding + layerHeight + layerGap + index * (layerHeight + layerGap) + layerHeight / 2"
            r="2"
            :style="{ fill: 'var(--va-background-element)' }"
          />
        </g>
      </g>

      <!-- Ellipsis indicator if more layers -->
      <g v-if="layerCount > maxVisibleLayers">
        <text
          :x="padding + layerWidth / 2"
          :y="padding + layerHeight + layerGap + maxVisibleLayers * (layerHeight + layerGap) + 8"
          text-anchor="middle"
          :style="{ fill: 'var(--va-text-secondary)', fontSize: '10px', fontWeight: '600' }"
        >
          ... ({{ layerCount - maxVisibleLayers }} more)
        </text>
      </g>

      <!-- Output Layer -->
      <rect
        :x="padding"
        :y="height - padding - layerHeight"
        :width="layerWidth"
        :height="layerHeight"
        fill="url(#layerGradient)"
        rx="2"
        class="layer-rect"
      />
      <text
        :x="padding + layerWidth / 2"
        :y="height - padding - layerHeight / 2"
        text-anchor="middle"
        dominant-baseline="middle"
        :style="{ fill: 'var(--va-text-inverted)', fontSize: '8px', fontWeight: '600' }"
      >
        Output
      </text>
    </g>

    <!-- Distribution Bar Visualization -->
    <g v-else-if="displayMode === 'distribution'">
      <!-- Background grid lines -->
      <line v-for="i in 3" :key="'grid-' + i"
        :x1="padding"
        :x2="width - padding"
        :y1="height - padding - (i * barMaxHeight / 3)"
        :y2="height - padding - (i * barMaxHeight / 3)"
        stroke="var(--va-background-border)"
        stroke-width="0.5"
        opacity="0.3"
      />

      <!-- Distribution bars -->
      <g v-for="(count, index) in distributionBars" :key="index">
        <rect
          :x="padding + index * (barWidth + barGap)"
          :y="height - padding - (count * barMaxHeight / maxModuleCountInLayer)"
          :width="barWidth"
          :height="Math.max(count * barMaxHeight / maxModuleCountInLayer, 1)"
          :fill="count > 0 ? 'var(--va-success)' : 'var(--va-background-border)'"
          :opacity="count > 0 ? 0.85 : 0.25"
          rx="1"
          class="bar-rect"
        >
          <title>{{ count }} module{{ count !== 1 ? 's' : '' }}</title>
        </rect>
      </g>

      <!-- Stats overlay -->
      <g v-if="moduleCount > 0">
        <rect
          :x="padding"
          :y="padding"
          :width="35"
          :height="14"
          fill="var(--va-success)"
          opacity="0.9"
          rx="3"
        />
        <text
          :x="padding + 17.5"
          :y="padding + 9"
          text-anchor="middle"
          dominant-baseline="middle"
          :style="{ fill: 'white', fontSize: '8px', fontWeight: '700' }"
        >
          {{ moduleCount }}
        </text>
      </g>
    </g>

    <!-- Compact Icon Mode -->
    <g v-else>
      <!-- Simple icon representation -->
      <rect
        :x="width / 2 - 20"
        :y="height / 2 - 15"
        width="40"
        height="30"
        fill="url(#layerGradient)"
        rx="4"
      />
      <text
        :x="width / 2"
        :y="height / 2"
        text-anchor="middle"
        dominant-baseline="middle"
        :style="{ fill: 'var(--va-text-inverted)', fontSize: '12px', fontWeight: '700' }"
      >
        {{ layerCount }}
      </text>
    </g>
  </svg>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  layerCount: {
    type: Number,
    default: 0
  },
  moduleCount: {
    type: Number,
    default: 0
  },
  moduleDistribution: {
    type: Object,
    default: () => ({})
  },
  width: {
    type: Number,
    default: 200
  },
  height: {
    type: Number,
    default: 100
  },
  displayMode: {
    type: String,
    default: 'stack', // 'stack', 'distribution', 'icon'
    validator: (value) => ['stack', 'distribution', 'icon'].includes(value)
  }
})

const padding = 10
const layerHeight = 8
const layerGap = 4
const maxVisibleLayers = 8

const layerWidth = computed(() => props.width - 2 * padding)

const visibleLayers = computed(() => {
  const count = Math.min(props.layerCount, maxVisibleLayers)
  return Array.from({ length: count }, (_, i) => i)
})

function hasModules(layerIndex) {
  return getModuleCount(layerIndex) > 0
}

function getModuleCount(layerIndex) {
  const key = `layer_${layerIndex}`
  return props.moduleDistribution[key] || 0
}

// Distribution bar chart calculations
const barGap = 2
const barMaxHeight = computed(() => props.height - 2 * padding - 12) // Reserve space for label

const distributionBars = computed(() => {
  if (!props.layerCount) return []

  // Sample layers for distribution (max 20 bars to fit)
  const maxBars = 20
  const step = Math.max(1, Math.floor(props.layerCount / maxBars))

  const bars = []
  for (let i = 0; i < Math.min(props.layerCount, maxBars); i++) {
    const layerIndex = i * step
    bars.push(getModuleCount(layerIndex))
  }

  return bars
})

const barWidth = computed(() => {
  const totalBars = distributionBars.value.length
  if (totalBars === 0) return 0
  return (props.width - 2 * padding - (totalBars - 1) * barGap) / totalBars
})

const maxModuleCountInLayer = computed(() => {
  const counts = Object.values(props.moduleDistribution)
  return counts.length > 0 ? Math.max(...counts, 1) : 1
})
</script>

<style scoped>
.blueprint-preview {
  display: block;
  max-width: 100%;
  height: auto;
  filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
}

.layer-rect {
  transition: all 0.2s ease;
}

.layer-rect:hover {
  opacity: 0.9;
}

.bar-rect {
  transition: all 0.2s ease;
}

.bar-rect:hover {
  opacity: 1 !important;
  filter: brightness(1.1);
}
</style>
