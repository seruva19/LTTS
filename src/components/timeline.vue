<script setup>
import { useLttsState } from '../state';
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
const state = useLttsState()
const isControlsCollapsed = ref(false)

const currentLayerIndex = ref(-1)
const isManualScrolling = ref(false)

// Track expanded groups
const expandedGroups = ref(new Set())


// Use WebSocket store state for processing layer
const currentProcessingLayer = computed(() => state.ws.currentProcessingLayer)
const isInferenceRunning = computed(() => state.ws.isInferenceRunning)

// Timeline collapse logic
const MAX_VISIBLE_BLOCKS = 20
const totalLayers = computed(() => state.blueprints.loadedBlueprint?.layers?.length || 0)
const shouldCollapse = computed(() => totalLayers.value > MAX_VISIBLE_BLOCKS)

// Calculate collapsed blocks with expansion support
const collapsedBlocks = computed(() => {
  if (!shouldCollapse.value) {
    // Show all layers normally - maintain original order from blueprint
    return (state.blueprints.loadedBlueprint?.layers || []).map((layer, index) => ({
      type: 'layer',
      layer,
      index,
      startIndex: index,
      endIndex: index,
      layerCount: 1
    }))
  }

  // Get layers in the correct order - either from categories or flat list
  let orderedLayers = []
  if (state.blueprints.loadedBlueprint?.categorization?.categories) {
    // Extract layers from categories in order
    Object.values(state.blueprints.loadedBlueprint.categorization.categories).forEach(category => {
      if (category.layers && Array.isArray(category.layers)) {
        orderedLayers.push(...category.layers)
      }
    })
  } else {
    // Fallback to flat layers list
    orderedLayers = state.blueprints.loadedBlueprint?.layers || []
  }

  const blocksPerGroup = Math.ceil(orderedLayers.length / MAX_VISIBLE_BLOCKS)
  const blocks = []

  for (let i = 0; i < MAX_VISIBLE_BLOCKS; i++) {
    const startIndex = i * blocksPerGroup
    const endIndex = Math.min(startIndex + blocksPerGroup - 1, orderedLayers.length - 1)
    const groupKey = `${startIndex}-${endIndex}`

    if (startIndex < orderedLayers.length) {
      const layersInGroup = orderedLayers.slice(startIndex, endIndex + 1)
      const totalModules = layersInGroup.reduce((sum, layer) => sum + getLayerModuleCount(layer), 0)

      // Always add the group block first (for re-collapsing)
      blocks.push({
        type: 'group',
        startIndex,
        endIndex,
        layerCount: endIndex - startIndex + 1,
        layers: layersInGroup,
        totalModules,
        // Use the first layer as representative
        representativeLayer: layersInGroup[0],
        groupKey,
        isExpanded: expandedGroups.value.has(groupKey)
      })

      // Check if this group is expanded and add individual layers
      if (expandedGroups.value.has(groupKey)) {
        layersInGroup.forEach((layer, layerIndex) => {
          const actualIndex = startIndex + layerIndex
          blocks.push({
            type: 'layer',
            layer,
            index: actualIndex,
            startIndex: actualIndex,
            endIndex: actualIndex,
            layerCount: 1,
            parentGroup: groupKey,
            isExpandedChild: true
          })
        })
      }
    }
  }

  return blocks
})

const scrollToLayer = (layerName, index) => {
  const layerElement = document.querySelector(`[data-layer-name="${layerName}"]`)

  if (layerElement) {
    // Prevent updateCurrentLayer from interfering during manual scroll
    isManualScrolling.value = true

    // Update current layer indicator
    currentLayerIndex.value = index

    // Use scrollIntoView which works regardless of container structure
    layerElement.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
      inline: 'nearest'
    })

    // Reset manual scrolling flag after scroll completes
    setTimeout(() => {
      isManualScrolling.value = false
      currentLayerIndex.value = -1
    }, 1000)

    return
  }

  // Fall back to direct container scrolling.

  if (layerElement) {
    // Get the scrollable container
    let container = state.ui.scrollableRef

    if (!container) {
      // Try to find a scrollable container
      container = document.querySelector('.layers-container') ||
                  document.querySelector('.workspace') ||
                  document.querySelector('[data-scrollable]') ||
                  document.querySelector('.main-content') ||
                  document.documentElement
          }

    if (container) {

      // Check if container is actually scrollable
      const isScrollable = container.scrollHeight > container.clientHeight

      // Check if there are any CSS constraints preventing scroll
      const computedStyle = window.getComputedStyle(container)

      if (!isScrollable) {
                // Try to find a scrollable parent
        let parent = container.parentElement
        while (parent && parent !== document.body) {
          const parentStyle = window.getComputedStyle(parent)
          const parentScrollable = parent.scrollHeight > parent.clientHeight &&
                                 (parentStyle.overflow === 'auto' || parentStyle.overflow === 'scroll' ||
                                  parentStyle.overflowY === 'auto' || parentStyle.overflowY === 'scroll')

          if (parentScrollable) {
                        container = parent
            break
          }
          parent = parent.parentElement
        }

        // If still no scrollable container, use window
        if (!isScrollable && container.scrollHeight <= container.clientHeight) {
                    const elementRect = layerElement.getBoundingClientRect()
          window.scrollTo({
            top: window.scrollY + elementRect.top - 100,
            behavior: 'smooth'
          })

          // Update current layer indicator
          currentLayerIndex.value = index
          isManualScrolling.value = true

          setTimeout(() => {
            isManualScrolling.value = false
            currentLayerIndex.value = -1
          }, 1000)
          return
        }
      }

      // Prevent updateCurrentLayer from interfering during manual scroll
      isManualScrolling.value = true

      // Use getBoundingClientRect for more accurate positioning
      const containerRect = container.getBoundingClientRect()
      const elementRect = layerElement.getBoundingClientRect()
      let scrollTop = container.scrollTop + elementRect.top - containerRect.top - 20


      // Update current layer indicator
      currentLayerIndex.value = index

      // Prefer the workspace scroll container when available.
      let workspaceContainer = document.querySelector('.workspace') ||
                              document.querySelector('.blueprint-workspace') ||
                              document.querySelector('[data-scrollable]')

      if (workspaceContainer && workspaceContainer !== container) {

        const workspaceScrollable = workspaceContainer.scrollHeight > workspaceContainer.clientHeight
        if (workspaceScrollable) {
                    container = workspaceContainer

          // Recalculate position for workspace container
          const containerRect = container.getBoundingClientRect()
          const elementRect = layerElement.getBoundingClientRect()
          scrollTop = container.scrollTop + elementRect.top - containerRect.top - 20
        }
      }

      const targetScrollTop = Math.max(0, scrollTop)

      container.style.scrollBehavior = 'auto'
      container.scrollTop = targetScrollTop

      setTimeout(() => {

        // Retry with smooth scrolling if the direct update was ignored.
        if (Math.abs(container.scrollTop - targetScrollTop) > 10) {
                    container.style.scrollBehavior = 'smooth'
          container.scrollTo({
            top: targetScrollTop,
            behavior: 'smooth'
          })
        }
      }, 50)

      // Reset manual scrolling flag and current indicator after scroll completes
      setTimeout(() => {
        isManualScrolling.value = false
        currentLayerIndex.value = -1
      }, 1000)
    } else {
          }
  } else {
      }
}

// Toggle group expansion
const toggleGroupExpansion = (groupKey) => {
  if (expandedGroups.value.has(groupKey)) {
    expandedGroups.value.delete(groupKey)
  } else {
    expandedGroups.value.add(groupKey)
  }
}

// Expand all groups
const expandAllGroups = () => {
  const allGroupKeys = new Set()
  // Get all possible group keys
  const orderedLayers = []
  if (state.blueprints.loadedBlueprint?.categorization?.categories) {
    Object.values(state.blueprints.loadedBlueprint.categorization.categories).forEach(category => {
      if (category.layers && Array.isArray(category.layers)) {
        orderedLayers.push(...category.layers)
      }
    })
  } else {
    orderedLayers.push(...(state.blueprints.loadedBlueprint?.layers || []))
  }

  const blocksPerGroup = Math.ceil(orderedLayers.length / MAX_VISIBLE_BLOCKS)
  for (let i = 0; i < MAX_VISIBLE_BLOCKS; i++) {
    const startIndex = i * blocksPerGroup
    const endIndex = Math.min(startIndex + blocksPerGroup - 1, orderedLayers.length - 1)
    if (startIndex < orderedLayers.length) {
      allGroupKeys.add(`${startIndex}-${endIndex}`)
    }
  }

  // Add all group keys to expanded groups
  expandedGroups.value = allGroupKeys
}

// Collapse all groups
const collapseAllGroups = () => {
  expandedGroups.value.clear()
}

// Handle group clicks - expand/collapse or scroll to layer
const scrollToGroup = (block) => {
  if (block.type === 'group') {
    // Toggle expansion on group click
    toggleGroupExpansion(block.groupKey)
  } else {
    // For individual layers, scroll to them
    scrollToLayer(block.layer.name, block.index)
  }
}

// Get layer module count
const getLayerModuleCount = (layer) => {
  const preCount = layer.preModules?.length || 0
  const postCount = layer.postModules?.length || 0
  return preCount + postCount
}

const getGroupPreModuleCount = (block) =>
  block.layers.reduce((sum, layer) => sum + (layer.preModules?.length || 0), 0)

const getGroupPostModuleCount = (block) =>
  block.layers.reduce((sum, layer) => sum + (layer.postModules?.length || 0), 0)

// Format large numbers for display
const formatNumber = (num) => {
  if (!num) return '0'
  if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B'
  if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M'
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K'
  return num.toLocaleString()
}

// Get layer type abbreviation
const getLayerTypeAbbreviation = (layer) => {
  if (!layer || !layer.type) return 'UNK'

  const type = layer.type.toLowerCase()
  if (type.includes('attention') || type.includes('attn')) return 'ATTN'
  if (type.includes('embedding')) return 'EMB'
  if (type.includes('transformer') || type.includes('block')) return 'TRF'
  if (type.includes('feed_forward') || type.includes('ffn') || type.includes('mlp')) return 'FFN'
  if (type.includes('normalization') || type.includes('norm') || type.includes('layernorm')) return 'NORM'
  if (type.includes('dropout')) return 'DROP'
  if (type.includes('linear') || type.includes('dense')) return 'LIN'
  if (type.includes('convolution') || type.includes('conv')) return 'CONV'
  if (type.includes('pooling') || type.includes('pool')) return 'POOL'
  if (type.includes('activation') || type.includes('relu') || type.includes('gelu')) return 'ACT'
  if (type.includes('output')) return 'OUT'
  if (type.includes('input')) return 'IN'

  // Fallback to first 3 letters
  return type.substring(0, 3).toUpperCase()
}

// Generate tooltip title for blocks
const getTooltipTitle = (block) => {
  if (block.type === 'group') {
    return `Group ${block.startIndex}-${block.endIndex}: ${block.totalModules} modules. ${block.isExpanded ? 'Click to collapse' : 'Click to expand'}`
  } else {
    let title = block.layer.name || 'Layer'
    if (block.layer.parameters) {
      title += ` (${formatNumber(block.layer.parameters)} params)`
    }
    if (block.layer.preModules?.length > 0) {
      title += ` [${block.layer.preModules.length} pre]`
    }
    if (block.layer.postModules?.length > 0) {
      title += ` [${block.layer.postModules.length} post]`
    }
    return title
  }
}

// Generate detailed tooltip content for layers
const getLayerTooltipContent = (layer) => {
  if (!layer) return 'Unknown layer'

  const parts = []
  parts.push(`<strong>${layer.name || 'Unnamed Layer'}</strong>`)
  parts.push(`Type: ${layer.type || 'Unknown'}`)

  if (layer.parameters) {
    parts.push(`Parameters: ${formatNumber(layer.parameters)}`)
  }

  if (layer.inputShape && Array.isArray(layer.inputShape)) {
    parts.push(`Input: ${layer.inputShape.join('×')}`)
  }

  if (layer.outputShape && Array.isArray(layer.outputShape)) {
    parts.push(`Output: ${layer.outputShape.join('×')}`)
  }

  if (layer.numHeads) {
    parts.push(`Heads: ${layer.numHeads}`)
  }

  if (layer.preModules && layer.preModules.length > 0) {
    parts.push(`Pre-modules: ${layer.preModules.length}`)
  }

  if (layer.postModules && layer.postModules.length > 0) {
    parts.push(`Post-modules: ${layer.postModules.length}`)
  }

  return parts.join('<br>')
}

// Track scroll position to highlight current visible layer
const updateCurrentLayer = () => {
  // Don't update during manual scrolling to prevent interference
  if (isManualScrolling.value) return

  if (!state.blueprints.loadedBlueprint || !state.blueprints.loadedBlueprint.layers) return

  // Try the stored reference first, then fallback
  let container = state.ui.scrollableRef
  if (!container) {
    container = document.querySelector('.layers-container') ||
                document.querySelector('.workspace') ||
                document.querySelector('[data-scrollable]') ||
                document.querySelector('.main-content')
  }

  if (!container) return
  const scrollTop = container.scrollTop
  const containerHeight = container.clientHeight
  const viewportCenter = scrollTop + containerHeight / 2

  // Find which layer is closest to viewport center
  let closestIndex = -1
  let closestDist = 999999999;

  (state.blueprints.loadedBlueprint.layers || []).forEach((layer, index) => {
    const element = document.querySelector(`[data-layer-name="${layer.name}"]`)
    if (element) {
      const elementTop = element.offsetTop
      const elementCenter = elementTop + element.offsetHeight / 2
      const distance = Math.abs(elementCenter - viewportCenter)

      if (distance < closestDist) {
        closestDist = distance
        closestIndex = index
      }
    }
  })

  if (closestIndex !== currentLayerIndex.value && currentLayerIndex.value === -1) {
    currentLayerIndex.value = closestIndex
  }
}

onMounted(() => {

  // Set up scroll listener with fallback
  const setupScrollListener = () => {
    let container = state.ui.scrollableRef
    if (!container) {
      container = document.querySelector('.workspace') ||
                  document.querySelector('[data-scrollable]') ||
                  document.querySelector('.main-content')
    }

    if (container) {
      container.addEventListener('scroll', updateCurrentLayer)
      updateCurrentLayer()
      return container
    }
    return null
  }

  // Try immediately and with delay for DOM to be ready
  let scrollContainer = setupScrollListener()
  if (!scrollContainer) {
    setTimeout(() => {
      scrollContainer = setupScrollListener()
    }, 100)
  }
})

onUnmounted(() => {
  // Remove listener from any container that might have it
  const containers = [
    state.ui.scrollableRef,
    document.querySelector('.workspace'),
    document.querySelector('[data-scrollable]'),
    document.querySelector('.main-content')
  ]

  containers.forEach(container => {
    if (container) {
      container.removeEventListener('scroll', updateCurrentLayer)
    }
  })
})

// Handle controls panel toggle
const handleControlsToggle = (event) => {
  isControlsCollapsed.value = event.detail.isCollapsed
}
const controlsPanelWidth = ref(50)
const handleControlsResize = (event) => {
  controlsPanelWidth.value = event.detail.width || 50
}

// Handle timeline visibility toggle
const isTimelineVisible = ref(true)
const handleTimelineToggle = (event) => {
  isTimelineVisible.value = event.detail.isVisible
}

// Set up event listener for controls panel toggle
onMounted(() => {
document.addEventListener('controls-panel-toggle', handleControlsToggle)
document.addEventListener('controls-panel-resize', handleControlsResize)
document.addEventListener('timeline-visibility-toggle', handleTimelineToggle)
document.addEventListener('expand-all-timeline-groups', handleTimelineExpandCollapse)
})

// Handle timeline expand/collapse events from transformer controls
const handleTimelineExpandCollapse = (event) => {
  if (event.detail.expand) {
    expandAllGroups()
  } else {
    collapseAllGroups()
  }
}

onUnmounted(() => {
document.removeEventListener('controls-panel-toggle', handleControlsToggle)
document.removeEventListener('controls-panel-resize', handleControlsResize)
document.removeEventListener('timeline-visibility-toggle', handleTimelineToggle)
document.removeEventListener('expand-all-timeline-groups', handleTimelineExpandCollapse)
})

</script>

<template>
  <div v-show="isTimelineVisible" class="timeline" :style="{ left: `${controlsPanelWidth}px` }">
    <div class="container">

      <div class="blocks">
        <div
          v-for="(block, blockIndex) in collapsedBlocks"
          :key="block.type === 'group' ? `group-${block.startIndex}-${block.endIndex}` : `layer-${block.index}`"
          class="layer-block"
          :class="{
            'current': block.type === 'layer' ? currentLayerIndex === block.index : (currentLayerIndex >= block.startIndex && currentLayerIndex <= block.endIndex),
            'processing': block.type === 'layer' ? (currentProcessingLayer === block.index && isInferenceRunning) : (currentProcessingLayer >= block.startIndex && currentProcessingLayer <= block.endIndex && isInferenceRunning),
            'has-modules': block.type === 'layer' ? getLayerModuleCount(block.layer) > 0 : block.totalModules > 0,
            'inference-active': isInferenceRunning,
            'group-block': block.type === 'group',
            'expanded-child': block.isExpandedChild,
            'expanded-group': block.isExpanded,
            'first-block': blockIndex === 0,
            'last-block': blockIndex === collapsedBlocks.length - 1
          }"
          @click="scrollToGroup(block)"
          :title="getTooltipTitle(block)">
          <div v-if="block.type === 'layer' && block.layer.preModules?.length > 0" class="module-count-pre">
            {{ block.layer.preModules.length }}
          </div>
          <div v-else-if="block.type === 'group' && getGroupPreModuleCount(block) > 0" class="module-count-pre">
            {{ getGroupPreModuleCount(block) }}
          </div>

          <!-- Post-layer module count (bottom) -->
          <div v-if="block.type === 'layer' && block.layer.postModules?.length > 0" class="module-count-post">
            {{ block.layer.postModules.length }}
          </div>
          <div v-else-if="block.type === 'group' && getGroupPostModuleCount(block) > 0" class="module-count-post">
            {{ getGroupPostModuleCount(block) }}
          </div>

          <!-- Center layer count -->
          <div class="layer-count">
            <template v-if="block.type === 'group'">
              {{ block.layerCount }}
            </template>
            <template v-else>
              {{ getLayerTypeAbbreviation(block.layer) }}
            </template>
          </div>

          <!-- Processing indicator -->
          <div v-if="block.type === 'layer' ? (currentProcessingLayer === block.index && isInferenceRunning) : (currentProcessingLayer >= block.startIndex && currentProcessingLayer <= block.endIndex && isInferenceRunning)" class="processing-indicator">
            <div class="processing-dot"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.timeline {
  position: fixed;
  top: 40px;
  left: 50px;
  width: 60px;
  height: calc(100vh - 40px);
  background: var(--va-background-element);
  border-right: 1px solid var(--va-background-border);
  z-index: 100;
  transition: left 0.3s ease;
  overflow: hidden;
}

.container {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.blocks {
  flex: 1;
  overflow-y: scroll;
  overflow-x: hidden;
  padding: 8px 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-height: 100%;
}

.layer-block {
  position: relative;
  height: 32px;
  margin: 0 4px;
  background: transparent;
  border-radius: 0;
  border-bottom: 1px solid var(--va-background-border);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  transition: background-color 0.2s ease;
  font-size: 11px;
  color: var(--va-text-secondary);
  overflow: visible;
}

.layer-block:hover {
  background: var(--va-background-secondary);
}

/* Tooltip content styling (for VaPopover) */
.tooltip-content {
  background: var(--va-background-element);
  border: 2px solid var(--va-primary);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
  min-width: 220px;
  max-width: 280px;
}

.tooltip-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.layer-number {
  font-weight: 700;
  font-size: 13px;
  color: var(--va-primary);
}

.layer-type {
  font-size: 10px;
  padding: 2px 6px;
  background: var(--va-background-secondary);
  border-radius: 4px;
  color: var(--va-text-secondary);
  text-transform: uppercase;
  font-weight: 600;
}

.tooltip-name {
  font-size: 11px;
  color: var(--va-text-primary);
  font-weight: 500;
  margin-bottom: 8px;
  word-break: break-word;
}

.tooltip-divider {
  height: 1px;
  background: var(--va-background-border);
  margin: 8px 0;
}

.tooltip-stats {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
}

.stat-label {
  color: var(--va-text-secondary);
  font-weight: 500;
}

.stat-value {
  font-weight: 700;
  color: var(--va-text-primary);
  padding: 2px 6px;
  background: var(--va-background-secondary);
  border-radius: 4px;
  min-width: 30px;
  text-align: center;
}

.stat-pre {
  background: rgba(var(--va-info-rgb, 33, 150, 243), 0.15);
  color: var(--va-info);
}

.stat-post {
  background: rgba(var(--va-success-rgb, 76, 175, 80), 0.15);
  color: var(--va-success);
}

.stat-total {
  background: rgba(var(--va-primary-rgb, 23, 126, 255), 0.15);
  color: var(--va-primary);
}

.tooltip-hint {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--va-background-border);
  font-size: 10px;
  color: var(--va-text-secondary);
  font-style: italic;
  text-align: center;
}

.layer-block.current {
  background: var(--va-background-primary);
  color: var(--va-text-primary);
}

.layer-block.processing {
  background: var(--va-warning);
  color: var(--va-text-primary);
  animation: processingPulse 1.5s ease-in-out infinite;
}

.layer-block.has-modules {
  background: var(--va-background-tertiary);
}

.layer-block.has-modules.current {
  background: var(--va-background-primary);
  box-shadow: 0 0 0 2px var(--va-primary);
}

.layer-block.has-modules.processing {
  background: var(--va-warning);
}

/* Module count positioning */
.module-count-pre {
  position: absolute;
  top: 1px;
  right: 1px;
  font-size: 6px;
  font-weight: 400;
  color: white;
  background: var(--va-info);
  border-radius: 2px;
  padding: 1px 2px;
  line-height: 1.2;
  min-width: 10px;
  text-align: center;
  z-index: 10;
}

.module-count-post {
  position: absolute;
  bottom: 1px;
  right: 1px;
  font-size: 6px;
  font-weight: 400;
  color: white;
  background: var(--va-success);
  border-radius: 2px;
  padding: 1px 2px;
  line-height: 1.2;
  min-width: 10px;
  text-align: center;
  z-index: 10;
}

/* Make sure current state doesn't override module badge colors */
.layer-block.current .module-count-pre,
.layer-block.current .module-count-post {
  background: var(--va-info);
  color: white;
}


/* Center layer count indicator */
.layer-count {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 10px;
  font-weight: 700;
  color: var(--va-text-primary);
  background: rgba(255, 255, 255, 0.85);
  border-radius: 3px;
  padding: 1px 4px;
  line-height: 1.2;
  z-index: 9;
}


/* Ensure current state doesn't affect module badges */
.layer-block.current .module-count-pre,
.layer-block.current .module-count-post {
  background: var(--va-info);
  color: white;
}

.layer-block.processing .layer-count {
  color: var(--va-warning);
  background: #ffffff;
}

.layer-block:not(.group-block).current .module-count-pre,
.layer-block:not(.group-block).current .module-count-post {
  background: var(--va-info);
  color: white;
}

.layer-block:not(.group-block).processing .module-count-pre,
.layer-block:not(.group-block).processing .module-count-post {
  background: var(--va-warning);
  color: white;
}

/* Group block specific styles */
.layer-block.group-block {
  background: transparent;
  border-left: 0;
  border-radius: 0;
  border-bottom: 1px solid var(--va-background-border);
  height: 36px;
  position: relative;
  cursor: pointer;
  margin-bottom: 3px;
  transition: background-color 0.2s ease;
}

.layer-block.group-block:hover {
  background: var(--va-background-primary);
}

.layer-block.group-block.current {
  background: var(--va-background-primary);
}

.layer-block.group-block.processing {
  background: var(--va-warning);
  animation: processingPulse 1.5s ease-in-out infinite;
}

.layer-block.group-block.has-modules {
  background: var(--va-background-tertiary);
}

.layer-block.group-block.expanded-group {
  background: var(--va-background-secondary);
}

/* Larger font for group blocks */
.layer-block.group-block .layer-count {
  font-size: 13px;
  font-weight: 600;
}

/* Expanded child layer styles */
.layer-block.expanded-child {
  background: transparent;
  margin-left: 4px;
  height: 32px; /* Same height as regular layer blocks */
  min-height: 32px;
  flex-shrink: 0; /* Prevent shrinking */
}

.layer-block.expanded-child:hover {
  background: var(--va-background-secondary);
  transform: translateX(1px);
}

/* Ensure proper spacing when groups are expanded */
.blocks:has(.expanded-child) {
  gap: 2px;
}

/* Group block when expanded should maintain its size */
.layer-block.group-block {
  flex-shrink: 0;
}


/* Processing indicator */
.processing-indicator {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 2;
}

.processing-dot {
  width: 6px;
  height: 6px;
  background: white;
  border-radius: 50%;
  animation: pulse 1s ease-in-out infinite;
}

/* Module indicator dot */
.module-indicator {
  position: absolute;
  bottom: 2px;
  left: 2px;
}

.module-dot {
  width: 4px;
  height: 4px;
  background: var(--va-info);
  border-radius: 50%;
  animation: blink 2s ease-in-out infinite;
}

.layer-block.current .module-dot {
  background: white;
}

.layer-block.processing .module-dot {
  background: white;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

@keyframes processingPulse {
  0%, 100% {
    background: var(--va-warning);
    transform: scale(1);
  }
  50% {
    background: var(--va-warning);
    transform: scale(1.02);
  }
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}




.layer-index-multiline {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100%;
  font-weight: bold;
  font-size: 12px;
  line-height: 1.2;
  text-align: center;
}

.layer-number-line {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  min-height: 20px;
}

/* Enhanced scrollbar styling for better visibility */
.blocks::-webkit-scrollbar {
  width: 6px;
}

.blocks::-webkit-scrollbar-track {
  background: var(--va-background-element);
  border-radius: 3px;
}

.blocks::-webkit-scrollbar-thumb {
  background: var(--va-background-border);
  border-radius: 3px;
  transition: background 0.2s ease;
}

.blocks::-webkit-scrollbar-thumb:hover {
  background: var(--va-primary);
}

/* Ensure timeline container can handle expanded content */
.timeline {
  overflow: hidden;
}

/* Smooth scrolling */
.blocks {
  scroll-behavior: smooth;
}
</style>
