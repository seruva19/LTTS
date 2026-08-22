<script setup>
import { useLttsState } from '../state'
import { useColors } from "vuestic-ui"
import { storeToRefs } from 'pinia'
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useModelLogStore } from '@/state/model-log-store'

const state = useLttsState()
const modelLogStore = useModelLogStore()
const { hasNewEntries } = storeToRefs(modelLogStore)

// UI visibility states
const isTimelineVisible = ref(true)
const areLabelsVisible = ref(true)
const showAdvancedTuning = ref(false)

// Computed properties for button states
const canRun = computed(() => {
  return state.blueprints.loadedBlueprint &&
         state.blueprints.loadedBlueprint.layers &&
         state.blueprints.loadedBlueprint.layers.length > 0 &&
         !state.ws.isInferenceRunning
})

const canStop = computed(() => {
  return state.ws.isInferenceRunning
})

// UI toggle functions
const toggleTimeline = () => {
  isTimelineVisible.value = !isTimelineVisible.value
  // Dispatch event to timeline component
  const event = new CustomEvent('timeline-visibility-toggle', {
    detail: { isVisible: isTimelineVisible.value }
  })
  document.dispatchEvent(event)
}

const toggleLabels = () => {
  areLabelsVisible.value = !areLabelsVisible.value
  // Dispatch event to blueprint component
  const event = new CustomEvent('section-labels-toggle', {
    detail: { isVisible: areLabelsVisible.value }
  })
  document.dispatchEvent(event)
}

const openAdvancedTuning = () => {
  showAdvancedTuning.value = true
  // Dispatch event to open advanced tuning modal
  const event = new CustomEvent('advanced-tuning-toggle', {
    detail: { isOpen: showAdvancedTuning.value }
  })
  document.dispatchEvent(event)
}

// Blueprint execution functions
const runBlueprint = () => {
  if (!canRun.value) return
  return state.blueprints.runLoadedBlueprint()
}

const stopBlueprint = () => {
  if (!canStop.value) return
  state.blueprints.stopLoadedBlueprint()
}

// Expand/Collapse all functionality
const expandAll = () => {
  if (state.blueprints.loadedBlueprint?.layers) {
    // The blueprint component keeps expandedLayers locally rather than in the
    // store, so expansion is driven by a DOM event it listens for.
    const event = new CustomEvent('expand-all-layers')
    document.dispatchEvent(event)
  }
}

const collapseAll = () => {
  // Similarly for collapse
  const event = new CustomEvent('collapse-all-layers')
  document.dispatchEvent(event)
}

const toggleModelLogPanel = () => {
  modelLogStore.togglePanel()
}

</script>

<template>
  <div class="toolbar">
    <VaButton
      size="small"
      preset="primary"
      @click="state.ui.scrollToTop"
      class="button"
      icon="keyboard_arrow_up"
      title="Scroll to top"
    />

    <VaButton
      size="small"
      preset="secondary"
      @click="toggleModelLogPanel"
      :class="['button', 'model-log-button', { 'model-log-button--alert': hasNewEntries }]"
      icon="list"
      title="Model event log"
    />

    <VaButton
      size="small"
      preset="primary"
      @click="state.ui.scrollToBottom"
      class="button scroll-bottom"
      icon="keyboard_arrow_down"
      title="Scroll to bottom"
    />

    <VaButton
      size="small"
      preset="secondary"
      @click="expandAll"
      class="button"
      icon="unfold_more"
      title="Expand all layers"
    />

    <VaButton
      size="small"
      preset="secondary"
      @click="collapseAll"
      class="button"
      icon="unfold_less"
      title="Collapse all layers"
    />

    <!-- UI Toggle Buttons -->
    <VaButton
      size="small"
      :preset="isTimelineVisible ? 'primary' : 'secondary'"
      @click="toggleTimeline"
      class="button"
      icon="visibility"
      :title="isTimelineVisible ? 'Hide timeline panel' : 'Show timeline panel'"
    />

    <VaButton
      size="small"
      :preset="areLabelsVisible ? 'primary' : 'secondary'"
      @click="toggleLabels"
      class="button"
      icon="route"
      :title="areLabelsVisible ? 'Hide section labels' : 'Show section labels'"
    />

    <VaButton
      size="small"
      preset="secondary"
      @click="openAdvancedTuning"
      class="button"
      icon="tune"
      title="Advanced section tuning"
    />

    <VaButton
      size="small"
      :disabled="!canRun"
      preset="primary"
      @click="runBlueprint"
      class="button run"
      icon="play_arrow"
      :title="canRun ? 'Run blueprint inference' : 'No blueprint loaded or inference running'"
      :loading="state.ws.isInferenceRunning"
    >
      {{ state.ws.isInferenceRunning ? 'Running...' : 'Run' }}
    </VaButton>

    <VaButton
      size="small"
      :disabled="!canStop"
      preset="primary"
      @click="stopBlueprint"
      class="button"
      icon="stop"
      :title="canStop ? 'Stop blueprint' : 'No inference running'"
      color="danger"
    />
  </div>
</template>

<style scoped>
  .toolbar {
    height: 30px;
    text-align: center;
    position: fixed;
    padding: 3px;
    top: 40px;
    left: 90px;
    width: calc(100% - 140px);

    font-size: 11px;
    background: var(--va-background-element);
  }

  .toolbar .button {
    margin: 0 2px;
    border-radius: 0;
    height: calc(100% - 14px);
  }

  .model-log-button {
    position: relative;
  }

  .model-log-button.model-log-button--alert::after {
    content: "";
    position: absolute;
    top: 3px;
    right: 3px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--va-danger);
  }

  .toolbar .button.run {
    margin-left: 50px;
    min-width: 60px;
    font-weight: 600;
    background: var(--va-primary);
    color: white;
  }

  .toolbar .button.run:hover {
    background: var(--va-primary-dark);
  }

  .toolbar .button.scroll-bottom {
    margin-right: 50px;
  }

  .menu-bar {
    width: calc(100% - 100px);
  }

  .menu-root {
    color: var(--va-text-primary) !important;
    background: none !important;
    margin: 10px;
    padding: 0 10px;
  }

  .menu-root:hover {

    color: var(--va-focus) !important;
  }

  .color-mode-switch{
    position: absolute;
    right: 5px;
    top: 5px;
    z-index: 1000;
    background-color: var(--va-background-border);
    border-radius: 50px;
    padding: 3px;
  }
</style>
