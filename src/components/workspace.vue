<script setup>
import { useLttsState } from '@/state'
import { ref, onMounted, computed, watch, onUnmounted } from 'vue'
import Blueprint from '@/components/blocks/blueprint.vue'

const state = useLttsState()
const isNavPanelCollapsed = ref(false)
const isTimelineVisible = ref(true)
const leftPanelWidth = ref(50)
const rightPanelWidth = ref(50)
const layerRailWidth = 60

const isModelListVisible = computed(() => {
  return state.models.showModelList
})

const handleControlsToggle = (event) => {
  isNavPanelCollapsed.value = event.detail.isCollapsed
}

const handleTimelineToggle = (event) => {
  isTimelineVisible.value = event.detail.isVisible
}
const handleControlsResize = (event) => { leftPanelWidth.value = event.detail.width || 50 }
const handleArtifactResize = (event) => { rightPanelWidth.value = event.detail.width || 50 }
const workspaceStyle = computed(() => {
  const leftOffset = leftPanelWidth.value + (isTimelineVisible.value ? layerRailWidth : 0)
  const rightOffset = isTimelineVisible.value ? rightPanelWidth.value : 0
  return {
    left: `${leftOffset}px`,
    width: `calc(100vw - ${leftOffset}px - ${rightOffset}px)`
  }
})

onMounted(() => {
  document.addEventListener('controls-panel-toggle', handleControlsToggle)
  document.addEventListener('timeline-visibility-toggle', handleTimelineToggle)
  document.addEventListener('controls-panel-resize', handleControlsResize)
  document.addEventListener('artifact-panel-resize', handleArtifactResize)
})

onUnmounted(() => {
  document.removeEventListener('controls-panel-toggle', handleControlsToggle)
  document.removeEventListener('timeline-visibility-toggle', handleTimelineToggle)
  document.removeEventListener('controls-panel-resize', handleControlsResize)
  document.removeEventListener('artifact-panel-resize', handleArtifactResize)
})

</script>

<template>
  <div class="workspace" :style="workspaceStyle" :class="{ 'controls-collapsed': isNavPanelCollapsed, 'timeline-hidden': !isTimelineVisible }">
    <div class="workspace-content">
      <div class="model-display">
        <Blueprint />
      </div>
    </div>
  </div>
</template>

<style scoped>
.workspace {
  position: fixed;
  left: 105px;
  top: 70px;
  width: calc(100vw - 90px - 65px);
  height: calc(100vh - 70px);
  background: var(--va-background-primary);
  overflow: hidden;
  transition: left 0.3s ease, width 0.3s ease;
}

.workspace-content {
  display: flex;
  height: 100%;
}

.model-display {
  flex: 1;
  overflow: hidden;
}

.controls {
  padding: 1rem;
}
</style>
