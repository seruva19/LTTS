<script setup>
import Menu from './components/menu.vue'
import Toolbar from './components/toolbar.vue'
import Timeline from './components/timeline.vue'
import OutputTimeline from './components/output-timeline.vue'
import Workspace from './components/workspace.vue'
import Modals from './components/modal-windows.vue'
import StatusBar from './components/status-bar.vue'
import ModulesBrowser from './components/modules-browser.vue'
import TransformerControls from './components/transformer-controls.vue'
import AdvancedTuning from './components/modals/advanced-tuning.vue'
import { useLauncher } from './launch'
import { useWebSocketStore } from '@/state/socket-store'
import { useModelStore } from '@/state/model-store'
import { useLttsState } from '@/state'
import { ref, onMounted, onUnmounted, computed, nextTick } from 'vue'

const appInitializer = useLauncher()
const wsStore = useWebSocketStore()
const modelStore = useModelStore()
const state = useLttsState()
const showModulesBrowser = ref(false)
const preselectedModule = ref(null)
const moduleFilter = ref(null)
const moduleBrowserMode = ref('explore') // 'explore' or 'attach'

// Global loading states
const isInferenceRunning = computed(() => state.ws.isInferenceRunning)
const isModelLoading = computed(() => modelStore.isModelLoading)
const currentLoadingModelId = computed(() => modelStore.currentLoadingModelId)

// Model loading steps tracking
const modelLoadingSteps = ref({
  modelLoaded: false,
  blueprintLoaded: false,
  uiRendered: false
})

// Listen for modules browser open event
const openModulesBrowser = (event) => {
  if (event.detail && event.detail.moduleId) {
    preselectedModule.value = event.detail.moduleId
  }
  if (event.detail && event.detail.filter) {
    moduleFilter.value = event.detail.filter
    moduleBrowserMode.value = 'attach' // When filter is provided, we're in attach mode
  } else {
    moduleFilter.value = null
    moduleBrowserMode.value = 'explore' // No filter means explore mode
  }
  showModulesBrowser.value = true
}

// Handle module attachment
const handleModuleAttach = (attachmentData) => {
  try {
    state.modules.attachModule(attachmentData)
    showModulesBrowser.value = false
    preselectedModule.value = null
    moduleFilter.value = null
  } catch (error) {
    console.error('Failed to attach module:', error)
  }
}

// Handle model loading events
const handleModelLoading = (event) => {
  const { modelId, status } = event.detail

  if (status === 'model-loaded') {
    modelLoadingSteps.value.modelLoaded = true
  }
}

// Handle blueprint loaded event
const handleBlueprintLoaded = (event) => {
  modelLoadingSteps.value.blueprintLoaded = true
}

// Handle UI fully rendered event
const handleUIRendered = (event) => {
  modelLoadingSteps.value.uiRendered = true

  // Reset loading steps after a delay
  setTimeout(() => {
    modelLoadingSteps.value = {
      modelLoaded: false,
      blueprintLoaded: false,
      uiRendered: false
    }
  }, 2000)
}

onMounted(() => {
  document.addEventListener('open-modules-browser', openModulesBrowser)
  document.addEventListener('model-loading', handleModelLoading)
  document.addEventListener('blueprint-loaded', handleBlueprintLoaded)
  document.addEventListener('ui-fully-rendered', handleUIRendered)

  // Set UI ready state once all child components are mounted and rendered
  nextTick(() => {
    modelStore.setUIReady(true)
  })
})

// Clean up event listeners
onUnmounted(() => {
  document.removeEventListener('open-modules-browser', openModulesBrowser)
  document.removeEventListener('model-loading', handleModelLoading)
  document.removeEventListener('blueprint-loaded', handleBlueprintLoaded)
  document.removeEventListener('ui-fully-rendered', handleUIRendered)
})
</script>

<template>
  <div v-if="appInitializer.isInitializing">
    Loading application...
  </div>
  <div v-else-if="appInitializer.error">
    {{ appInitializer.error }}
  </div>
  <template v-else>
    <!-- Global inference loading indicator (non-intrusive) -->
    <div v-if="isInferenceRunning" class="global-loading-indicator">
      <div class="loading-content">
        <VaProgressCircle indeterminate size="small" color="primary" />
        <div class="loading-text">Running Inference...</div>
      </div>
    </div>

    <!-- Model loading indicator (more prominent and persistent) -->
    <div v-if="isModelLoading" class="model-loading-indicator">
      <div class="loading-content">
        <VaProgressCircle indeterminate size="medium" color="primary" />
        <div class="loading-text">
          <div class="loading-title">Loading Model</div>
          <div class="loading-subtitle">{{ currentLoadingModelId }}</div>
          <div class="loading-message">
            <div>Please wait while the model and UI are being prepared...</div>
            <div class="loading-steps">
              <div class="loading-step" :class="{ 'completed': modelLoadingSteps.modelLoaded }">
                <VaIcon name="check_circle" v-if="modelLoadingSteps.modelLoaded" color="success" />
                <VaIcon name="pending" v-else color="primary" />
                <span>Loading model from server</span>
              </div>
              <div class="loading-step" :class="{ 'completed': modelLoadingSteps.blueprintLoaded }">
                <VaIcon name="check_circle" v-if="modelLoadingSteps.blueprintLoaded" color="success" />
                <VaIcon name="pending" v-else color="primary" />
                <span>Generating model blueprint</span>
              </div>
              <div class="loading-step" :class="{ 'completed': modelLoadingSteps.uiRendered }">
                <VaIcon name="check_circle" v-if="modelLoadingSteps.uiRendered" color="success" />
                <VaIcon name="pending" v-else color="primary" />
                <span>Rendering UI components</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <Modals />
    <TransformerControls />
    <Menu></Menu>
    <Toolbar />
    <Workspace />
    <Timeline />
    <OutputTimeline />
    <StatusBar />

    <!-- Modules Browser Modal -->
    <VaModal
      v-model="showModulesBrowser"
      size="large"
      :close-on-click-outside="false"
      hide-default-actions
    >
      <template #header>
        <h2>{{ moduleBrowserMode === 'attach' ? 'Attach Module' : 'Modules Browser' }}</h2>
      </template>
      <ModulesBrowser
        :preselected-module="preselectedModule"
        :filter="moduleFilter"
        :mode="moduleBrowserMode"
        @close="showModulesBrowser = false; preselectedModule = null; moduleFilter = null"
        @attach="handleModuleAttach"
      />
    </VaModal>

    <!-- Advanced Tuning Modal -->
    <AdvancedTuning />
  </template>
</template>

<style scoped>
.global-loading-indicator {
  position: fixed;
  top: 80px;
  right: 20px;
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  pointer-events: none;
}

.loading-content {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
}

.loading-text {
  font-size: 14px;
  font-weight: 500;
  color: var(--va-text-primary);
  white-space: nowrap;
}

/* Model loading indicator styles */
.model-loading-indicator {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  z-index: 1100;
  padding: 24px;
  min-width: 300px;
  text-align: center;
}

.model-loading-indicator .loading-content {
  flex-direction: column;
  gap: 16px;
}

.model-loading-indicator .loading-text {
  display: flex;
  flex-direction: column;
  gap: 8px;
  white-space: normal;
}

.loading-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--va-text-primary);
}

.loading-subtitle {
  font-size: 14px;
  font-weight: 500;
  color: var(--va-text-secondary);
  word-break: break-all;
}

.loading-message {
  font-size: 14px;
  color: var(--va-text-secondary);
  margin-top: 8px;
}

.loading-steps {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-align: left;
}

.loading-step {
  display: flex;
  align-items: center;
  gap: 8px;
  opacity: 0.7;
  transition: opacity 0.3s ease;
}

.loading-step.completed {
  opacity: 1;
}

.loading-step span {
  font-size: 14px;
}
</style>
