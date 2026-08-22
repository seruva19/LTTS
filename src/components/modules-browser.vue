<template>
  <div class="modules-browser">
    <!-- Fixed Header -->
    <div class="browser-header">
      <!-- Controls Row: Toggle and Dropdown on same line -->
      <div class="header-controls">
        <!-- Source Selector (only in explore mode) -->
        <div v-if="mode === 'explore'" class="source-selector">
          <VaSelect
            v-model="moduleSource"
            :options="sourceOptions"
            text-by="text"
            value-by="value"
            @update:modelValue="handleSourceChange"
          >
            <template #prepend-inner>
              <VaIcon name="cloud" />
            </template>
          </VaSelect>
        </div>

        <!-- Module Selection -->
        <div v-if="!isLoading" class="module-selection">
          <VaSelect
            v-model="selectedModuleId"
            :options="moduleOptions"
            text-by="text"
            value-by="value"
            placeholder="Select a module..."
            clearable
          >
            <template #prepend-inner>
              <VaIcon name="extension" />
            </template>
          </VaSelect>
        </div>

        <!-- Loading State -->
        <div v-if="isLoading" class="loading-spinner">
          <VaProgressCircular indeterminate size="small" />
        </div>
      </div>

      <!-- Attach Mode Filter Info -->
      <VaAlert
        v-if="mode === 'attach' && filter"
        color="info"
        border="left"
        class="header-alert"
      >
        <strong>Attaching for:</strong> {{ getFilterDescription() }}
      </VaAlert>

      <!-- Empty State: No modules available -->
      <VaAlert
        v-if="moduleOptions.length === 0 && !isLoading"
        color="warning"
        border="left"
        class="header-alert"
      >
        {{ moduleSource === 'local'
          ? 'No modules available. Please open a Transformer model to load modules.'
          : 'No remote modules found. Please check registry URL in settings.' }}
      </VaAlert>
    </div>

    <!-- Scrollable Content -->
    <div class="browser-content">
      <!-- Module Visualization -->
      <div v-if="selectedModule" class="module-visualization">
        <!-- Remote Module Info Card -->
        <div v-if="moduleSource === 'remote'" class="remote-module-card">
          <VaCard>
            <VaCardTitle>{{ selectedModule.name }}</VaCardTitle>
            <VaCardContent>
              <div class="module-details">
                <p><strong>Version:</strong> {{ selectedModule.version }}</p>
                <p><strong>Author:</strong> {{ selectedModule.author }}</p>
                <p><strong>Description:</strong> {{ selectedModule.description }}</p>
                <p><strong>Size:</strong> {{ formatBytes(selectedModule.size_bytes) }}</p>
                <p><strong>Event Types:</strong> {{ selectedModule.event_types.join(', ') }}</p>
                <div v-if="selectedModule.tags.length > 0" class="module-tags">
                  <VaChip v-for="tag in selectedModule.tags" :key="tag" size="small">
                    {{ tag }}
                  </VaChip>
                </div>
                <div class="module-actions">
                  <VaButton
                    v-if="!selectedModule.installed"
                    color="primary"
                    :loading="isInstalling"
                    @click="handleInstall(selectedModule.name)"
                  >
                    <VaIcon name="download" class="mr-2" />
                    Install Module
                  </VaButton>
                  <VaButton
                    v-else
                    color="danger"
                    :loading="isUninstalling"
                    @click="handleUninstall(selectedModule.name)"
                  >
                    <VaIcon name="delete" class="mr-2" />
                    Uninstall Module
                  </VaButton>
                </div>
              </div>
            </VaCardContent>
          </VaCard>
        </div>

        <!-- Local Module Visualization -->
        <div v-else>
          <ModuleStructureVisualizer :key="selectedModuleId" :module="selectedModule" />
        </div>
      </div>

      <!-- No Module Selected -->
      <div v-else-if="!isLoading" class="no-selection">
        <VaIcon name="extension" size="3rem" color="secondary" />
        <h3>Select a Module to Explore</h3>
        <p>{{ moduleSource === 'local'
          ? 'Choose a module from the dropdown to view its structure and UI schema'
          : 'Choose a module from the dropdown to view its details and install it' }}</p>
      </div>
    </div>

    <!-- Fixed Footer -->
    <div class="browser-footer">
      <VaButton preset="secondary" @click="emitClose">Close</VaButton>
      <VaButton
        v-if="mode === 'attach'"
        preset="primary"
        :disabled="!selectedModuleId"
        @click="handleAttach"
      >
        Attach Module
      </VaButton>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useLttsState } from '@/state'
import { useSettingsStore } from '@/state/settings-store'
import { useToast } from '@/composables/useToast'
import ModuleStructureVisualizer from '@/components/ui-renderer/module-structure-visualizer.vue'
import axios from 'axios'
import { baseUrl } from '@/constants'
import { MODULE_REGISTRY_URL } from '@/state/settings-defaults'

const state = useLttsState()
const settingsStore = useSettingsStore()
const toast = useToast()
const selectedModuleId = ref(null)
const moduleSource = ref('local')
const remoteModules = ref([])
const isLoading = ref(false)
const isInstalling = ref(false)
const isUninstalling = ref(false)

// Props
const props = defineProps({
  preselectedModule: {
    type: String,
    default: null
  },
  filter: {
    type: Object,
    default: null
  },
  mode: {
    type: String,
    default: 'explore', // 'explore' or 'attach'
    validator: (value) => ['explore', 'attach'].includes(value)
  }
})
const emit = defineEmits(['close', 'attach'])

// Source options
const sourceOptions = [
  { text: 'Local Modules', value: 'local' },
  { text: 'Remote Registry', value: 'remote' }
]

// Computed properties
const moduleOptions = computed(() => {
  if (moduleSource.value === 'local') {
    if (!state.modules.modules) return []

    const all = Object.entries(state.modules.modules)
    const filtered = all.filter(([moduleName, module]) => {
      const events = module.metadata?.event_types || []
      if (!props.filter) return true
      if (props.filter.eventType) {
        return events.length === 0 || events.includes(props.filter.eventType)
      }
      if (props.filter.scope === 'layer') {
        return events.length === 0 || events.includes('layer_before') || events.includes('layer_after')
      }
      if (props.filter.scope === 'model') {
        return events.includes('model_before') || events.includes('model_after')
      }
      return true
    })
    return filtered.map(([moduleName]) => ({ text: moduleName, value: moduleName }))
  } else {
    const filteredRemote = remoteModules.value.filter(module => {
      const events = module.event_types || []
      if (!props.filter) return true
      if (props.filter.eventType) {
        return events.length === 0 || events.includes(props.filter.eventType)
      }
      if (props.filter.scope === 'layer') {
        return events.length === 0 || events.includes('layer_before') || events.includes('layer_after')
      }
      if (props.filter.scope === 'model') {
        return events.includes('model_before') || events.includes('model_after')
      }
      return true
    })
    return filteredRemote.map(module => ({
      text: `${module.name} (v${module.version})${module.installed ? ' - Installed' : ''}`,
      value: module.name
    }))
  }
})

const selectedModule = computed(() => {
  if (!selectedModuleId.value) return null

  if (moduleSource.value === 'local') {
    if (!state.modules.modules) return null
    return state.modules.modules[selectedModuleId.value]
  } else {
    return remoteModules.value.find(m => m.name === selectedModuleId.value)
  }
})

// Methods
const emitClose = () => {
  emit('close')
}

const handleModuleSelection = (moduleId) => {
  selectedModuleId.value = moduleId
}

const handleAttach = () => {
  if (!selectedModuleId.value || !props.filter) return

  // Check if this is a layer-level module and we have layer information
  const isLayerLevel = props.filter.eventType === 'layer_before' || props.filter.eventType === 'layer_after'

  if (isLayerLevel && !props.filter.layerPath) {
    toast.error('Please select a layer first before attaching a layer-level module')
    return
  }

  const currentModelId =
    state.models.selectedModel?.id ||
    Object.values(state.models.loadedModels || {}).find(model => model?.loaded)?.id ||
    state.blueprints.loadedBlueprint?.source?.id ||
    null

  // Prepare attachment data
  const attachmentData = {
    name: selectedModuleId.value,
    eventType: props.filter.eventType,
    modelId: currentModelId,
    layerPath: props.filter.layerPath || null, // Include layer path for layer-level modules
    parameters: {} // Default empty parameters, can be configured later
  }

  // Emit attach event to parent
  emit('attach', attachmentData)
}

const handleSourceChange = async (source) => {
  selectedModuleId.value = null
  if (source === 'remote') {
    await fetchRemoteModules()
  }
}

const fetchRemoteModules = async () => {
  isLoading.value = true
  try {
    // Only override the backend when the user actually chose a registry. Left
    // at the default, the backend resolves it itself - LTTS_MODULES_REGISTRY,
    // the personal config, then a sibling LTTS-modules checkout - so someone
    // working from a local checkout sees their own modules rather than none.
    const registryUrl = settingsStore.settings?.modules?.registry_url
    const params = registryUrl && registryUrl !== MODULE_REGISTRY_URL
      ? { registry_url: registryUrl }
      : {}

    const response = await axios.get(`${baseUrl}/api/ltts/registry/modules`, { params })
    remoteModules.value = response.data.modules
  } catch (error) {
    console.error('Error fetching remote modules:', error)
    toast.error('Failed to fetch remote modules')
  } finally {
    isLoading.value = false
  }
}

const handleInstall = async (moduleName) => {
  isInstalling.value = true
  try {
    const response = await axios.post(`${baseUrl}/api/ltts/registry/modules/${moduleName}/install`)
    if (response.data.success) {
      toast.success(response.data.message)
      // Refresh remote modules to update installation status
      await fetchRemoteModules()
      // Update the selected module
      selectedModuleId.value = moduleName
    }
  } catch (error) {
    console.error('Error installing module:', error)
    toast.error(`Failed to install module: ${error.response?.data?.detail || error.message}`)
  } finally {
    isInstalling.value = false
  }
}

const handleUninstall = async (moduleName) => {
  isUninstalling.value = true
  try {
    const response = await axios.delete(`${baseUrl}/api/ltts/registry/modules/${moduleName}/uninstall`)
    if (response.data.success) {
      toast.success(response.data.message)
      // Refresh remote modules to update installation status
      await fetchRemoteModules()
      // Update the selected module
      selectedModuleId.value = moduleName
    }
  } catch (error) {
    console.error('Error uninstalling module:', error)
    toast.error(`Failed to uninstall module: ${error.response?.data?.detail || error.message}`)
  } finally {
    isUninstalling.value = false
  }
}

const formatBytes = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

const getFilterDescription = () => {
  if (!props.filter) return ''

  const eventTypeLabels = {
    'layer_before': 'Layer Before',
    'layer_after': 'Layer After',
    'model_before': 'Model Before',
    'model_after': 'Model After'
  }

  let description = ''

  if (props.filter.eventType) {
    description = eventTypeLabels[props.filter.eventType] || props.filter.eventType
  } else if (props.filter.scope === 'layer') {
    description = 'Layer-level modules'
  } else if (props.filter.scope === 'model') {
    description = 'Model-level modules'
  } else {
    description = 'Filtered modules'
  }

  // Add layer path if available
  if (props.filter.layerPath) {
    description += ` (Layer: ${props.filter.layerPath})`
  }

  return description
}

// Watch for module list changes
watch(() => state.modules.modules, () => {
  if (moduleSource.value === 'local') {
    selectedModuleId.value = null
  }
})

// Watch for preselected module
watch(() => props.preselectedModule, (newModuleId) => {
  if (newModuleId && state.modules.modules && newModuleId in state.modules.modules) {
    selectedModuleId.value = newModuleId
  }
}, { immediate: true })
</script>

<style scoped>
.modules-browser {
  display: flex;
  flex-direction: column;
  height: 70vh;
  min-height: 500px;
  max-height: 800px;
}

.browser-header {
  flex-shrink: 0;
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid var(--va-background-border);
  background: var(--va-background-element);
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.browser-content {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

.browser-footer {
  flex-shrink: 0;
  padding: 0.5rem 0.75rem;
  border-top: 1px solid var(--va-background-border);
  background: var(--va-background-element);
  display: flex;
  justify-content: flex-end;
}

.source-selector {
  flex-shrink: 0;
  min-width: 180px;
}

.source-selector :deep(.va-input-wrapper) {
  margin-bottom: 0;
}

.loading-spinner {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
}

.module-selection {
  flex: 1;
  min-width: 0;
}

.module-selection :deep(.va-input-wrapper) {
  margin-bottom: 0;
}

.header-alert {
  margin: 0.5rem 0 0 0;
}

.module-visualization {
  /* Content flows naturally */
}

.remote-module-card {
  padding: 1rem 0;
}

.module-details {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.module-details p {
  margin: 0;
  line-height: 1.6;
}

.module-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.module-actions {
  margin-top: 1.5rem;
  display: flex;
  gap: 1rem;
}

.mute-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.no-selection {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem 1rem;
  text-align: center;
  color: var(--va-text-secondary);
  background: var(--va-background-element);
  border-radius: 8px;
  border: 1px dashed var(--va-background-border);
}

.no-selection h3 {
  margin: 1rem 0 0.5rem 0;
  color: var(--va-text-primary);
}

.no-selection p {
  margin: 0;
  max-width: 400px;
}
</style>

<style>
/* Global styles to ensure modal appears above everything */
.va-modal--overlay {
  z-index: 9999 !important;
}

.va-modal {
  z-index: 10000 !important;
}
</style>
