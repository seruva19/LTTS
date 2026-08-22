<script setup>
import { useLttsState } from '@/state';
import { useModuleStore } from '@/state/module-store'
import { useModelStore } from '@/state/model-store'
import { useToast } from '@/composables/useToast'
import { computed, watch, ref } from 'vue'
import ModuleConfig from './module-config.vue'

const state = useLttsState()
const moduleStore = useModuleStore()
const modelStore = useModelStore()
const toast = useToast()

const showModuleConfig = ref(false)
const selectedModuleId = ref(null)
const selectedModelId = ref(null)

watch(() => state.modules.showModulesList, async (newValue) => {
  if (newValue) {
    try {
      await state.modules.fetchModules()
      if (state.modules.moduleList.length === 0) {
        toast.info('No modules found. Clone LTTS-modules next to LTTS, or set LTTS_MODULES_DIR.')
      }
    } catch (error) {
      toast.error('Failed to load modules. Please check your connection and try again.')
      console.error('Module fetch error:', error)
    }
  }
})

const items = computed(() => {
  return state.modules.moduleList.map(module => ({
    ...module,
    hasUISchema: !!module.ui_schema?.parameters?.length,
    eventTypes: module.metadata?.event_types?.join(', ') || 'N/A',
    version: module.metadata?.version || '1.0.0'
  }))
})

const tableColumns = [
  { key: 'name', label: 'Module Name', sortable: true },
  { key: 'version', label: 'Version', sortable: true },
  { key: 'eventTypes', label: 'Event Types' },
  { key: 'hasUISchema', label: 'Configurable', sortable: true },
  { key: 'actions', label: 'Actions', width: '120px' }
]

const currentModel = computed(() => {
  // Return the loaded model.
  return Object.values(modelStore.models).find(model => model.loaded) || null
})

const configureModule = (moduleId) => {
  selectedModuleId.value = moduleId
  selectedModelId.value = currentModel.value?.id || null
  showModuleConfig.value = true
}

const exploreModule = (moduleId) => {
  // Open the module browser with this module selected.
  const event = new CustomEvent('open-modules-browser', {
    detail: { moduleId }
  })
  document.dispatchEvent(event)
}

const onModuleExecuted = () => {
  // Callback when module is executed
}

</script>

<template>
  <VaModal
    class="modal-window"
    v-model="state.modules.showModulesList"
    title="Available Modules"
    size="large"
    max-height="70vh"
    hide-default-actions
  >
    <div v-if="state.isLoading" class="loading-indicator">
      <VaProgressBar indeterminate size="small" />
      Loading modules...
    </div>

    <div v-else-if="items.length === 0" class="no-modules">
      <VaAlert color="info" border="left">
        No modules found. Analysis modules live in the LTTS-modules repository — clone it next to LTTS, or point LTTS_MODULES_DIR at your checkout.
      </VaAlert>
    </div>

    <VaDataTable
      v-else
      :items="items"
      :columns="tableColumns"
      hoverable
      class="minimal-table"
    >
      <template #cell(hasUISchema)="{ value }">
        <span :class="['config-badge', value ? 'config-available' : 'config-unavailable']">
          {{ value ? 'Yes' : 'No' }}
        </span>
      </template>

      <template #cell(actions)="{ rowData }">
            <div class="action-buttons">
              <VaButton
                size="small"
                preset="secondary"
                @click="exploreModule(rowData.name)"
                title="Explore Module Structure"
              >
                <VaIcon name="visibility" size="small" />
              </VaButton>
              <VaButton
                class="action-btn configure-btn"
                :disabled="!currentModel"
                @click="configureModule(rowData.name)"
                size="small"
              >
                Configure
              </VaButton>
            </div>
          </template>
    </VaDataTable>

    <div v-if="!currentModel" class="model-warning">
      <div class="warning-content">
        <span class="warning-text">No model loaded. Please load a model first to configure modules.</span>
      </div>
    </div>
  </VaModal>

  <!-- Module Configuration Modal -->
  <ModuleConfig
    v-model="showModuleConfig"
    :module-id="selectedModuleId"
    :model-id="selectedModelId"
    @module-executed="onModuleExecuted"
  />
</template>

<style scoped>
.loading-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 2rem;
  justify-content: center;
  color: #6b7280;
  font-size: 0.875rem;
}

.no-modules {
  padding: 2rem;
  text-align: center;
  color: #6b7280;
}

.minimal-table {
  --va-data-table-thead-background: transparent;
  --va-data-table-tbody-tr-border: 1px solid #f3f4f6;
  border: none;
}

.config-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.config-available {
  background-color: #ecfdf5;
  color: #065f46;
}

.config-unavailable {
  background-color: #f3f4f6;
  color: #6b7280;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.action-btn {
  padding: 0.375rem 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: white;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 0.875rem;
}

.action-btn:hover:not(:disabled) {
  background: #f3f4f6;
  border-color: #9ca3af;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.configure-btn {
  background: var(--va-primary);
  color: white;
  border-color: var(--va-primary);
}

.configure-btn:hover:not(:disabled) {
  background: var(--va-primary-hover);
  border-color: var(--va-primary-hover);
}

.model-warning {
  margin-top: 1.5rem;
  padding: 1rem;
  background-color: #fffbeb;
  border-radius: 8px;
  border: 1px solid #fed7aa;
}

.warning-content {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.warning-text {
  color: #92400e;
  font-weight: 500;
  font-size: 0.875rem;
}

.modal-window {
  --va-modal-padding: 1.5rem;
}
</style>
