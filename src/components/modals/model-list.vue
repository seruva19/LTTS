<script setup>
import { useLttsState } from '@/state';
import { useModelStore } from '@/state/model-store'
import { useToast } from '@/composables/useToast'
import { computed, watch, ref } from 'vue'

const state = useLttsState()
const modelStore = useModelStore()
const toast = useToast()

const isLoadingModel = ref(false)

watch(() => state.models.showModelList, (newValue) => {
  if (newValue) {
    state.models.fetchModels()
  }
})

const items = computed(() => {
  return state.models.modelList.map(model => ({
    ...model,
    status: model.loaded ? 'Loaded' : 'Available',
    engine: model.engine || 'transformers',
    description: model.description || 'No description available'
  }))
})

const tableColumns = [
  { key: 'name', label: 'Model Name', sortable: true },
  { key: 'engine', label: 'Engine', sortable: true },
  { key: 'status', label: 'Status', sortable: true },
  { key: 'description', label: 'Description' },
  { key: 'actions', label: 'Actions', width: '200px' }
]

const selectedRows = ref([])

const selectModel = (model) => {
    selectedRows.value = [model]
  state.models.selectModel(model)
}

// Inline row styling to guarantee visual highlight
const rowBind = (item) => {
  const selectedId = selectedRows.value?.[0]?.id || state.models.selectedModel?.id
  const isSelected = selectedId === item.id
  if (isSelected) {
    return {
      style: 'background-color: #eff6ff; border-left: 4px solid #3b82f6;'
    }
  }
  return {}
}

const onSelectionChange = (selection) => {
    if (selection.length > 0) {
    state.models.selectModel(selection[0])
  }
}

const onRowClick = (event) => {
    // Programmatically trigger selection like checkbox does
  selectedRows.value = [event.item]
  state.models.selectModel(event.item)
}

const onRowDoubleClick = (event) => {
  const model = event.item
  // Programmatically trigger selection like checkbox does
  selectedRows.value = [model]
  state.models.selectModel(model)
  if (!model.loaded) {
    loadModel(model.id, model.engine)
  }
}

const loadModel = async (modelId, engine = 'transformers') => {
  isLoadingModel.value = true
  try {
        await modelStore.loadModel(modelId, engine)

    // Also load blueprint from model
    const loadedModel = modelStore.loadedModels[modelId]
    if (loadedModel) {
      state.blueprints.loadBlueprintFromModel(loadedModel)
    }

    // Close modal after successful load
    state.models.hideModels()
  } catch (error) {
    toast.error(`Failed to load model: ${error.message}`)
  } finally {
    isLoadingModel.value = false
  }
}

const unloadModel = (modelId) => {
  modelStore.unloadModel(modelId)
}

const newModelId = ref('')
const isAddingModel = ref(false)

const addModel = async () => {
  const modelId = newModelId.value.trim()
  if (!modelId) return

  isAddingModel.value = true
  try {
    const result = await modelStore.addModel(modelId)
    newModelId.value = ''
    toast.success(`Added ${result.model_id} (${result.task || 'task auto-detected'})`)
  } catch (error) {
    toast.error(error.response?.data?.detail || `Failed to add model: ${error.message}`)
  } finally {
    isAddingModel.value = false
  }
}

const removeModel = async (modelId) => {
  try {
    await modelStore.removeModel(modelId)
    toast.success(`Removed ${modelId} from the registry`)
  } catch (error) {
    toast.error(error.response?.data?.detail || `Failed to remove model: ${error.message}`)
  }
}

const loadSelectedModel = () => {
  if (state.models.selectedModel) {
    loadModel(state.models.selectedModel.id, state.models.selectedModel.engine)
    state.models.hideModels()
  }
}

</script>

<template>
  <VaModal
    class="modal-window"
    v-model="state.models.showModelList"
    title="Available Models"
    size="large"
    max-height="70vh"
    hide-default-actions
  >
    <div v-if="state.models.isLoading" class="loading-indicator">
      <VaProgressBar indeterminate size="small" />
      Loading models...
    </div>

    <div class="add-model-row">
      <input
        v-model="newModelId"
        class="add-model-input"
        placeholder="Hugging Face model id, e.g. HuggingFaceTB/SmolLM-135M"
        :disabled="isAddingModel"
        @keyup.enter="addModel"
      />
      <button
        class="action-btn load-btn"
        :disabled="isAddingModel || !newModelId.trim()"
        @click="addModel"
      >
        {{ isAddingModel ? 'Checking...' : 'Add model' }}
      </button>
    </div>

    <div v-if="!state.models.isLoading && items.length === 0" class="no-models">
      <VaAlert color="info" border="left">
        No models registered yet. Paste a Hugging Face model id above to add one.
      </VaAlert>
    </div>

    <VaDataTable
      v-else
      :items="items"
      :columns="tableColumns"
      hoverable
      selectable
      select-mode="single"
      v-model:selected="selectedRows"
      @selection-change="onSelectionChange"
      @row:click="onRowClick"
      @row:dblclick="onRowDoubleClick"
      :row-bind="rowBind"
      class="minimal-table hide-checkboxes"
    >
      <template #cell(status)="{ value }">
        <span :class="['status-badge', value === 'Loaded' ? 'status-loaded' : 'status-available']">
          {{ value }}
        </span>
      </template>

      <template #cell(actions)="{ rowData }">
        <div class="action-buttons">
          <button
            v-if="!rowData.loaded"
            class="action-btn load-btn"
            :disabled="isLoadingModel"
            @click="loadModel(rowData.id, rowData.engine)"
          >
            {{ isLoadingModel ? 'Loading...' : 'Load' }}
          </button>
          <button
            v-else
            class="action-btn unload-btn"
            @click="unloadModel(rowData.id)"
          >
            Unload
          </button>
          <button
            class="action-btn remove-btn"
            title="Remove from the registry (downloaded files are kept)"
            @click="removeModel(rowData.id)"
          >
            Remove
          </button>
        </div>
      </template>
    </VaDataTable>

    <div class="selected-model-info" :class="{ 'no-selection': !state.models.selectedModel }">
      <div class="selected-model">
        <span class="selected-text">
          {{ state.models.selectedModel ? `Selected: ${state.models.selectedModel.name}` : 'No model selected' }}
        </span>
        <button
          class="action-btn load-selected-btn"
          :disabled="!state.models.selectedModel || state.models.selectedModel.loaded || isLoadingModel"
          @click="loadSelectedModel"
        >
          {{ !state.models.selectedModel ? 'Select Model' : (state.models.selectedModel.loaded ? 'Already Loaded' : 'Load Selected') }}
        </button>
      </div>
    </div>

    <template #footer>
      <button class="close-btn" @click="state.models.hideModels()">
        Close
      </button>
    </template>
  </VaModal>
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

.no-models {
  padding: 2rem;
  text-align: center;
  color: #6b7280;
}

.minimal-table {
  --va-data-table-thead-background: transparent;
  --va-data-table-tbody-tr-border: 1px solid #f3f4f6;
  border: none;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.status-loaded {
  background-color: #ecfdf5;
  color: #065f46;
}

.status-available {
  background-color: #f3f4f6;
  color: #6b7280;
}

.add-model-row {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 1rem;
}

.add-model-input {
  flex: 1;
  padding: 0.375rem 0.75rem;
  border: 1px solid var(--va-background-border, #d1d5db);
  border-radius: 4px;
  background: transparent;
  color: inherit;
  font-size: 0.875rem;
}

.remove-btn {
  border-color: #ef4444;
  color: #ef4444;
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
  color: #374151;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}

.action-btn:hover:not(:disabled) {
  background-color: #f9fafb;
  border-color: #9ca3af;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.load-btn:hover:not(:disabled) {
  background-color: #f0f9ff;
  border-color: #3b82f6;
  color: #1d4ed8;
}

.unload-btn:hover:not(:disabled) {
  background-color: #fef2f2;
  border-color: #ef4444;
  color: #dc2626;
}

.selected-model-info {
  margin-top: 1.5rem;
  padding: 1rem;
  background-color: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  transition: all 0.15s ease;
}

.selected-model-info.no-selection {
  background-color: #f9fafb;
  border-color: #d1d5db;
  opacity: 0.7;
}

.selected-model-info.no-selection .selected-text {
  color: #9ca3af;
}

.selected-model-info.no-selection .load-selected-btn {
  background-color: #f3f4f6;
  color: #9ca3af;
  border-color: #d1d5db;
}

.selected-model {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.selected-text {
  color: #475569;
  font-weight: 500;
}

.load-selected-btn {
  background-color: #3b82f6;
  color: white;
  border-color: #3b82f6;
}

.load-selected-btn:hover:not(:disabled) {
  background-color: #2563eb;
  border-color: #2563eb;
}

.load-selected-btn:disabled {
  background-color: #9ca3af;
  border-color: #9ca3af;
}

.close-btn {
  padding: 0.5rem 1rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: white;
  color: #6b7280;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}

.close-btn:hover {
  background-color: #f9fafb;
  color: #374151;
}

.modal-window {
  --va-modal-padding: 1.5rem;
}
::v-deep(.hide-checkboxes .va-data-table__table-cell-select) {
  display: none !important;
}


/* Reserve space for selection border to avoid flicker */
.minimal-table .va-data-table__table-tr {
  border-left: 4px solid transparent;
}

@media (max-width: 768px) {
  .action-buttons {
    flex-direction: column;
    gap: 0.25rem;
  }

  .selected-model {
    flex-direction: column;
    align-items: stretch;
    gap: 0.75rem;
  }
}
</style>
