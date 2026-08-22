<template>
  <VaModal
    v-model="isOpen"
    title="Blueprint Manager"
    size="medium"
    class="blueprint-modal"
    hide-default-actions
    @before-close="closeModal"
    :z-index="9999"
  >
    <div class="blueprint-manager">
      <div class="manager-header">
        <div class="current-config">
          <div class="config-badge">
            <VaIcon name="psychology" size="small" />
            <span>{{ currentModel || 'No Model' }}</span>
          </div>
          <div class="config-badge">
            <VaIcon name="layers" size="small" />
            <span>{{ layerCount }} layers</span>
          </div>
          <div class="config-badge">
            <VaIcon name="extension" size="small" />
            <span>{{ activeModuleCount }} modules</span>
          </div>
        </div>
      </div>

      <div v-if="showSaveForm" class="save-section">
        <div class="form-inline">
          <VaInput
            v-model="saveForm.name"
            placeholder="Blueprint name..."
            class="name-input"
          />
          <VaInput
            v-model="saveForm.description"
            placeholder="Description (optional)..."
            class="desc-input"
          />
          <VaButton
            preset="primary"
            :disabled="!canSave"
            :loading="isExecuting"
            @click="saveBlueprintAction"
          >
            <VaIcon name="save" size="small" />
            Save
          </VaButton>
          <VaButton
            preset="plain"
            @click="toggleSaveForm"
          >
            Cancel
          </VaButton>
        </div>
      </div>

      <div class="library-section">
        <div class="library-controls">
          <VaInput
            v-model="searchQuery"
            placeholder="Search blueprints..."
            class="search-input"
          >
            <template #prependInner>
              <VaIcon name="search" />
            </template>
          </VaInput>

          <VaSelect
            v-model="sortBy"
            :options="sortOptions"
            placeholder="Sort by..."
            class="sort-select"
          />

          <VaButton
            preset="primary"
            @click="toggleSaveForm"
            v-if="!showSaveForm"
          >
            <VaIcon name="add" size="small" />
            New Blueprint
          </VaButton>

          <VaButton
            preset="plain"
            icon="refresh"
            @click="refreshBlueprints"
            title="Refresh"
          />
        </div>

        <div v-if="isLoading" class="loading-state">
          <VaProgressCircle indeterminate size="small" />
          <span>Loading...</span>
        </div>

        <div v-else-if="error" class="error-state">
          <VaAlert color="danger" border="left">
            {{ error }}
          </VaAlert>
        </div>

        <div v-else-if="filteredBlueprints.length === 0 && searchQuery" class="empty-state">
          <VaIcon name="search_off" size="3rem" color="secondary" />
          <h4>No Results</h4>
          <p>Try a different search</p>
        </div>

        <div v-else-if="filteredBlueprints.length === 0" class="empty-state">
          <VaIcon name="folder_open" size="3rem" color="secondary" />
          <h4>No Blueprints</h4>
          <p>Create your first blueprint</p>
        </div>

        <div v-else class="blueprints-grid">
          <BlueprintCard
            v-for="blueprint in filteredBlueprints"
            :key="blueprint.id"
            :blueprint="blueprint"
            :isSelected="selectedBlueprint?.id === blueprint.id"
            :isLoading="loadingBlueprintId === blueprint.id"
            @select="selectBlueprint"
            @load="loadBlueprintAction"
            @delete="confirmDeleteBlueprint"
            @duplicate="duplicateBlueprint"
          />
        </div>

        <div v-if="filteredBlueprints.length > 0 && selectedBlueprint" class="selected-actions">
          <VaButton
            preset="primary"
            :loading="isExecuting"
            @click="loadBlueprintAction(selectedBlueprint)"
          >
            <VaIcon name="download" size="small" />
            Load "{{ selectedBlueprint.name }}"
          </VaButton>
        </div>
      </div>

      <div v-if="statusMessage" class="status-alert">
        <VaAlert
          :color="statusType === 'success' ? 'success' : statusType === 'error' ? 'danger' : 'info'"
          border="left"
          closeable
          @close="statusMessage = ''"
        >
          {{ statusMessage }}
        </VaAlert>
      </div>
    </div>
  </VaModal>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useModuleStore } from '../../state/module-store'
import { useModelStore } from '../../state/model-store'
import { useBlueprintStore } from '../../state/blueprint-store'
import { useLttsState } from '../../state'
import { useBlueprintThumbnail } from '../../composables/useBlueprintThumbnail'
import BlueprintCard from '../blueprints/BlueprintCard.vue'

// Stores
const moduleStore = useModuleStore()
const modelStore = useModelStore()
const blueprintStore = useBlueprintStore()
const state = useLttsState()
const { captureBlueprintWorkspace } = useBlueprintThumbnail()

// Modal state
const isOpen = computed({
  get: () => state.ui.showBlueprintManager,
  set: (value) => {
    if (!value) state.ui.closeBlueprintManager()
  }
})

const showSaveForm = ref(false)
const statusMessage = ref('')
const statusType = ref('info')
const isExecuting = ref(false)
const loadingBlueprintId = ref(null)
const searchQuery = ref('')
const sortBy = ref('date-desc')

const sortOptions = [
  { value: 'date-desc', text: 'Newest First' },
  { value: 'date-asc', text: 'Oldest First' },
  { value: 'name-asc', text: 'Name (A-Z)' },
  { value: 'name-desc', text: 'Name (Z-A)' },
  { value: 'modules-desc', text: 'Most Modules' },
  { value: 'layers-desc', text: 'Most Layers' }
]

// Save form
const saveForm = ref({
  name: '',
  description: ''
})

// Computed properties
const blueprints = computed(() => blueprintStore.blueprints)
const isLoading = computed(() => blueprintStore.isLoading)
const error = computed(() => blueprintStore.error)
const selectedBlueprint = computed(() => blueprintStore.getSelectedBlueprint())

const currentModel = computed(() => {
  return modelStore.currentModel?.name || modelStore.currentModel?.id || null
})

const layerCount = computed(() => {
  return blueprintStore.loadedBlueprint?.layers?.length || 0
})

const activeModuleCount = computed(() => {
  const globalModules = moduleStore.attachedModules.length
  let blueprintModules = 0
  if (blueprintStore.loadedBlueprint?.layers) {
    blueprintModules = blueprintStore.loadedBlueprint.layers.reduce((total, layer) => {
      return total + (layer.preModules?.length || 0) + (layer.postModules?.length || 0)
    }, 0)
  }
  return globalModules + blueprintModules
})

const canSave = computed(() => {
  return saveForm.value.name.trim().length > 0
})

const filteredBlueprints = computed(() => {
  let result = Array.isArray(blueprints.value) ? blueprints.value : []

  // Apply search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(bp =>
      bp.name?.toLowerCase().includes(query) ||
      bp.description?.toLowerCase().includes(query)
    )
  }

  // Apply sorting
  result = [...result].sort((a, b) => {
    switch (sortBy.value) {
      case 'date-desc':
        return new Date(b.created_at || b.createdAt) - new Date(a.created_at || a.createdAt)
      case 'date-asc':
        return new Date(a.created_at || a.createdAt) - new Date(b.created_at || b.createdAt)
      case 'name-asc':
        return (a.name || '').localeCompare(b.name || '')
      case 'name-desc':
        return (b.name || '').localeCompare(a.name || '')
      case 'modules-desc':
        return (b.module_count || b.moduleCount || 0) - (a.module_count || a.moduleCount || 0)
      case 'layers-desc':
        return (b.layers?.length || 0) - (a.layers?.length || 0)
      default:
        return 0
    }
  })

  return result
})

// Methods
function closeModal() {
  isOpen.value = false
  resetForm()
}

function resetForm() {
  saveForm.value = { name: '', description: '' }
  blueprintStore.clearSelection()
  statusMessage.value = ''
  searchQuery.value = ''
  showSaveForm.value = false
}

function toggleSaveForm() {
  showSaveForm.value = !showSaveForm.value
  if (showSaveForm.value) {
    saveForm.value = { name: '', description: '' }
  }
}

function selectBlueprint(blueprint) {
  if (selectedBlueprint.value?.id === blueprint.id) {
    blueprintStore.clearSelection()
  } else {
    blueprintStore.setSelectedBlueprint(blueprint.id)
  }
}

async function saveBlueprintAction() {
  try {
    isExecuting.value = true
    statusMessage.value = ''

    // Capture thumbnail of current blueprint workspace
    let thumbnail = null
    try {
      thumbnail = await captureBlueprintWorkspace()
      console.debug('[Blueprint] Thumbnail captured')
    } catch (err) {
      console.warn('[Blueprint] Failed to capture thumbnail:', err)
      // Continue without thumbnail
    }

    // Capture current client state
    const clientState = {
      model_state: {
        selected_model: modelStore.selectedModel,
        loaded_models: Object.keys(modelStore.loadedModels || {})
      },
      ui_state: {
        current_view: 'workspace',
        timestamp: new Date().toISOString()
      },
      modules: moduleStore.attachedModules.map(module => ({
        name: module.name,
        modelId: module.modelId,
        eventType: module.eventType,
        attachedAt: module.attachedAt,
        parameters: module.parameters
      })),
      blueprint: {
        id: blueprintStore.loadedBlueprint?.id,
        name: blueprintStore.loadedBlueprint?.name || 'Unnamed Blueprint',
        description: blueprintStore.loadedBlueprint?.description || '',
        layers: blueprintStore.loadedBlueprint?.layers || [],
        modules: blueprintStore.loadedBlueprint?.modules || [],
        architecture: blueprintStore.loadedBlueprint?.architecture || 'Transformer',
        engine: blueprintStore.loadedBlueprint?.engine || 'transformers',
        categorization: blueprintStore.loadedBlueprint?.categorization || null,
        thumbnail: thumbnail // Include thumbnail
      },
      workspace_config: {}
    }

    const result = await blueprintStore.saveBlueprint(
      saveForm.value.name.trim(),
      saveForm.value.description.trim(),
      clientState
    )

    statusMessage.value = `Blueprint "${result.name}" saved`
    statusType.value = 'success'

    setTimeout(() => {
      showSaveForm.value = false
      saveForm.value = { name: '', description: '' }
    }, 1000)
  } catch (err) {
    statusMessage.value = err.message || 'Failed to save blueprint'
    statusType.value = 'error'
  } finally {
    isExecuting.value = false
  }
}

async function loadBlueprintAction(blueprint) {
  if (!blueprint) return

  try {
    loadingBlueprintId.value = blueprint.id
    isExecuting.value = true
    statusMessage.value = ''

    await blueprintStore.loadBlueprint(blueprint.id)

    statusMessage.value = `Blueprint "${blueprint.name}" loaded`
    statusType.value = 'success'

    setTimeout(() => {
      closeModal()
    }, 1500)
  } catch (err) {
    statusMessage.value = err.message || 'Failed to load blueprint'
    statusType.value = 'error'
  } finally {
    isExecuting.value = false
    loadingBlueprintId.value = null
  }
}

async function confirmDeleteBlueprint(blueprint) {
  if (!confirm(`Are you sure you want to delete "${blueprint.name}"?`)) return

  try {
    await blueprintStore.deleteBlueprint(blueprint.id)
    statusMessage.value = 'Blueprint deleted'
    statusType.value = 'success'

    // Clear selection if deleted blueprint was selected
    if (selectedBlueprint.value?.id === blueprint.id) {
      blueprintStore.clearSelection()
    }
  } catch (err) {
    statusMessage.value = err.message || 'Failed to delete blueprint'
    statusType.value = 'error'
  }
}

async function duplicateBlueprint(blueprint) {
  try {
    const response = await blueprintStore.loadBlueprint(blueprint.id)

    saveForm.value = {
      name: `${blueprint.name} (Copy)`,
      description: blueprint.description || ''
    }
    showSaveForm.value = true

    statusMessage.value = 'Blueprint loaded for duplication. Edit and save.'
    statusType.value = 'info'
  } catch (err) {
    statusMessage.value = err.message || 'Failed to duplicate blueprint'
    statusType.value = 'error'
  }
}

async function refreshBlueprints() {
  await blueprintStore.fetchBlueprints()
}

// Lifecycle
onMounted(() => {
  refreshBlueprints()
  blueprintStore.loadSelectionFromLocalStorage()
})

watch(isOpen, (newValue) => {
  if (newValue) {
    refreshBlueprints()
    blueprintStore.loadSelectionFromLocalStorage()
  }
})
</script>

<style scoped>
.blueprint-manager {
  min-height: 420px;
  display: flex;
  flex-direction: column;
  gap: 0.875rem;
}

.manager-header {
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--va-background-border);
}

.current-config {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.config-badge {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.75rem;
  background: var(--va-background-secondary);
  border-radius: 6px;
  font-size: 0.8125rem;
  font-weight: 500;
  color: var(--va-text-primary);
}

.config-badge .va-icon {
  color: var(--va-primary);
  opacity: 0.8;
}

.save-section {
  background: var(--va-background-element);
  border: 1px solid var(--va-primary);
  border-radius: 8px;
  padding: 0.875rem;
}

.form-inline {
  display: flex;
  gap: 0.625rem;
  align-items: center;
}

.name-input {
  flex: 2;
  min-width: 200px;
}

.desc-input {
  flex: 3;
  min-width: 250px;
}

.library-section {
  display: flex;
  flex-direction: column;
  gap: 0.875rem;
  flex: 1;
}

.library-controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  margin-bottom: 0.875rem;
}

.search-input {
  flex: 1;
}

.sort-select {
  min-width: 160px;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.625rem;
  padding: 2rem;
  color: var(--va-text-secondary);
  font-size: 0.875rem;
}

.error-state {
  padding: 0.5rem 0;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 2.5rem 1rem;
  gap: 0.625rem;
}

.empty-state h4 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--va-text-primary);
}

.empty-state p {
  margin: 0;
  color: var(--va-text-secondary);
  font-size: 0.875rem;
}

.blueprints-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 1rem;
  padding: 0.25rem 0;
  max-height: 480px;
  overflow-y: auto;
  overflow-x: hidden;
}

.blueprints-grid::-webkit-scrollbar {
  width: 6px;
}

.blueprints-grid::-webkit-scrollbar-track {
  background: var(--va-background-element);
  border-radius: 3px;
}

.blueprints-grid::-webkit-scrollbar-thumb {
  background: var(--va-background-border);
  border-radius: 3px;
}

.blueprints-grid::-webkit-scrollbar-thumb:hover {
  background: var(--va-primary);
}

.selected-actions {
  display: flex;
  justify-content: center;
  padding-top: 0.875rem;
  margin-top: 0.5rem;
  border-top: 1px solid var(--va-background-border);
}

.status-alert {
  margin-top: 0.875rem;
}

@media (max-width: 1024px) {
  .blueprints-grid {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  }
}

@media (max-width: 768px) {
  .current-config {
    flex-direction: column;
    gap: 0.5rem;
  }

  .form-inline {
    flex-direction: column;
    align-items: stretch;
  }

  .name-input,
  .desc-input {
    min-width: auto;
  }

  .library-controls {
    flex-wrap: wrap;
  }

  .sort-select {
    min-width: auto;
    flex: 1;
  }

  .blueprints-grid {
    grid-template-columns: 1fr;
    max-height: 360px;
  }
}

.blueprint-modal :deep(.va-modal__container) {
  max-width: 920px;
}

.blueprint-modal {
  z-index: 10000 !important;
}
</style>
