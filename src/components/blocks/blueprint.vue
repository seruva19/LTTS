<script setup>
import { ref, onMounted, onUnmounted, computed, nextTick, provide } from 'vue'
import { useLttsState } from '@/state'
import { useLayerState } from '@/composables/useLayerState'
import { useLayerFormatting } from '@/composables/useLayerFormatting'
import { useLayerFiltering } from '@/composables/useLayerFiltering'
import { useLayerNavigation } from '@/composables/useLayerNavigation'
import { useContextMenu } from '@/composables/useContextMenu'
import ModelInput from '@/components/blocks/model-input.vue'
import ModelOutput from '@/components/blocks/model-output.vue'
import ModelLevelModules from '@/components/blocks/model-level-modules.vue'
import ModelInfoHeader from '@/components/blocks/blueprint/ModelInfoHeader.vue'
import ModelLogPanel from '@/components/model-log-panel.vue'
import ActiveFilterBar from '@/components/blocks/blueprint/ActiveFilterBar.vue'
import CategorySection from '@/components/blocks/blueprint/CategorySection.vue'
import ContextMenu from '@/components/blocks/blueprint/ContextMenu.vue'
import LayerCard from '@/components/blocks/blueprint/LayerCard.vue'
import LazyMount from '@/components/blocks/blueprint/LazyMount.vue'
import ResearchStudyContext from '@/components/blocks/blueprint/ResearchStudyContext.vue'

const state = useLttsState()
const workspace = ref(null)

const layerState = useLayerState()
const { formatNumber } = useLayerFormatting()
const currentProcessingLayer = computed(() => state.ws.currentProcessingLayer)
const isInferenceRunning = computed(() => state.ws.isInferenceRunning)

const areSectionLabelsVisible = ref(true)
const sectionVisibility = ref({
  embeddings: true,
  attention_mechanisms: true,
  transformer_blocks: true,
  feed_forward: true,
  normalization: true,
  output_processing: true
})

const displayOptions = ref({
  showParameters: true,
  showModules: true,
  compactMode: false
})
const selectedModuleFilter = ref('all')
const searchQuery = ref('')

const layerMatchesSearch = (layer) => {
  const query = searchQuery.value.trim().toLowerCase()
  if (!query) {
    return true
  }
  return (layer?.name || '').toLowerCase().includes(query) ||
    (layer?.type || '').toLowerCase().includes(query)
}

const categorizedLayers = computed(() => {
  if (!state.blueprints.loadedBlueprint?.categorization?.categories) return null
  return state.blueprints.loadedBlueprint.categorization.categories
})

const {
  activeFilter,
  displayCategories,
  filterByBreadcrumb,
  clearFilter
} = useLayerFiltering(categorizedLayers)

const layers = computed(() => state.blueprints.loadedBlueprint?.layers || [])
const architectureContext = computed(() => ({
  topology: state.blueprints.loadedBlueprint?.architectureTopology || 'unknown',
  capabilities: state.blueprints.loadedBlueprint?.capabilities || {}
}))
const layerMatchesModuleFilter = (layer) => {
  if (!layer || selectedModuleFilter.value === 'all') {
    return true
  }
  const names = [...(layer.preModules || []), ...(layer.postModules || [])]
    .map((module) => {
      if (typeof module === 'string') {
        return module
      }
      return module?.name || module?.module?.name || module?.metadata?.name || null
    })
    .filter(Boolean)
  return names.includes(selectedModuleFilter.value)
}

const layerPassesFilters = (layer) => layerMatchesModuleFilter(layer) && layerMatchesSearch(layer)

const filteredDisplayCategories = computed(() => {
  const categories = displayCategories.value
  if (!categories || (selectedModuleFilter.value === 'all' && !searchQuery.value.trim())) {
    return categories
  }

  const filtered = {}
  Object.entries(categories).forEach(([categoryKey, categoryData]) => {
    const matchingLayers = (categoryData.layers || []).filter(layerPassesFilters)
    if (matchingLayers.length > 0) {
      filtered[categoryKey] = {
        ...categoryData,
        layers: matchingLayers,
        totalCount: matchingLayers.length,
        totalParameters: matchingLayers.reduce((sum, layer) => sum + (layer.parameters || 0), 0)
      }
    }
  })
  return filtered
})

const filteredLayers = computed(() => (layers.value || []).filter(layerPassesFilters))

const activeFilterLabel = computed(() => {
  if (selectedModuleFilter.value !== 'all') {
    return `module:${selectedModuleFilter.value}`
  }
  return activeFilter.value
})
useLayerNavigation(layers, layerState, computed(() => state.ui.scrollableRef))

const {
  contextMenu,
  showContextMenu,
  addBookmark,
  removeBookmark
} = useContextMenu()

const moduleOptions = computed(() => {
  if (!state.modules.modules) return []
  return Object.keys(state.modules.modules).map(moduleName => ({
    text: moduleName,
    value: moduleName
  }))
})

const preModuleOptions = computed(() => {
  if (!state.modules.modules) return []
  return Object.entries(state.modules.modules)
    .filter(([, module]) => {
      const eventTypes = module.metadata?.event_types || []
      return eventTypes.length === 0 || eventTypes.includes('layer_before')
    })
    .map(([moduleName, module]) => ({
      text: moduleName,
      value: moduleName,
      targetRoles: module.attachment_contract?.target_roles || [],
      supportedTopologies: module.attachment_contract?.supported_topologies || [],
      requiredCapabilities: module.attachment_contract?.required_capabilities || [],
      optionalDependencies: module.attachment_contract?.optional_dependencies || []
    }))
})

const postModuleOptions = computed(() => {
  if (!state.modules.modules) return []
  return Object.entries(state.modules.modules)
    .filter(([, module]) => {
      const eventTypes = module.metadata?.event_types || []
      return eventTypes.length === 0 || eventTypes.includes('layer_after')
    })
    .map(([moduleName, module]) => ({
      text: moduleName,
      value: moduleName,
      targetRoles: module.attachment_contract?.target_roles || [],
      supportedTopologies: module.attachment_contract?.supported_topologies || [],
      requiredCapabilities: module.attachment_contract?.required_capabilities || [],
      optionalDependencies: module.attachment_contract?.optional_dependencies || []
    }))
})

const getTotalAttachedModules = () => {
  if (!state.blueprints.loadedBlueprint?.layers) return 0
  return (state.blueprints.loadedBlueprint.layers || []).reduce((total, layer) => {
    const preModules = layer.preModules?.length || 0
    const postModules = layer.postModules?.length || 0
    return total + preModules + postModules
  }, 0)
}

const getLayerIndex = (layer) => {
  if (!state.blueprints.loadedBlueprint?.layers) return -1
  return state.blueprints.loadedBlueprint.layers.findIndex(l => l.name === layer.name)
}

const getSectionVisibility = (categoryId) => {
  const categoryMapping = {
    'embeddings': 'embeddings',
    'attention_mechanisms': 'attention_mechanisms',
    'attention': 'attention_mechanisms',
    'transformer_blocks': 'transformer_blocks',
    'transformer': 'transformer_blocks',
    'feed_forward': 'feed_forward',
    'ffn': 'feed_forward',
    'normalization': 'normalization',
    'norm': 'normalization',
    'output_processing': 'output_processing',
    'output': 'output_processing'
  }

  const mappedKey = categoryMapping[categoryId.toLowerCase()] || categoryId.toLowerCase()
  return sectionVisibility.value[mappedKey] !== false
}

const handleLayerHeaderClick = (event, layerName) => {
  const target = event.target
  const isControlElement = target.closest('.layer-indicators') ||
                          target.closest('.layer-header-right') ||
                          target.closest('.breadcrumb-segment') ||
                          target.closest('.va-button') ||
                          target.closest('button') ||
                          target.closest('input') ||
                          target.closest('select') ||
                          target.closest('textarea') ||
                          target.classList.contains('breadcrumb-segment') ||
                          target.classList.contains('va-button') ||
                          target.tagName === 'BUTTON' ||
                          target.tagName === 'INPUT' ||
                          target.tagName === 'SELECT' ||
                          target.tagName === 'TEXTAREA'

  if (!isControlElement) {
    layerState.toggleExpanded(layerName)
  }

  const wasSelected = layerState.isSelected(layerName)
  layerState.selectLayer(layerName)

  if (wasSelected) {
    state.ui.setSelectedLayer(null)
  } else {
    const layerObj = (state.blueprints.loadedBlueprint?.layers || []).find(l => l.name === layerName)
    state.ui.setSelectedLayer(layerObj || null)
  }
}

const handleControlsToggle = (event) => {
}

const handleSectionLabelsToggle = (event) => {
  areSectionLabelsVisible.value = event.detail.isVisible
}

const handleSectionVisibilityUpdate = (event) => {
  sectionVisibility.value = { ...event.detail.visibility }
}

const handleDisplayOptionsUpdate = (event) => {
  displayOptions.value = { ...event.detail.options }
}

const handleBookmarkUpdate = (event) => {
  layerState.updateBookmarks(event.detail.bookmarkedLayers)
}

const handleModuleFilterUpdate = (event) => {
  selectedModuleFilter.value = event.detail?.moduleId || 'all'
}

const clearAllFilters = () => {
  selectedModuleFilter.value = 'all'
  clearFilter()
}

const handleExpandAllLayers = () => {
  layerState.expandAllLayers(state.blueprints.loadedBlueprint?.layers)
}

const handleCollapseAllLayers = () => {
  layerState.collapseAllLayers()
}

provide('layerState', layerState)
provide('architectureContext', architectureContext)
provide('preModuleOptions', preModuleOptions)
provide('postModuleOptions', postModuleOptions)
provide('handleLayerHeaderClick', handleLayerHeaderClick)
provide('showContextMenu', showContextMenu)
provide('getLayerIndex', getLayerIndex)
provide('filterByBreadcrumb', filterByBreadcrumb)

onMounted(() => {
  nextTick(() => {
    state.ui.setScrollableRef(workspace.value)
  })

  state.modules.fetchModules()

  document.addEventListener('expand-all-layers', handleExpandAllLayers)
  document.addEventListener('collapse-all-layers', handleCollapseAllLayers)
  document.addEventListener('controls-panel-toggle', handleControlsToggle)
  document.addEventListener('section-labels-toggle', handleSectionLabelsToggle)
  document.addEventListener('section-visibility-update', handleSectionVisibilityUpdate)
  document.addEventListener('display-options-update', handleDisplayOptionsUpdate)
  document.addEventListener('bookmark-update', handleBookmarkUpdate)
  document.addEventListener('layer-module-filter', handleModuleFilterUpdate)
})

onUnmounted(() => {
  document.removeEventListener('expand-all-layers', handleExpandAllLayers)
  document.removeEventListener('collapse-all-layers', handleCollapseAllLayers)
  document.removeEventListener('controls-panel-toggle', handleControlsToggle)
  document.removeEventListener('section-labels-toggle', handleSectionLabelsToggle)
  document.removeEventListener('section-visibility-update', handleSectionVisibilityUpdate)
  document.removeEventListener('display-options-update', handleDisplayOptionsUpdate)
  document.removeEventListener('bookmark-update', handleBookmarkUpdate)
  document.removeEventListener('layer-module-filter', handleModuleFilterUpdate)
})
</script>

<template>
  <div v-if="state.blueprints.loadedBlueprint && (state.blueprints.loadedBlueprint.layers || []).length > 0" class="layers-container" ref="workspace">
    <ModelLogPanel />
    <ResearchStudyContext />
    <ModelInfoHeader
      :blueprint="state.blueprints.loadedBlueprint"
      :total-modules="getTotalAttachedModules()"
    />

    <div class="structure-controls">
      <VaInput
        v-model="searchQuery"
        class="structure-search"
        placeholder="Search layers…"
        clearable
      >
        <template #prependInner>
          <VaIcon name="search" size="small" />
        </template>
      </VaInput>
      <VaButton
        preset="secondary"
        icon="unfold_more"
        size="small"
        title="Expand all layers"
        @click="handleExpandAllLayers"
      />
      <VaButton
        preset="secondary"
        icon="unfold_less"
        size="small"
        title="Collapse all layers"
        @click="handleCollapseAllLayers"
      />
    </div>

    <ModelLevelModules />

    <ModelInput />

    <ActiveFilterBar
      :active-filter="activeFilterLabel"
      @clear="clearAllFilters"
    />

    <div v-if="filteredDisplayCategories && Object.keys(filteredDisplayCategories).length > 0" class="categories-container">
      <CategorySection
        v-for="(category, categoryId) in filteredDisplayCategories"
        :key="categoryId"
        v-show="getSectionVisibility(categoryId)"
        :category="category"
        :category-id="categoryId"
        :are-section-labels-visible="areSectionLabelsVisible"
        :is-inference-running="isInferenceRunning"
        :current-processing-layer="currentProcessingLayer"
      />
    </div>

    <div v-else-if="state.blueprints.loadedBlueprint && filteredLayers.length > 0" class="layers-grid">
      <LazyMount v-for="layer in filteredLayers" :key="layer.name">
        <LayerCard
          :layer="layer"
          :is-inference-running="isInferenceRunning"
          :current-processing-layer="currentProcessingLayer"
        />
      </LazyMount>
    </div>

    <ModelOutput />
  </div>

  <div v-else class="empty-state">
    <div class="empty-content">
      <VaIcon name="layers" size="4rem" color="secondary" />
      <h3>No Model Loaded</h3>
      <p>Load a transformer model to visualize its architecture and attach analysis modules.</p>
    </div>
  </div>

  <ContextMenu
    :context-menu="contextMenu"
    @add-bookmark="addBookmark"
    @remove-bookmark="removeBookmark"
    @select-layer="layerState.selectLayer"
    @toggle-expanded="layerState.toggleExpanded"
  />
</template>

<style scoped>
.layers-container {
  height: calc(100vh - 70px);
  overflow-y: auto;
  padding: 1rem;
  background: var(--va-background-primary);
  position: relative;
  z-index: 1;
}

.structure-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.5rem 0 1rem 0;
}

.structure-search {
  flex: 1;
}

.categories-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.layers-grid {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 400px;
}

.empty-content {
  text-align: center;
  max-width: 300px;
}

.empty-content h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 1rem 0 0.5rem 0;
  color: var(--va-text-primary);
}

.empty-content p {
  font-size: 0.875rem;
  color: var(--va-text-secondary);
  line-height: 1.5;
  margin: 0;
}

@media (max-width: 768px) {
  .layers-container {
    padding: 1rem;
  }

  .categories-container {
    gap: 1.5rem;
  }
}
</style>
