import { ref, computed } from 'vue'

export function useLayerState() {
  const selectedLayerId = ref(null)
  const expandedLayers = ref(new Set())
  const bookmarkedLayers = ref(new Set())
  const showModuleControls = ref({})
  const preModuleSelection = ref({})
  const postModuleSelection = ref({})

  const isSelected = (layerName) => selectedLayerId.value === layerName
  const isExpanded = (layerName) => expandedLayers.value.has(layerName)
  const isBookmarked = (layerName) => bookmarkedLayers.value.has(layerName)

  const selectLayer = (layerName) => {
    selectedLayerId.value = selectedLayerId.value === layerName ? null : layerName
  }

  const toggleExpanded = (layerName) => {
    if (expandedLayers.value.has(layerName)) {
      expandedLayers.value.delete(layerName)
    } else {
      expandedLayers.value.add(layerName)
    }
  }

  const expandAllLayers = (layers) => {
    if (layers) {
      layers.forEach(layer => {
        expandedLayers.value.add(layer.name)
      })
    }
  }

  const collapseAllLayers = () => {
    expandedLayers.value.clear()
  }

  const toggleModuleControls = (layerName) => {
    showModuleControls.value[layerName] = !showModuleControls.value[layerName]
  }

  const updateBookmarks = (bookmarkedLayerNames) => {
    bookmarkedLayers.value = new Set(bookmarkedLayerNames)
  }

  return {
    selectedLayerId,
    expandedLayers,
    bookmarkedLayers,
    showModuleControls,
    preModuleSelection,
    postModuleSelection,
    isSelected,
    isExpanded,
    isBookmarked,
    selectLayer,
    toggleExpanded,
    expandAllLayers,
    collapseAllLayers,
    toggleModuleControls,
    updateBookmarks
  }
}
