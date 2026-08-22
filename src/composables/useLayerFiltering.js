import { ref, computed } from 'vue'

export function useLayerFiltering(categorizedLayers) {
  const activeFilter = ref(null)
  const filteredCategories = ref(null)

  const filterByBreadcrumb = (path) => {
    if (activeFilter.value === path) {
      activeFilter.value = null
      filteredCategories.value = null
      return
    }

    activeFilter.value = path

    if (!categorizedLayers.value) {
      filteredCategories.value = null
      return
    }

    const filtered = {}

    Object.entries(categorizedLayers.value).forEach(([categoryKey, categoryData]) => {
      const matchingLayers = categoryData.layers.filter(layer => {
        return layer.fullPath && layer.fullPath.startsWith(path)
      })

      if (matchingLayers.length > 0) {
        filtered[categoryKey] = {
          ...categoryData,
          layers: matchingLayers,
          totalCount: matchingLayers.length,
          totalParameters: matchingLayers.reduce((sum, layer) => sum + (layer.parameters || 0), 0)
        }
      }
    })

    filteredCategories.value = filtered
  }

  const clearFilter = () => {
    activeFilter.value = null
    filteredCategories.value = null
  }

  const displayCategories = computed(() => {
    return filteredCategories.value || categorizedLayers.value
  })

  return {
    activeFilter,
    filteredCategories,
    displayCategories,
    filterByBreadcrumb,
    clearFilter
  }
}
