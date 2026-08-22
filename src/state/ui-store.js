import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUIStore = defineStore('ui', () => {
  const showSettingsPanel = ref(false)
  const showBlueprintManager = ref(false)
  const scrollableRef = ref(null)
  const userPrompt = ref('Hello, how are you doing?')
  const selectedLayer = ref(null)

  // Selected layer tracking
  const selectedLayerName = ref(null)
  const selectedLayerId = ref(null)

  function showSettings() {
    showSettingsPanel.value = true
  }

  function hideSettings() {
    showSettingsPanel.value = false
  }

  function openBlueprintManager() {
    showBlueprintManager.value = true
  }

  function closeBlueprintManager() {
    showBlueprintManager.value = false
  }

  function setScrollableRef(el) {
    scrollableRef.value = el
  }

  function scrollToTop() {
    // Try the stored reference first
    let element = scrollableRef.value

    // Fallback to finding the scrollable container
    if (!element) {
      element = document.querySelector('.layers-container') ||
        document.querySelector('.workspace') ||
        document.querySelector('[data-scrollable]') ||
        document.querySelector('.main-content') ||
        document.documentElement
    }

    if (element) {
      element.scrollTo({
        top: 0,
        behavior: 'smooth'
      })
    } else {
    }
  }

  function scrollToBottom() {
    // Try the stored reference first
    let element = scrollableRef.value

    // Fallback to finding the scrollable container
    if (!element) {
      element = document.querySelector('.layers-container') ||
        document.querySelector('.workspace') ||
        document.querySelector('[data-scrollable]') ||
        document.querySelector('.main-content') ||
        document.documentElement
    }

    if (element) {
      element.scrollTo({
        top: element.scrollHeight - element.clientHeight,
        behavior: 'smooth'
      })
    } else {
    }
  }

  function setSelectedLayer(layer) {
    if (!layer) {
      selectedLayer.value = null
      selectedLayerName.value = null
      selectedLayerId.value = null
      return
    }
    selectedLayer.value = layer
    selectedLayerName.value = layer.name || null
    selectedLayerId.value = layer.id || null
  }

  return {
    showSettingsPanel,
    showBlueprintManager,
    scrollableRef,
    userPrompt,
    selectedLayer,
    selectedLayerName,
    selectedLayerId,
    showSettings,
    hideSettings,
    openBlueprintManager,
    closeBlueprintManager,
    setScrollableRef,
    scrollToTop,
    scrollToBottom,
    setSelectedLayer
  }
})
