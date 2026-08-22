import { onMounted, onUnmounted, nextTick } from 'vue'
import { useBlueprintStore } from '@/state/blueprint-store'

// Estimated collapsed card height (matches LazyMount minHeight=72) plus the
// flex gap between cards (~1rem). Used to compute scroll offsets for windowed
// (unmounted) cards so we can scroll to them before they exist in the DOM.
const APPROX_CARD_HEIGHT = 72 + 16

function getLayersContainer(scrollableRef) {
  let container = scrollableRef?.value
  if (!container) {
    container = document.querySelector('.layers-container') ||
      document.querySelector('.workspace') ||
      document.querySelector('[data-scrollable]') ||
      document.querySelector('.main-content') ||
      document.documentElement
  }
  return container
}

function pulseElement(el) {
  if (!el) return
  el.classList.add('jump-highlight')
  setTimeout(() => el.classList.remove('jump-highlight'), 950)
}

// Standalone, DOM-query-free scroll-to-index that works with windowed
// (LazyMount-unmounted) cards. Scrolls the container to a computed offset to
// bring the target card into the mount window, then on nextTick re-queries the
// now-mounted card and finishes with scrollIntoView + a highlight pulse.
export function scrollToLayerIndex(index, { scrollableRef = null } = {}) {
  if (index == null || index < 0) return

  const blueprintStore = useBlueprintStore()
  const container = getLayersContainer(scrollableRef)
  if (!container) return

  const offset = Math.max(0, index * APPROX_CARD_HEIGHT - 20)
  container.scrollTo({
    top: offset,
    behavior: 'smooth'
  })

  const orderedLayers = blueprintStore.loadedBlueprint?.layers || []
  const layer = orderedLayers[index]

  nextTick(() => {
    if (!layer) return
    const selector = layer.name
      ? `[data-layer-name="${layer.name}"]`
      : (layer.id != null ? `[data-layer-id="${layer.id}"]` : null)
    if (!selector) return
    const el = document.querySelector(selector)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      pulseElement(el)
    }
  })
}

// Resolve a layer's index in the ordered layer array (by name) and delegate to
// scrollToLayerIndex.
export function scrollToLayerByName(layerName, options = {}) {
  if (!layerName) return
  const blueprintStore = useBlueprintStore()
  const orderedLayers = blueprintStore.loadedBlueprint?.layers || []
  const index = orderedLayers.findIndex(layer => layer.name === layerName)
  if (index >= 0) {
    scrollToLayerIndex(index, options)
  }
}

export function useLayerNavigation(layers, layerState, scrollableRef) {
  const scrollToLayer = (layerName) => scrollToLayerByName(layerName, { scrollableRef })

  const handleKeydown = (event) => {
    if (!layers.value) return

    // Ignore navigation shortcuts while a text field has focus.
    const active = document.activeElement
    const tag = (active?.tagName || '').toLowerCase()
    if (
      tag === 'input' ||
      tag === 'textarea' ||
      tag === 'select' ||
      active?.isContentEditable
    ) {
      return
    }

    const layerList = layers.value
    let currentIndex = layerList.findIndex(layer => layerState.isSelected(layer.name))

    switch (event.key) {
      case 'ArrowUp':
      case 'k':
        event.preventDefault()
        if (currentIndex > 0) {
          layerState.selectLayer(layerList[currentIndex - 1].name)
          scrollToLayer(layerList[currentIndex - 1].name)
        } else if (currentIndex === -1 && layerList.length > 0) {
          layerState.selectLayer(layerList[layerList.length - 1].name)
          scrollToLayer(layerList[layerList.length - 1].name)
        }
        break
      case 'ArrowDown':
      case 'j':
        event.preventDefault()
        if (currentIndex < layerList.length - 1 && currentIndex !== -1) {
          layerState.selectLayer(layerList[currentIndex + 1].name)
          scrollToLayer(layerList[currentIndex + 1].name)
        } else if (currentIndex === -1 && layerList.length > 0) {
          layerState.selectLayer(layerList[0].name)
          scrollToLayer(layerList[0].name)
        }
        break
      case 'Enter':
      case ' ':
        event.preventDefault()
        if (currentIndex !== -1) {
          layerState.toggleExpanded(layerList[currentIndex].name)
        }
        break
      case 'Escape':
        if (layerState.selectedLayerId.value) {
          layerState.selectedLayerId.value = null
        }
        break
    }
  }

  onMounted(() => {
    document.addEventListener('keydown', handleKeydown)
  })

  onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
  })

  return {
    scrollToLayer,
    scrollToLayerIndex: (index) => scrollToLayerIndex(index, { scrollableRef }),
    handleKeydown
  }
}
