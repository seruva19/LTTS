import { ref, onMounted, onUnmounted } from 'vue'

export function useContextMenu() {
  const contextMenu = ref({
    visible: false,
    x: 0,
    y: 0,
    layerName: '',
    layerIndex: -1,
    layerId: null
  })

  const showContextMenu = (event, layerName, layerIndex, layerId = null) => {
    contextMenu.value = {
      visible: true,
      x: event.clientX,
      y: event.clientY,
      layerName,
      layerIndex,
      layerId
    }
  }

  const hideContextMenu = () => {
    contextMenu.value.visible = false
  }

  const addBookmark = () => {
    const event = new CustomEvent('add-bookmark', {
      detail: {
        layerName: contextMenu.value.layerName,
        layerIndex: contextMenu.value.layerIndex,
        layerId: contextMenu.value.layerId
      }
    })
    document.dispatchEvent(event)
    hideContextMenu()
  }

  const removeBookmark = () => {
    const event = new CustomEvent('remove-bookmark', {
      detail: {
        layerName: contextMenu.value.layerName,
        layerIndex: contextMenu.value.layerIndex,
        layerId: contextMenu.value.layerId
      }
    })
    document.dispatchEvent(event)
    hideContextMenu()
  }

  onMounted(() => {
    document.addEventListener('click', hideContextMenu)
  })

  onUnmounted(() => {
    document.removeEventListener('click', hideContextMenu)
  })

  return {
    contextMenu,
    showContextMenu,
    hideContextMenu,
    addBookmark,
    removeBookmark
  }
}
