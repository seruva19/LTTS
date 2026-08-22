import { defineStore } from 'pinia'
import { ref, computed, nextTick } from 'vue'
import axios from 'axios'
import { baseUrl } from '../constants'
import { useToast } from '@/composables/useToast'

export const useModelStore = defineStore('models', () => {
  const models = ref({})
  const loadedModels = ref({})
  const selectedModel = ref(null)
  const showModelList = ref(false)
  const isLoading = ref(false)
  const error = ref(null)
  const modelsListLoaded = ref(false)
  const isUIReady = ref(false)
  const pendingSuccessMessages = ref([])
  const modelUIRendered = ref({}) // Track UI rendering status for each model
  const modelLoadingComplete = ref({}) // Track actual loading completion for each model
  const isModelLoading = ref(false) // Global state for model loading
  const currentLoadingModelId = ref('') // Currently loading model ID
  const toast = useToast()

  const modelList = computed(() => {
    return Object.entries(models.value).map(([key, value]) => ({
      id: key,
      name: key,
      loaded: !!loadedModels.value[key],
      ...value
    }))
  })

  const currentModel = computed(() => {
    if (selectedModel.value?.id && loadedModels.value[selectedModel.value.id]?.loaded) {
      return loadedModels.value[selectedModel.value.id]
    }
    return Object.values(loadedModels.value).find(model => model.loaded) || null
  })

  async function fetchModels({ force = false } = {}) {
    if (modelsListLoaded.value && !force) return

    isLoading.value = true
    error.value = null
    try {
      models.value = await loadModels()
      modelsListLoaded.value = true
    } catch (err) {
      error.value = 'Failed to fetch models'
      toast.error({ title: 'Failed to fetch models', message: err?.message || String(err) })
    } finally {
      isLoading.value = false
    }
  }

  // Validate model metadata before downloading weights.
  async function addModel(modelId, { task = null, checkRemote = true } = {}) {
    const response = await axios.post(`${baseUrl}/api/models`, {
      model_id: modelId,
      task,
      check_remote: checkRemote
    })
    await fetchModels({ force: true })
    return response.data
  }

  async function removeModel(modelId) {
    await axios.delete(`${baseUrl}/api/models/${modelId}`)
    await fetchModels({ force: true })
  }

  let _loadRequestId = 0

  async function loadModel(modelId, engine = 'transformers', params = {}) {
    const requestId = ++_loadRequestId
    isLoading.value = true
    isModelLoading.value = true
    currentLoadingModelId.value = modelId
    error.value = null

    // Reset model loading status
    modelLoadingComplete.value[modelId] = false
    modelUIRendered.value[modelId] = false

    try {
      const response = await axios.post(`${baseUrl}/api/models/load`, {
        model_id: modelId,
        engine,
        params
      })

      if (requestId !== _loadRequestId) {
        // A newer load request has started — discard this stale response
        return
      }

      if (response.data.status === 'ok') {
        loadedModels.value[modelId] = {
          id: modelId,
          engine,
          loaded: true,
          loadedAt: new Date(),
          ...models.value[modelId]
        }

        // Set as selected model
        selectedModel.value = loadedModels.value[modelId]

        // Mark model as loaded but not yet rendered
        modelLoadingComplete.value[modelId] = true

        // Dispatch event to notify that the model has been loaded
        document.dispatchEvent(new CustomEvent('model-loading', {
          detail: {
            modelId,
            status: 'model-loaded'
          }
        }))

        // Start checking for UI rendering
        startCheckingUIRendering(modelId)

        return response.data
      } else {
        const errorMessage = response.data.message || `Failed to load model ${modelId}`
        error.value = errorMessage
        toast.error({
          title: `Model Loading Failed: ${modelId}`,
          message: errorMessage,
          duration: 8000
        })
        isModelLoading.value = false
        currentLoadingModelId.value = ''
        throw new Error(errorMessage)
      }
    } catch (err) {
      if (requestId !== _loadRequestId) {
        // Stale request — a newer load is in progress, don't clobber its state
        return
      }
      // Format a more user-friendly error message
      const errorMessage = formatModelLoadingError(modelId, err)
      error.value = errorMessage

      // Show a detailed error toast with suggestions
      toast.error({
        title: `Model Loading Failed: ${modelId}`,
        message: errorMessage,
        duration: 8000 // Show for longer time so user can read it
      })

      // Reset global model loading state on error
      isModelLoading.value = false
      currentLoadingModelId.value = ''

      // Log detailed error for debugging
      console.error(`Model loading error for ${modelId}:`, err)

      throw err
    } finally {
      isLoading.value = false
      // Note: We don't reset isModelLoading here because we want it to stay true until UI is rendered
    }
  }

  // Wait for the blueprint store's 'blueprint-loaded' event, then give Vue
  // a couple of frames to flush the render before clearing the loading overlay.
  function startCheckingUIRendering(modelId) {
    const finish = () => {
      if (modelUIRendered.value[modelId]) return
      modelUIRendered.value[modelId] = true

      clearTimeout(fallbackTimer)
      document.removeEventListener('blueprint-loaded', blueprintLoadedHandler)

      isModelLoading.value = false
      currentLoadingModelId.value = ''

      if (modelLoadingComplete.value[modelId]) {
        addPendingSuccessMessage(`Model ${modelId} loaded`)
        document.dispatchEvent(new CustomEvent('ui-fully-rendered', {
          detail: { modelId }
        }))
      }
    }

    // Safety net if 'blueprint-loaded' never fires
    const fallbackTimer = setTimeout(() => {
      if (modelLoadingComplete.value[modelId]) {
        finish()
      }
    }, 15000)

    const blueprintLoadedHandler = () => {
      // Let Vue flush the DOM updates triggered by the blueprint load
      requestAnimationFrame(() => requestAnimationFrame(finish))
    }

    document.addEventListener('blueprint-loaded', blueprintLoadedHandler)

    document.dispatchEvent(new CustomEvent('model-loading', {
      detail: { modelId, status: 'waiting-for-ui' }
    }))
  }

  function unloadModel(modelId) {
    if (loadedModels.value[modelId]) {
      delete loadedModels.value[modelId]

      // Clear selection if this was the selected model, fall back to another loaded one
      if (selectedModel.value?.id === modelId) {
        selectedModel.value = Object.values(loadedModels.value).find(model => model.loaded) || null
      }
    }
  }

  function selectModel(model) {
    selectedModel.value = model
  }

  function showModels() {
    showModelList.value = true
    fetchModels()
  }

  function hideModels() {
    showModelList.value = false
  }

  // Method to set UI ready state
  function setUIReady(ready = true) {
    isUIReady.value = ready
    if (ready) {
      // Process any pending success messages
      processPendingMessages()
    }
  }

  // Add a pending success message
  function addPendingSuccessMessage(message) {
    pendingSuccessMessages.value.push(message)
    // If UI is already ready, process immediately
    if (isUIReady.value) {
      processPendingMessages()
    }
  }

  // Process all pending success messages
  function processPendingMessages() {
    // Use nextTick to ensure UI is fully rendered
    nextTick(() => {
      pendingSuccessMessages.value.forEach(message => {
        toast.success(message)
      })
      pendingSuccessMessages.value = []
    })
  }

  // Helper function to format user-friendly error messages for model loading failures
  function formatModelLoadingError(modelId, error) {
    const errorMessage = error.message || String(error)

    // Check for common error patterns and provide more helpful messages
    if (errorMessage.includes('CORS') || errorMessage.includes('Network Error')) {
      return `Network error while loading model ${modelId}. Please check your internet connection and try again.`
    }

    if (errorMessage.includes('memory') || errorMessage.includes('CUDA') || errorMessage.includes('GPU')) {
      return `Insufficient memory to load model ${modelId}. Try closing other applications or using a smaller model.`
    }

    if (errorMessage.includes('not found') || errorMessage.includes('404')) {
      return `Model ${modelId} could not be found. Please verify the model ID is correct.`
    }

    if (errorMessage.includes('token') || errorMessage.includes('authentication') || errorMessage.includes('auth')) {
      return `Authentication error while loading model ${modelId}. Please check your API token in settings.`
    }

    if (errorMessage.includes('timeout') || errorMessage.includes('timed out')) {
      return `Loading model ${modelId} timed out. The server might be busy or the model is too large.`
    }

    // Default message for other errors
    return `Failed to load model ${modelId}: ${errorMessage}. Please try again or check the console for details.`
  }

  return {
    models,
    loadedModels,
    selectedModel,
    showModelList,
    isLoading,
    error,
    modelList,
    currentModel,
    isUIReady,
    isModelLoading,
    currentLoadingModelId,
    fetchModels,
    addModel,
    removeModel,
    loadModel,
    unloadModel,
    selectModel,
    showModels,
    hideModels,
    setUIReady,
    addPendingSuccessMessage
  }
})

export const loadModels = async () => {
  const response = await axios.get(`${baseUrl}/api/models`)
  return response.data.models
}
