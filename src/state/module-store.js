import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'
import { baseUrl } from '../constants'
import { useToast } from '@/composables/useToast'
import { UUID } from 'uuidjs'

export const useModuleStore = defineStore('modules', () => {
  const modules = ref({})
  const showModulesList = ref(false)
  const isLoading = ref(false)
  const error = ref(null)
  const modulesLoaded = ref(false)
  const attachedModules = ref([])
  const toast = useToast()
  const mutedModules = ref(new Set())

  const moduleList = computed(() => {
    return Object.entries(modules.value).map(([key, value]) => ({
      name: key,
      ...value
    }))
  })

  const attachedModuleCount = computed(() => {
    return attachedModules.value.length
  })

  const mutedList = computed(() => Array.from(mutedModules.value))

  function isMuted(name) {
    if (!name) return false
    return mutedModules.value.has(name)
  }

  function toggleMute(name) {
    if (!name) return
    if (mutedModules.value.has(name)) {
      mutedModules.value.delete(name)
      toast.info(`Module "${name}" unmuted`)
    } else {
      mutedModules.value.add(name)
      toast.warning(`Module "${name}" muted`)
    }
    saveMutedToLocalStorage()
  }

  function loadMutedFromLocalStorage() {
    try {
      const stored = localStorage.getItem('ltts-muted-modules')
      if (stored) {
        const list = JSON.parse(stored)
        if (Array.isArray(list)) {
          mutedModules.value = new Set(list)
        }
      }
    } catch (err) {
      console.warn('Failed to load muted modules from storage:', err)
    }
  }

  function saveMutedToLocalStorage() {
    try {
      localStorage.setItem('ltts-muted-modules', JSON.stringify(Array.from(mutedModules.value)))
    } catch (err) {
      console.warn('Failed to save muted modules to storage:', err)
    }
  }

  async function fetchModules(force = false) {
    if (modulesLoaded.value && !force) return

    isLoading.value = true
    error.value = null
    try {
      modules.value = await loadModules()
      modulesLoaded.value = true
    } catch (err) {
      error.value = 'Failed to fetch modules'
      toast.error({ title: 'Failed to fetch modules', message: err?.message || String(err) })
    } finally {
      isLoading.value = false
    }
  }

  function showModules() {
    showModulesList.value = true
    fetchModules(true) // Force refresh when showing modules
  }

  function hideModules() {
    showModulesList.value = false
  }

  function attachModule(moduleInfo) {
    // Validate event type compatibility
    const module = modules.value[moduleInfo.name]
    if (module && module.metadata?.event_types) {
      const supportedEventTypes = module.metadata.event_types
      if (!supportedEventTypes.includes(moduleInfo.eventType)) {
        const errorMsg = `Cannot attach module "${moduleInfo.name}" at event type "${moduleInfo.eventType}". Supported event types: ${supportedEventTypes.join(', ')}`
        toast.error(errorMsg)
        throw new Error(errorMsg)
      }
    }

    const layerPath = moduleInfo.layerPath || moduleInfo.parameters?.layer_path || null
    const parameters = {
      ...(moduleInfo.parameters || {}),
    }
    if (parameters.emit_mode === undefined || parameters.emit_mode === null) {
      parameters.emit_mode = 'final'
    }
    if (parameters.emit_every_n === undefined || parameters.emit_every_n === null) {
      parameters.emit_every_n = 5
    }
    if (layerPath && parameters.layer_path === undefined) {
      parameters.layer_path = layerPath
    }

    const attachment = {
      id: `${moduleInfo.name}_${Date.now()}`,
      name: moduleInfo.name,
      modelId: moduleInfo.modelId,
      eventType: moduleInfo.eventType,
      layerPath,
      parameters,
      attachedAt: new Date().toISOString()
    }

    attachedModules.value.push(attachment)
    toast.success(`Module "${moduleInfo.name}" attached`)

    return attachment
  }

  function detachModule(attachmentId) {
    const index = attachedModules.value.findIndex(m => m.id === attachmentId)
    if (index !== -1) {
      attachedModules.value.splice(index, 1)
      return true
    }
    return false
  }

  function clearAttachedModules() {
    attachedModules.value = []
  }

  // Initialize muted modules from storage on store creation
  loadMutedFromLocalStorage()

  return {
    modules,
    showModulesList,
    modulesLoaded,
    isLoading,
    error,
    moduleList,
    attachedModules,
    attachedModuleCount,
    mutedModules,
    mutedList,
    isMuted,
    toggleMute,
    loadMutedFromLocalStorage,
    fetchModules,
    showModules,
    hideModules,
    attachModule,
    detachModule,
    clearAttachedModules
  }
})

export const loadModules = async () => {
  const response = await axios.get(`${baseUrl}/api/modules`)
  return response.data.modules
}


export const createModule = async (blueprint, id, modules) => {
  const module = modules.find(m => m.id == id)

  let newModule = {
    source: module,
    token: UUID.generate(),
  }

  blueprint.modules.push(newModule)

  return [newModule, blueprint]
}
