import { defineStore } from 'pinia'
import { ref, computed, watch, nextTick } from 'vue'
import { useUIStore } from './ui-store'
import axios from 'axios'
import { baseUrl } from '@/constants'
import { useToast } from '@/composables/useToast'
import { UUID } from 'uuidjs'
import { useWebSocketStore } from './socket-store'
import { useModuleStore } from './module-store'
import { collectInferenceModules } from '@/utils/module-collection'

const NUMERIC_SEGMENT = /^\d+$/

function resolveLayerIdentifier(layer) {
  if (!layer) {
    return ''
  }
  return (
    layer.layerPath ||
    layer.layer_path ||
    layer.fullPath ||
    layer.originalPath ||
    layer.metadata?.layer_path ||
    layer.path ||
    layer.name ||
    ''
  )
}

function estimateArchitecturalLayerCount(layers = []) {
  const identifiers = new Set()
  layers.forEach((layer) => {
    const identifier = resolveLayerIdentifier(layer)
    if (!identifier) {
      return
    }
    const parts = String(identifier).split('.')
    for (let i = 0; i < parts.length; i += 1) {
      if (NUMERIC_SEGMENT.test(parts[i])) {
        identifiers.add(parts.slice(0, i + 1).join('.'))
        break
      }
    }
  })
  return identifiers.size
}

function sumLayerParameters(layers = []) {
  return layers.reduce((total, layer) => {
    const value = layer?.parameters ?? layer?.param_count ?? 0
    return total + (typeof value === 'number' ? value : 0)
  }, 0)
}

function applyBlueprintMetrics(blueprint) {
  if (!blueprint) {
    return
  }
  const layers = Array.isArray(blueprint.layers) ? blueprint.layers : []
  if (typeof blueprint.detailedLayerCount !== 'number') {
    blueprint.detailedLayerCount = layers.length
  }
  if (typeof blueprint.totalParameters !== 'number') {
    blueprint.totalParameters = sumLayerParameters(layers)
  }
  if (typeof blueprint.architecturalLayerCount !== 'number') {
    blueprint.architecturalLayerCount = estimateArchitecturalLayerCount(layers)
  }
}

export const useBlueprintStore = defineStore('blueprints', () => {
  const loadedBlueprint = ref(null)
  const blueprints = ref([])
  const blueprintsLoaded = ref(false)
  const isLoading = ref(false)
  const error = ref(null)
  const selectedBlueprintId = ref(null)
  const toast = useToast()
  let _runRequestId = 0
  let _loadRequestId = 0

  // Auto-save blueprint to local storage when it changes
  function saveBlueprintToLocalStorage() {
    if (loadedBlueprint.value) {
      try {
        applyBlueprintMetrics(loadedBlueprint.value)
        const blueprintData = {
          id: loadedBlueprint.value.id,
          name: loadedBlueprint.value.name,
          description: loadedBlueprint.value.description || '',
          layers: loadedBlueprint.value.layers || [],
          modules: loadedBlueprint.value.modules || [],
          architecture: loadedBlueprint.value.architecture,
          engine: loadedBlueprint.value.engine,
          source: loadedBlueprint.value.source,
          totalParameters: loadedBlueprint.value.totalParameters ?? null,
          architecturalLayerCount: loadedBlueprint.value.architecturalLayerCount ?? null,
          detailedLayerCount: loadedBlueprint.value.detailedLayerCount ?? null,
          // Save categorization data
          categorization: loadedBlueprint.value.categorization || null,
          timestamp: new Date().toISOString()
        }
        localStorage.setItem('ltts-current-blueprint', JSON.stringify(blueprintData))
        console.debug('[Blueprint] Auto-saved to local storage', blueprintData.name)
      } catch (err) {
        console.warn('Failed to save blueprint to local storage:', err)
      }
    }
  }

  // Save selected blueprint to local storage
  function saveSelectionToLocalStorage() {
    if (selectedBlueprintId.value) {
      try {
        localStorage.setItem('ltts-selected-blueprint-id', selectedBlueprintId.value)
        console.debug('[Blueprint] Selection saved to local storage', selectedBlueprintId.value)
      } catch (err) {
        console.warn('Failed to save blueprint selection to local storage:', err)
      }
    } else {
      try {
        localStorage.removeItem('ltts-selected-blueprint-id')
      } catch (err) {
        console.warn('Failed to remove blueprint selection from local storage:', err)
      }
    }
  }

  // Load blueprint from local storage on startup
  function loadBlueprintFromLocalStorage() {
    try {
      const stored = localStorage.getItem('ltts-current-blueprint')
      if (stored) {
        const blueprintData = JSON.parse(stored)

        // Debug logging for categorization data
        console.debug('[Blueprint] Loading from local storage:', blueprintData.name)
        console.debug('[Blueprint] Has categorization data:', !!blueprintData.categorization)
        if (blueprintData.categorization) {
          console.debug('[Blueprint] Categories:', Object.keys(blueprintData.categorization.categories || {}))
        }

        loadedBlueprint.value = blueprintData
        applyBlueprintMetrics(loadedBlueprint.value)
        console.debug('[Blueprint] Loaded from local storage', blueprintData.name)

        // Dispatch a custom event to notify that the blueprint has been loaded
        // This will be used by the model store to detect when the UI is ready
        nextTick(() => {
          document.dispatchEvent(new CustomEvent('blueprint-loaded', {
            detail: {
              modelId: blueprintData.source?.id || 'local-storage',
              blueprint: blueprintData
            }
          }))
        })

        return true
      }
    } catch (err) {
      console.warn('Failed to load blueprint from local storage:', err)
    }
    return false
  }

  // Load blueprint selection from local storage on startup
  function loadSelectionFromLocalStorage() {
    try {
      const stored = localStorage.getItem('ltts-selected-blueprint-id')
      if (stored) {
        selectedBlueprintId.value = stored
        console.debug('[Blueprint] Selection loaded from local storage', stored)
        return true
      }
    } catch (err) {
      console.warn('Failed to load blueprint selection from local storage:', err)
    }
    return false
  }

  // Watch for changes to loadedBlueprint and auto-save (debounced)
  let _autoSaveTimer = null
  watch(loadedBlueprint, () => {
    clearTimeout(_autoSaveTimer)
    _autoSaveTimer = setTimeout(() => {
      saveBlueprintToLocalStorage()
    }, 400)
  }, { deep: true })

  // Watch for changes to selectedBlueprintId and auto-save
  watch(selectedBlueprintId, () => {
    saveSelectionToLocalStorage()
  })

  async function loadBlueprintFromModel(model) {
    if (!model) return

    const requestId = ++_loadRequestId
    isLoading.value = true
    error.value = null
    try {
      const blueprint = await createBlueprint(model)
      if (requestId !== _loadRequestId) {
        // A newer request has started — discard this stale blueprint
        return
      }
      loadedBlueprint.value = blueprint
      applyBlueprintMetrics(loadedBlueprint.value)

      // Dispatch a custom event to notify that the blueprint has been loaded
      // This will be used by the model store to detect when the UI is ready
      document.dispatchEvent(new CustomEvent('blueprint-loaded', {
        detail: {
          modelId: model.id,
          blueprint: blueprint
        }
      }))

      toast.success(`Blueprint loaded from model: ${model.name || model.id || 'Unknown Model'}`)
    } catch (err) {
      error.value = `Failed to load blueprint from model: ${err.message}`
      toast.error({ title: 'Failed to load blueprint', message: err?.message || String(err) })
    } finally {
      isLoading.value = false
    }
  }

  async function fetchBlueprints() {
    isLoading.value = true
    error.value = null
    try {
      const response = await axios.get(`${baseUrl}/api/blueprints/list`)
      if (response.data.status === 'success') {
        // Transform backend blueprints to frontend format for list view
        const blueprintList = Array.isArray(response.data.blueprints) ? response.data.blueprints : []
        blueprints.value = blueprintList.map(bp => ({
          id: bp.id,
          name: bp.name,
          description: bp.description,
          createdAt: bp.created_at,
          updatedAt: bp.updated_at,
          hasModel: bp.has_model,
          moduleCount: bp.module_count,
          tags: bp.tags
        }))
        blueprintsLoaded.value = true
      } else {
        throw new Error(response.data.message || 'Failed to fetch blueprints')
      }
    } catch (err) {
      error.value = `Failed to fetch blueprints: ${err.response?.data?.detail || err.message}`
      toast.error({ title: 'Failed to fetch blueprints', message: err?.response?.data?.detail || err?.message || String(err) })
    } finally {
      isLoading.value = false
    }
  }

  async function saveBlueprint(name, description = '', clientState = null) {
    if (!name) {
      throw new Error('Blueprint name is required')
    }

    isLoading.value = true
    error.value = null

    try {
      if (loadedBlueprint.value) {
        applyBlueprintMetrics(loadedBlueprint.value)
      }

      // Prepare complete blueprint data for saving
      let blueprintData = clientState

      // If no explicit client state provided, use current loaded blueprint
      if (!blueprintData && loadedBlueprint.value) {
        blueprintData = {
          blueprint: {
            id: loadedBlueprint.value.id,
            name: name,
            description: description,
            layers: loadedBlueprint.value.layers || [],
            modules: loadedBlueprint.value.modules || [],
            architecture: loadedBlueprint.value.architecture,
            engine: loadedBlueprint.value.engine,
            totalParameters: loadedBlueprint.value.totalParameters ?? sumLayerParameters(loadedBlueprint.value.layers || []),
            detailedLayerCount: loadedBlueprint.value.detailedLayerCount ?? (loadedBlueprint.value.layers || []).length,
            architecturalLayerCount: loadedBlueprint.value.architecturalLayerCount ?? estimateArchitecturalLayerCount(loadedBlueprint.value.layers || []),
            categorization: loadedBlueprint.value.categorization || null
          },
          // Include model state if available
          model_state: loadedBlueprint.value.source ? {
            selected_model: {
              id: loadedBlueprint.value.source.id,
              engine: loadedBlueprint.value.source.engine || 'transformers',
              loaded: true,
              loadedAt: new Date().toISOString()
            }
          } : null
        }
        applyBlueprintMetrics(blueprintData.blueprint)
      }

      // If client state is provided but missing blueprint structure, extract from loaded blueprint
      if (blueprintData && !blueprintData.blueprint && loadedBlueprint.value) {
        blueprintData.blueprint = {
          id: loadedBlueprint.value.id,
          name: name,
          description: description,
          layers: loadedBlueprint.value.layers || [],
          modules: loadedBlueprint.value.modules || [],
          architecture: loadedBlueprint.value.architecture,
          engine: loadedBlueprint.value.engine,
          totalParameters: loadedBlueprint.value.totalParameters ?? sumLayerParameters(loadedBlueprint.value.layers || []),
          detailedLayerCount: loadedBlueprint.value.detailedLayerCount ?? (loadedBlueprint.value.layers || []).length,
          architecturalLayerCount: loadedBlueprint.value.architecturalLayerCount ?? estimateArchitecturalLayerCount(loadedBlueprint.value.layers || []),
          categorization: loadedBlueprint.value.categorization || null
        }
        applyBlueprintMetrics(blueprintData.blueprint)
      }

      const response = await axios.post(`${baseUrl}/api/blueprint/save`, {
        name: name,
        description: description,
        client_state: blueprintData,
        model_id: loadedBlueprint.value?.source?.id
      })

      if (response.data.status === 'success') {
        // Refresh blueprint list
        await fetchBlueprints()
        return response.data
      } else {
        throw new Error(response.data.message || 'Failed to save blueprint')
      }
    } catch (err) {
      error.value = `Failed to save blueprint: ${err.response?.data?.detail || err.message}`
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function loadBlueprint(blueprintId) {
    if (!blueprintId) {
      throw new Error('Blueprint ID is required')
    }

    isLoading.value = true
    error.value = null

    try {
      const response = await axios.post(`${baseUrl}/api/blueprint/load`, {
        blueprint_id: blueprintId
      })

      if (response.data.status === 'success') {
        // Transform the backend blueprint structure to match frontend expectations
        const backendBlueprint = response.data.blueprint

        // Extract layers - handle various possible structures
        let layers = []
        if (backendBlueprint.model_state) {
          // Try to get layers from layer_structure
          if (backendBlueprint.model_state.layer_structure) {
            if (Array.isArray(backendBlueprint.model_state.layer_structure)) {
              layers = backendBlueprint.model_state.layer_structure
            } else if (Array.isArray(backendBlueprint.model_state.layer_structure.layers)) {
              layers = backendBlueprint.model_state.layer_structure.layers
            } else if (typeof backendBlueprint.model_state.layer_structure === 'object') {
              // Convert object to array
              layers = Object.entries(backendBlueprint.model_state.layer_structure).map(([name, layer]) => ({
                name: name,
                ...layer
              }))
            }
          }

          // If no layers from layer_structure, check if layers exist directly
          if (layers.length === 0 && Array.isArray(backendBlueprint.model_state.layers)) {
            layers = backendBlueprint.model_state.layers
          }
        }

        // Transform attached modules to match expected format
        let modules = []
        if (backendBlueprint.model_state && Array.isArray(backendBlueprint.model_state.attached_modules)) {
          modules = backendBlueprint.model_state.attached_modules.map(module => ({
            name: module.name,
            event_type: module.event_type,
            parameters: module.parameters,
            metadata: module.metadata,
            ui_schema: module.ui_schema,
            attached_at: module.attached_at
          }))
        }

        // Also check client_config for modules (fallback)
        if (modules.length === 0 && backendBlueprint.client_config) {
          if (Array.isArray(backendBlueprint.client_config.modules)) {
            modules = backendBlueprint.client_config.modules
          } else if (backendBlueprint.client_config.blueprint && Array.isArray(backendBlueprint.client_config.blueprint.modules)) {
            modules = backendBlueprint.client_config.blueprint.modules
          }
        }

        // Check blueprint layers for preModules and postModules (layer-specific modules)
        if (backendBlueprint.client_config && backendBlueprint.client_config.blueprint) {
          const blueprintLayers = backendBlueprint.client_config.blueprint.layers || []
          if (blueprintLayers.length > 0) {
            layers = blueprintLayers.map((layer, index) => {
              const baseLayer = layers.find(l => l.name === layer.name) || layer
              return {
                name: layer.name || baseLayer.name || `Layer ${index + 1}`,
                id: layer.id || baseLayer.id || UUID.generate(),
                type: layer.type || baseLayer.type || 'unknown',
                parameters: layer.parameters || baseLayer.parameters || 0,
                preModules: Array.isArray(layer.preModules) ? layer.preModules : [],
                postModules: Array.isArray(layer.postModules) ? layer.postModules : [],
                ...baseLayer // Preserve other properties
              }
            })
          }
        }

        const totalParameters = sumLayerParameters(layers)
        const detailedLayerCount = layers.length
        const architecturalLayerCount =
          backendBlueprint.client_config?.blueprint?.architecturalLayerCount ??
          backendBlueprint.model_state?.architectural_layer_count ??
          estimateArchitecturalLayerCount(layers)

        // Create frontend-compatible blueprint structure
        const frontendBlueprint = {
          id: backendBlueprint.id,
          name: backendBlueprint.name || 'Unnamed Blueprint',
          description: backendBlueprint.description || '',
          layers: layers.map((layer, index) => {
            // Ensure layer has required properties
            const layerName = layer.name || `Layer ${index + 1}`
            return {
              name: layerName,
              id: layer.id || UUID.generate(),
              type: layer.type || 'unknown',
              parameters: layer.parameters || layer.param_count || 0,
              preModules: Array.isArray(layer.preModules) ? layer.preModules : [],
              postModules: Array.isArray(layer.postModules) ? layer.postModules : [],
              ...layer // Spread other properties
            }
          }),
          modules: modules,
          architecture: backendBlueprint.model_state?.model_class || 'Transformer',
          engine: backendBlueprint.model_state?.engine || 'transformers',
          source: {
            id: backendBlueprint.model_state?.model_id || backendBlueprint.model_state?.id,
            engine: backendBlueprint.model_state?.engine || 'transformers'
          },
          totalParameters:
            backendBlueprint.client_config?.blueprint?.totalParameters ??
            backendBlueprint.model_state?.total_parameters ??
            totalParameters,
          detailedLayerCount:
            backendBlueprint.client_config?.blueprint?.detailedLayerCount ??
            detailedLayerCount,
          architecturalLayerCount: architecturalLayerCount,
          // Restore categorization data from client_config
          categorization: backendBlueprint.client_config?.blueprint?.categorization || null,
          createdAt: backendBlueprint.created_at,
          updatedAt: backendBlueprint.updated_at
        }

        applyBlueprintMetrics(frontendBlueprint)

        for (const layer of frontendBlueprint.layers || []) {
          normalizeLayerAttachments(layer)
        }

        console.debug('[Blueprint] Loaded blueprint payload', frontendBlueprint)
        loadedBlueprint.value = frontendBlueprint

        // Restore modules to model if we have a loaded model
        await restoreModulesToModel(frontendBlueprint)

        return frontendBlueprint
      } else {
        throw new Error(response.data.message || 'Failed to load blueprint')
      }
    } catch (err) {
      error.value = `Failed to load blueprint: ${err.response?.data?.detail || err.message}`
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function restoreModulesToModel(blueprint) {
    try {
      if (!blueprint || !blueprint.source?.id) return

      const modelId = blueprint.source.id

      // Restore layer-specific modules (preModules and postModules)
      for (const layer of blueprint.layers || []) {
        // Restore pre-layer modules
        for (const preModule of layer.preModules || []) {
          try {
            await axios.post(`${baseUrl}/api/execute_module`, {
              module: preModule.name,
              model_id: modelId,
              module_path: layer.name,
              params: preModule.parameters || {},
              client_token: 'blueprint_restore_' + Date.now()
            })
          } catch (err) {
            console.warn(`Failed to restore pre-module ${preModule.name} to layer ${layer.name}:`, err)
          }
        }

        // Restore post-layer modules
        for (const postModule of layer.postModules || []) {
          try {
            await axios.post(`${baseUrl}/api/execute_module`, {
              module: postModule.name,
              model_id: modelId,
              module_path: layer.name,
              params: postModule.parameters || {},
              client_token: 'blueprint_restore_' + Date.now()
            })
          } catch (err) {
            console.warn(`Failed to restore post-module ${postModule.name} to layer ${layer.name}:`, err)
          }
        }
      }

      // Restore global modules
      for (const module of blueprint.modules || []) {
        try {
          await axios.post(`${baseUrl}/api/execute_module`, {
            module: module.name,
            model_id: modelId,
            params: module.parameters || {},
            client_token: 'blueprint_restore_' + Date.now()
          })
        } catch (err) {
          console.warn(`Failed to restore global module ${module.name}:`, err)
        }
      }

      console.debug('[Blueprint] Module restoration completed for', blueprint.name)
    } catch (err) {
      console.error('Error restoring modules to model:', err)
    }
  }

  async function deleteBlueprint(blueprintId) {
    if (!blueprintId) {
      throw new Error('Blueprint ID is required')
    }

    isLoading.value = true
    error.value = null

    try {
      const response = await axios.delete(`${baseUrl}/api/blueprint/${blueprintId}`)

      if (response.data.status === 'success') {
        // Refresh blueprint list
        await fetchBlueprints()
        return true
      } else {
        throw new Error(response.data.message || 'Failed to delete blueprint')
      }
    } catch (err) {
      error.value = `Failed to delete blueprint: ${err.response?.data?.detail || err.message}`
      throw err
    } finally {
      isLoading.value = false
    }
  }

  function addPreModuleToModel(blueprint, layer, module) {
    // Ensure preModules array exists
    if (!Array.isArray(layer.preModules)) {
      layer.preModules = []
    }
    layer.preModules.push(module)
  }

  function addPostModuleToModel(blueprint, layer, module) {
    // Ensure postModules array exists
    if (!Array.isArray(layer.postModules)) {
      layer.postModules = []
    }
    layer.postModules.push(module)
  }

  function resolveLayerPath(layer) {
    if (!layer) return null

    const metadataPath = layer.metadata?.layer_path || layer.metadata?.full_path
    return layer.layerPath ||
      layer.layer_path ||
      layer.fullPath ||
      layer.originalPath ||
      metadataPath ||
      layer.path ||
      layer.name ||
      null
  }

  function normalizeLayerAttachments(layer) {
    if (!layer) {
      return
    }

    const layerPath = resolveLayerPath(layer)

    const normalizeCollection = (collection) => {
      if (!Array.isArray(collection)) {
        return
      }

      collection.forEach(attachment => {
        if (!attachment) {
          return
        }
        if (!attachment.parameters || typeof attachment.parameters !== 'object') {
          attachment.parameters = {}
        }
        if (!attachment.layerPath) {
          attachment.layerPath = layerPath
        }
        attachment.parameters = ensureDefaultModuleParameters(attachment.parameters)
      })
    }

    normalizeCollection(layer.preModules)
    normalizeCollection(layer.postModules)
  }

  function ensureDefaultModuleParameters(originalParameters) {
    const params = { ...(originalParameters || {}) }
    if (params.emit_mode === undefined || params.emit_mode === null) {
      params.emit_mode = 'final'
    }
    if (params.emit_every_n === undefined || params.emit_every_n === null) {
      params.emit_every_n = 5
    }
    return params
  }

  function collectLayerReferences(blueprint, layer) {
    const references = new Set()
    if (!layer) {
      return []
    }

    const candidates = []
    const activeBlueprint = blueprint || loadedBlueprint.value

    if (layer) {
      candidates.push(layer)
    }

    if (activeBlueprint?.layers) {
      candidates.push(...activeBlueprint.layers)
    }

    if (activeBlueprint?.categorization?.categories) {
      Object.values(activeBlueprint.categorization.categories).forEach(category => {
        if (Array.isArray(category.layers)) {
          candidates.push(...category.layers)
        }
      })
    }

    const matchesLayer = (candidate) => {
      if (!candidate) return false
      if (candidate === layer) return true
      if (layer.id && candidate.id === layer.id) return true
      if (layer.fullPath && candidate.fullPath === layer.fullPath) return true
      if (layer.path && candidate.path === layer.path) return true
      if (layer.name && candidate.name === layer.name) return true
      return false
    }

    candidates.forEach(candidate => {
      if (matchesLayer(candidate)) {
        references.add(candidate)
      }
    })

    return Array.from(references)
  }

  function ensureModuleAttachment(target, type, attachment) {
    if (!target) {
      return
    }

    const key = type === 'pre' ? 'preModules' : 'postModules'
    if (!Array.isArray(target[key])) {
      target[key] = []
    }

    const alreadyExists = target[key].some(existing => existing?.id === attachment.id)
    if (!alreadyExists) {
      target[key].push(attachment)
    }
  }

  function syncModuleAttachment(blueprint, layer, type, attachment) {
    const references = collectLayerReferences(blueprint, layer)
    references.forEach(reference => normalizeLayerAttachments(reference))
    references.forEach(reference => ensureModuleAttachment(reference, type, attachment))
  }

  function addPreModuleToLayer(blueprint, modulesList, layer, moduleName) {
    const availableModules = Array.isArray(modulesList) ? modulesList : modulesList?.value
    const module = availableModules?.find(m => m.name === moduleName)
    if (!module) {
      console.debug('[Blueprint] Unable to attach pre-module', moduleName, 'to', layer?.name, '- module not found')
      return
    }

    // Validate that module supports layer_before event type
    const eventTypes = module.metadata?.event_types || []
    if (eventTypes.length > 0 && !eventTypes.includes('layer_before')) {
      const errorMsg = `Cannot attach module "${moduleName}" as pre-layer module. Module does not support "layer_before" event type. Supported types: ${eventTypes.join(', ')}`
      toast.error(errorMsg)
      console.warn('[Blueprint]', errorMsg)
      return
    }

    const moduleId = `${moduleName}_${Date.now()}`
    const layerPath = resolveLayerPath(layer)
    const attachment = {
      id: moduleId,
      name: module.name,
      module: module.metadata,
      parameters: ensureDefaultModuleParameters(module.parameters),
      layerPath
    }

    syncModuleAttachment(blueprint, layer, 'pre', attachment)

    console.debug('[Blueprint] Attached pre-module', module.name, 'to', layer?.name, attachment)
    return attachment
  }

  function addPostModuleToLayer(blueprint, modulesList, layer, moduleName) {
    const availableModules = Array.isArray(modulesList) ? modulesList : modulesList?.value
    const module = availableModules?.find(m => m.name === moduleName)
    if (!module) {
      console.debug('[Blueprint] Unable to attach post-module', moduleName, 'to', layer?.name, '- module not found')
      return
    }

    // Validate that module supports layer_after event type
    const eventTypes = module.metadata?.event_types || []
    if (eventTypes.length > 0 && !eventTypes.includes('layer_after')) {
      const errorMsg = `Cannot attach module "${moduleName}" as post-layer module. Module does not support "layer_after" event type. Supported types: ${eventTypes.join(', ')}`
      toast.error(errorMsg)
      console.warn('[Blueprint]', errorMsg)
      return
    }

    const moduleId = `${moduleName}_${Date.now()}`
    const layerPath = resolveLayerPath(layer)
    const attachment = {
      id: moduleId,
      name: module.name,
      module: module.metadata,
      parameters: ensureDefaultModuleParameters(module.parameters),
      layerPath
    }

    syncModuleAttachment(blueprint, layer, 'post', attachment)

    console.debug('[Blueprint] Attached post-module', module.name, 'to', layer?.name, attachment)
    return attachment
  }

  function updateModuleParameters(layer, moduleId, parameters) {
    if (!layer || !moduleId) {
      console.warn('updateModuleParameters: layer or moduleId is null/undefined', { layer, moduleId })
      return
    }

    const activeBlueprint = loadedBlueprint.value
    const references = collectLayerReferences(activeBlueprint, layer)

    let updated = false

    references.forEach(reference => {
      ['preModules', 'postModules'].forEach(collectionKey => {
        const collection = reference?.[collectionKey]
        if (Array.isArray(collection)) {
          const targetModule = collection.find(m => m.id === moduleId)
          if (targetModule) {
            targetModule.parameters = parameters
            updated = true
          }
        }
      })
    })

    if (!updated) {
      console.warn('updateModuleParameters: module not found', { layer, moduleId, parameters })
    }
  }

  function removeModuleFromLayer(blueprint, layer, moduleType, moduleId) {
    if (!layer || !moduleId) {
      console.debug('[Blueprint] removeModuleFromLayer: missing layer or moduleId', { layer, moduleId })
      return
    }

    const key = moduleType === 'pre' ? 'preModules' : 'postModules'
    const activeBlueprint = blueprint || loadedBlueprint.value
    const references = collectLayerReferences(activeBlueprint, layer)

    let removalLogged = false

    references.forEach(reference => {
      const collection = reference?.[key]
      if (!Array.isArray(collection)) {
        return
      }

      const index = collection.findIndex(module => module.id === moduleId)
      if (index !== -1) {
        const [removed] = collection.splice(index, 1)
        if (!removalLogged) {
          console.debug('[Blueprint] Removed', `${moduleType}-module`, removed?.name || moduleId, 'from', reference?.name)
          removalLogged = true
        }
      }
    })
  }

  async function runBlueprint() {
    if (!loadedBlueprint.value) {
      throw new Error('No blueprint loaded')
    }

    isLoading.value = true
    error.value = null

    const uiStore = useUIStore()

    // Set global inference running state for loader
    const wsStore = useWebSocketStore()
    wsStore.isInferenceRunning = true

    try {
      // Collect modules by layer name (API expects Dict[str, List[ModuleSpec]])
      const moduleStore = useModuleStore()
      const modules = collectInferenceModules({
        blueprint: loadedBlueprint.value,
        moduleStore
      })

      console.debug('[Blueprint] Running inference with modules payload', modules)
      const runRequestId = ++_runRequestId
      const response = await axios.post(`${baseUrl}/api/inference`, {
        model_id: loadedBlueprint.value.source?.id,
        prompt: uiStore.userPrompt, // Use user input from UI store
        modules: Object.keys(modules).length > 0 ? modules : null,
        client_token: wsStore.clientToken,
        parameters: {
          force_eager_attention: true
        }
      })

      if (runRequestId !== _runRequestId) {
        // A newer run has started — discard this stale response
        return
      }

      if (response.data.status === 'success') {
        toast.success('Blueprint execution started')

        // Emit event so model-output component can update
        window.dispatchEvent(new CustomEvent('inference-result', {
          detail: {
            result: response.data.result,
            model_id: response.data.model_id,
            output_info: response.data.output_info
          }
        }))

        return response.data
      } else if (response.data.status === 'cancelled') {
        toast.info({ title: 'Inference cancelled', message: 'Stopped by user request' })
        return response.data
      } else {
        throw new Error(response.data.message || 'Failed to run blueprint')
      }
    } catch (err) {
      error.value = `Failed to run blueprint: ${err.response?.data?.detail || err.message}`
      toast.error({ title: 'Failed to run blueprint', message: err?.response?.data?.detail || err?.message || String(err) })
      throw err
    } finally {
      isLoading.value = false
      // Reset global inference running state
      wsStore.isInferenceRunning = false
    }
  }

  function haltBlueprint() {
    try {
      // Send stop signal via WebSocket if available
      const wsStore = useWebSocketStore()
      if (wsStore.isConnected) {
        wsStore.send({
          type: 'stop_inference',
          timestamp: new Date().toISOString()
        })
      }

      toast.info('Stop request sent — inference will be cancelled at the next layer boundary')
    } catch (err) {
      error.value = `Failed to stop blueprint: ${err.message}`
      toast.error({ title: 'Failed to stop blueprint', message: err?.message || String(err) })
      throw err
    }
  }

  // Shared wrappers used by toolbar/menu so run/stop logic lives in one place.
  function runLoadedBlueprint() {
    if (!loadedBlueprint.value) {
      toast.warning('Please load a model first')
      return Promise.resolve()
    }
    return runBlueprint().catch(err => {
      // runBlueprint already surfaces the error via toast/error state.
      console.error('Failed to run blueprint:', err)
    })
  }

  function stopLoadedBlueprint() {
    try {
      haltBlueprint()
    } catch (err) {
      console.error('Failed to stop blueprint:', err)
    }
  }

  function setSelectedBlueprint(blueprintId) {
    selectedBlueprintId.value = blueprintId
    console.debug('[Blueprint] Selection updated', blueprintId)
  }

  function clearSelection() {
    selectedBlueprintId.value = null
    console.debug('[Blueprint] Selection cleared')
  }

  function getSelectedBlueprint() {
    if (!selectedBlueprintId.value || !blueprints.value) {
      return null
    }
    return blueprints.value.find(bp => bp.id === selectedBlueprintId.value) || null
  }

  return {
    loadedBlueprint,
    blueprints,
    blueprintsLoaded,
    isLoading,
    error,
    selectedBlueprintId,

    fetchBlueprints,
    loadBlueprintFromModel,
    loadBlueprintFromLocalStorage,
    loadSelectionFromLocalStorage,
    saveBlueprintToLocalStorage,
    saveBlueprint,
    loadBlueprint,
    deleteBlueprint,

    // Selection management
    setSelectedBlueprint,
    clearSelection,
    getSelectedBlueprint,

    addPreModuleToModel,
    addPostModuleToModel,

    addPreModuleToLayer,
    addPostModuleToLayer,
    updateModuleParameters,
    removeModuleFromLayer,
    restoreModulesToModel,
    runBlueprint,
    haltBlueprint,
    runLoadedBlueprint,
    stopLoadedBlueprint,

    // Model-level module functions
    addModelModuleToBlueprint,
    removeModelModuleFromBlueprint,
    updateModelModuleSettings
  }
})

export const loadBlueprints = async () => {
  const response = await axios.get(`${baseUrl}/api/blueprints`)
  return response.data.blueprints
}

export const createBlueprint = async (m) => {
  const { id } = m

  let modelSource = {
    id,
    engine: m.engine || 'transformers',  // Use the model's engine if provided, otherwise default to 'transformers'
  }

  let newBlueprint = {
    source: modelSource,
    token: UUID.generate(),
    name: m.name || id,
    type: 'ltts-model',
    layers: [],
    modules: [],
    modelModules: [],  // Model-level modules

    addPreModule: (layer, module) => {
      layer.preModules.push(module)
    },

    addPostModule: (layer, module) => {
      layer.postModules.push(module)
    },

    addModelModule: (module) => {
      newBlueprint.modelModules.push(module)
    },

    removeModelModule: (moduleName) => {
      newBlueprint.modelModules = newBlueprint.modelModules.filter(m => m.name !== moduleName)
    },

    run: () => {

    },
    stop: () => {

    }
  }

  const response = await axios.post(`${baseUrl}/api/blueprint_from_model`,
    newBlueprint.source
  )
  const { model } = response.data
  const lttsModel = JSON.parse(model)
  const rawLayers = Array.isArray(lttsModel.layers) ? lttsModel.layers : []
  const clonedLayers = rawLayers.map(layer => ({ ...layer }))
  const totalParameters =
    typeof lttsModel.parameters === 'number'
      ? lttsModel.parameters
      : sumLayerParameters(clonedLayers)
  const detailedLayerCount =
    typeof lttsModel.detailedLayerCount === 'number'
      ? lttsModel.detailedLayerCount
      : clonedLayers.length
  const architecturalLayerCount =
    typeof lttsModel.architecturalLayerCount === 'number'
      ? lttsModel.architecturalLayerCount
      : estimateArchitecturalLayerCount(clonedLayers)

  newBlueprint.layers = clonedLayers
  newBlueprint.totalParameters = totalParameters
  newBlueprint.detailedLayerCount = detailedLayerCount
  newBlueprint.architecturalLayerCount = architecturalLayerCount

  // Preserve categorization data from the model structure
  if (lttsModel.categorization) {
    newBlueprint.categorization = lttsModel.categorization
  }

  // Preserve architecture metadata for module compatibility checks.
  const categorizationMetadata = lttsModel.categorization || {}
  newBlueprint.architectureTopology =
    lttsModel.architectureTopology || categorizationMetadata.architectureTopology || null
  newBlueprint.architectureFamily =
    lttsModel.architectureFamily || categorizationMetadata.architectureFamily || null
  newBlueprint.capabilities =
    lttsModel.capabilities || categorizationMetadata.capabilities || {}

  newBlueprint.layers.forEach((l) => {
    l.id = UUID.generate()
    l.preModules = []
    l.postModules = []
  })

  applyBlueprintMetrics(newBlueprint)

  return newBlueprint
}

// Helper functions for model-level modules
export function addModelModuleToBlueprint(blueprint, moduleName, settings = {}) {
  if (!blueprint.modelModules) {
    blueprint.modelModules = []
  }

  blueprint.modelModules.push({
    name: moduleName,
    settings: settings
  })

  return blueprint
}

export function removeModelModuleFromBlueprint(blueprint, moduleName) {
  if (blueprint.modelModules) {
    blueprint.modelModules = blueprint.modelModules.filter(m => m.name !== moduleName)
  }

  return blueprint
}

export function updateModelModuleSettings(blueprint, moduleName, settings) {
  if (blueprint.modelModules) {
    const module = blueprint.modelModules.find(m => m.name === moduleName)
    if (module) {
      module.settings = { ...module.settings, ...settings }
    }
  }

  return blueprint
}
