export const MODEL_MODULES_KEY = '__model__'

function normalizeModuleEntry(entry) {
  if (!entry) return null
  if (typeof entry === 'string') {
    return { name: entry, parameters: {} }
  }
  const candidateName = entry.name || entry.module?.name || entry.metadata?.name || entry.source?.metadata?.name || ''
  if (!candidateName || typeof candidateName !== 'string') return null
  return {
    name: candidateName,
    parameters: entry.parameters ? { ...entry.parameters } : {}
  }
}

// Collect all attached modules (layer pre/post + model-level + module-store attachments)
// into the Dict[str, List[ModuleSpec]] payload expected by /api/inference.
// Never mutates the blueprint or module entries.
export function collectInferenceModules({ blueprint, moduleStore }) {
  const modules = {}
  const isMuted = (name) => (moduleStore?.isMuted ? moduleStore.isMuted(name) : false)

  if (blueprint?.layers) {
    for (const layer of blueprint.layers) {
      const layerModules = []
      for (const entry of [...(layer.preModules || []), ...(layer.postModules || [])]) {
        const normalized = normalizeModuleEntry(entry)
        if (normalized && !isMuted(normalized.name)) {
          layerModules.push(normalized)
        }
      }
      if (layerModules.length > 0) {
        modules[layer.fullPath || layer.name || layer.path] = layerModules
      }
    }
  }

  if (Array.isArray(blueprint?.modelModules)) {
    for (const module of blueprint.modelModules) {
      if (!module?.name || isMuted(module.name)) continue
      const parameters = { ...(module.settings || {}) }
      delete parameters.action
      delete parameters.train_vector
      delete parameters.reset_vector
      modules[MODEL_MODULES_KEY] ||= []
      modules[MODEL_MODULES_KEY].push({ name: module.name, parameters })
    }
  }

  if (moduleStore && Array.isArray(moduleStore.attachedModules)) {
    moduleStore.attachedModules.forEach((attachment) => {
      if (!attachment || isMuted(attachment.name)) return
      const eventType = attachment.eventType || 'layer_after'
      if (eventType === 'model_after' || eventType === 'model_before') {
        modules[MODEL_MODULES_KEY] ||= []
        modules[MODEL_MODULES_KEY].push({
          name: attachment.name,
          parameters: { ...(attachment.parameters || {}), event_type: eventType }
        })
        return
      }
      if (!eventType.startsWith('layer_')) return
      const targetLayer = attachment.layerPath || attachment.parameters?.layer_path || null
      if (!targetLayer) return
      modules[targetLayer] ||= []
      modules[targetLayer].push({
        name: attachment.name,
        parameters: { ...(attachment.parameters || {}), layer_path: targetLayer }
      })
    })
  }

  return modules
}
