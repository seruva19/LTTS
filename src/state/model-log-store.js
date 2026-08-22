import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { UUID as uuidv4 } from 'uuidjs'

const MAX_ENTRIES = 400

function determineSeverity(entry) {
  if (entry.metrics?.errors > 0 || entry.metrics?.last_error) {
    return 'error'
  }
  if (entry.eventType === 'module_uninvoked' || entry.metrics?.invocations === 0) {
    return 'warn'
  }
  if (entry.emitted) {
    return 'success'
  }
  return 'info'
}

export const useModelLogStore = defineStore('model-log', () => {
  const logs = ref([])
  const pendingEntries = ref([])
  const lastUpdate = ref(null)
  const panelVisible = ref(false)
  const hasNewEntries = ref(false)

  function pushEntry(entry) {
    const severity = determineSeverity(entry)
    logs.value.unshift({
      ...entry,
      severity,
      id: entry.id || uuidv4.generate()
    })
    if (logs.value.length > MAX_ENTRIES) {
      logs.value.length = MAX_ENTRIES
    }
    lastUpdate.value = Date.now()
    if (!panelVisible.value) {
      hasNewEntries.value = true
    }
  }

  function normalizeModuleEntry(payload, moduleEntry) {
    const timestamp = payload.timestamp ? new Date(payload.timestamp) : new Date()
    const metrics = moduleEntry.metrics || {}
    return {
      id: moduleEntry.id,
      timestamp,
      eventType: payload.event_type,
      layerPath: moduleEntry.target_layer || payload.layer_path || moduleEntry.requested_layer,
      requestedLayer: moduleEntry.requested_layer,
      moduleName: moduleEntry.module_name,
      moduleRepr: moduleEntry.module_repr,
      emitted: Boolean(moduleEntry.emitted),
      metrics,
      status: moduleEntry.status || metrics.last_status || (moduleEntry.emitted ? 'emitted' : 'no-emission'),
      raw: {
        payload,
        module: moduleEntry
      }
    }
  }

  function addFromPayload(payload, options = {}) {
    if (!payload || !Array.isArray(payload.modules)) {
      return
    }
    const { defer = false } = options
    const entries = payload.modules.map(moduleEntry =>
      normalizeModuleEntry(payload, moduleEntry)
    )
    if (defer) {
      pendingEntries.value.push(...entries)
      return
    }
    entries.forEach(pushEntry)
  }

  function clear() {
    logs.value = []
    pendingEntries.value = []
    hasNewEntries.value = false
  }

  function beginBatch() {
    pendingEntries.value = []
  }

  function flushPending() {
    if (!pendingEntries.value.length) {
      return
    }
    pendingEntries.value.forEach(pushEntry)
    pendingEntries.value = []
  }

  function togglePanel() {
    panelVisible.value = !panelVisible.value
    if (panelVisible.value) {
      hasNewEntries.value = false
    }
  }

  function openPanel() {
    panelVisible.value = true
    hasNewEntries.value = false
  }

  function closePanel() {
    panelVisible.value = false
  }

  function markEntriesSeen() {
    hasNewEntries.value = false
  }

  const allLogs = computed(() => logs.value)
  const recentIssues = computed(() =>
    logs.value.filter(entry => entry.severity === 'error' || entry.severity === 'warn')
  )

  watch(
    () => logs.value.length,
    () => {
      // Automatically remove any undefined entries (defensive)
      logs.value = logs.value.filter(Boolean)
    }
  )

  return {
    logs: allLogs,
    recentIssues,
    lastUpdate,
    panelVisible,
    hasNewEntries,
    addFromPayload,
    pushEntry,
    clear,
    beginBatch,
    flushPending,
    togglePanel,
    openPanel,
    closePanel,
    markEntriesSeen
  }
})
