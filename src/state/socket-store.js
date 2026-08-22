import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

import { useToast } from '@/composables/useToast'
import { wsBaseUrl } from '../constants'
import { UUID as uuidv4 } from 'uuidjs'
import { useModelLogStore } from './model-log-store'

export const useWebSocketStore = defineStore('websocket', () => {
  const ws = ref(null)
  const isConnected = ref(false)
  const messages = ref([])
  const modelLogStore = useModelLogStore()
  const error = ref(null)
  const lastMessage = ref(null)
  const clientToken = ref(uuidv4.generate())
  const toast = useToast()
  const moduleResults = ref({}) // Store results by module/token
  const visualizations = ref([]) // Store visualization data
  const visualizationEvents = ref([]) // Raw renderable-output event history
  const currentProcessingLayer = ref(-1) // Track current processing layer
  const isInferenceRunning = ref(false) // Track inference state

  // Reconnection state
  let _reconnectTimer = null
  let _reconnectAttempts = 0
  const _maxReconnectAttempts = 20
  const _baseReconnectDelay = 1000 // 1 second
  const _maxReconnectDelay = 30000 // 30 seconds
  const isReconnecting = ref(false)

  // Heartbeat state
  let _heartbeatTimer = null
  const _heartbeatInterval = 30000 // 30 seconds
  let _lastPongAt = 0

  const MAX_MESSAGES = 1000
  let _replayDepth = 0

  // Event system for custom events
  const eventListeners = ref({})

  // Add event listener method
  function addEventListener(eventType, callback) {
    if (!eventListeners.value[eventType]) {
      eventListeners.value[eventType] = []
    }
    eventListeners.value[eventType].push(callback)
  }

  // Remove event listener method
  function removeEventListener(eventType, callback) {
    if (eventListeners.value[eventType]) {
      const index = eventListeners.value[eventType].indexOf(callback)
      if (index > -1) {
        eventListeners.value[eventType].splice(index, 1)
      }
    }
  }

  // Emit custom event
  function emitEvent(eventType, data = null) {
    if (eventListeners.value[eventType]) {
      eventListeners.value[eventType].forEach(callback => {
        try {
          callback(data)
        } catch (err) {
          console.error(`Error in event listener for ${eventType}:`, err)
        }
      })
    }
  }

  function _clearReconnectTimer() {
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer)
      _reconnectTimer = null
    }
  }

  function _clearHeartbeat() {
    if (_heartbeatTimer) {
      clearInterval(_heartbeatTimer)
      _heartbeatTimer = null
    }
  }

  function _startHeartbeat() {
    _clearHeartbeat()
    _lastPongAt = Date.now()
    _heartbeatTimer = setInterval(() => {
      if (ws.value && ws.value.readyState === WebSocket.OPEN) {
        if (Date.now() - _lastPongAt > _heartbeatInterval * 2) {
          // No pong for 2 intervals — connection is stale, force reconnect
          try {
            ws.value.close()
          } catch {
            // ignore, onclose will handle reconnect
          }
          return
        }
        try {
          ws.value.send(JSON.stringify({ type: 'ping', token: 'heartbeat' }))
        } catch {
          // Connection probably dead — close will trigger reconnect
        }
      }
    }, _heartbeatInterval)
  }

  function _scheduleReconnect() {
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      isReconnecting.value = false
      error.value = 'Max reconnection attempts reached. Refresh the page to retry.'
      toast.error('WebSocket connection lost. Please refresh.')
      return
    }

    isReconnecting.value = true
    // Exponential backoff with jitter
    const delay = Math.min(
      _baseReconnectDelay * Math.pow(2, _reconnectAttempts) + Math.random() * 1000,
      _maxReconnectDelay
    )
    _reconnectAttempts++

    _reconnectTimer = setTimeout(() => {
      if (!isConnected.value) {
        connect()
      }
    }, delay)
  }

  // Initialize WebSocket connection
  function connect() {
    // Don't create duplicate connections
    if (ws.value && (ws.value.readyState === WebSocket.CONNECTING || ws.value.readyState === WebSocket.OPEN)) {
      return
    }

    _reconnectAttempts = 0

    // Match the page's scheme: a build served over HTTPS (Colab proxy, tunnel,
    // …) must use wss, otherwise the browser blocks the mixed-content socket.
    const wsProto =
      typeof window !== 'undefined' && window.location.protocol === 'https:'
        ? 'wss'
        : 'ws'
    const url = `${wsProto}://${wsBaseUrl}/ws`
    error.value = null

    try {
      ws.value = new WebSocket(url)

      ws.value.onopen = () => {
        const wasReconnecting = _reconnectAttempts > 0 || isReconnecting.value
        isConnected.value = true
        isReconnecting.value = false
        error.value = null
        _clearReconnectTimer()
        _startHeartbeat()

        if (wasReconnecting) {
          toast.info('WebSocket reconnected')
        }

        _reconnectAttempts = 0
      }

      ws.value.onclose = (event) => {
        isConnected.value = false
        _clearHeartbeat()

        // Don't reconnect on intentional close (code 1000) or server shutdown
        if (event.code === 1000 || event.code === 1001) {
          isReconnecting.value = false
          return
        }

        _scheduleReconnect()
      }

      ws.value.onerror = () => {
        error.value = 'WebSocket connection error'
        // Don't toast on every error during reconnection — too noisy
        if (!isReconnecting.value) {
          toast.error('WebSocket connection error')
        }
      }

      ws.value.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          // Handle server shutdown notification
          if (data.type === 'server_shutdown' || data.message?.type === 'server_shutdown') {
            toast.info('Server is shutting down...')
            isReconnecting.value = false
            _clearReconnectTimer()
            disconnect()
            return
          }

          handleMessage(data)
        } catch (err) {
          error.value = `Failed to parse message: ${err.message}`
        }
      }
    } catch (err) {
      error.value = `Failed to establish WebSocket connection: ${err.message}`
      _scheduleReconnect()
    }
  }

  function handleMessage(data) {
    lastMessage.value = data
    messages.value.push(data)
    if (messages.value.length > MAX_MESSAGES) {
      messages.value = messages.value.slice(-MAX_MESSAGES)
    }

    // Normalize envelope: prefer nested message payload when present
    const msg = data.message?.type ? data.message : data

    if (msg.type === 'pong') {
      _lastPongAt = Date.now()
      return
    }

    // Check if this is a control vector message based on token
    if (data.token === 'control_vector_events') {
      handleVisualization(data.message)
      return
    }

    // Also check for visualization messages with default token
    if (msg.type === 'visualization' && data.token === 'default') {
      handleVisualization(msg)
      return
    }

    // Handle different message types
    switch (msg.type) {
      case 'config_update':
        // Handle configuration updates
        emitEvent('config_update', data)
        break
      case 'notification':
        // Handle notifications
        emitEvent('notification', data)
        break
      case 'visualization':
        // Handle module visualization results
        handleVisualization(msg)
        emitEvent('visualization', data)
        break
      case 'module_result':
        // Handle module execution results
        handleModuleResult(msg)
        emitEvent('module_result', data)
        break
      case 'module_event_log': {
        modelLogStore.addFromPayload(msg, { defer: isInferenceRunning.value })
        emitEvent('module_event_log', data)
        break
      }
      case 'error':
        // Handle error messages
        handleError(msg)
        modelLogStore.flushPending()
        emitEvent('error', data)
        break
      case 'info':
        // Handle info messages
        toast.info(msg.message)
        emitEvent('info', data)
        break
      case 'inference_start':
        isInferenceRunning.value = true
        modelLogStore.beginBatch()
        // Clear visualizations on inference start to prevent accumulation
        clearVisualizations()
        emitEvent('inference_start', data)
        break
      case 'inference_complete':
        isInferenceRunning.value = false
        currentProcessingLayer.value = -1
        modelLogStore.flushPending()
        emitEvent('inference_complete', data)
        break
      case 'inference_cancelled':
        isInferenceRunning.value = false
        currentProcessingLayer.value = -1
        modelLogStore.flushPending()
        emitEvent('inference_cancelled', data)
        break
      case 'layer_processing':
        currentProcessingLayer.value = msg.layer_index ?? -1
        emitEvent('layer_processing', data)
        break
      case 'layer_inference':
        currentProcessingLayer.value = msg.layer_index ?? -1
        emitEvent('layer_inference', data)
        break
      case 'control_vector_processed':
        // Check if this message contains visualization data
        if (msg.visualization || msg.data?.visualization) {
          handleVisualization(msg)
        }
        emitEvent('control_vector_processed', data)
        break
      default:
        emitEvent('unknown_message', data)
    }
  }

  function replayMessage(data) {
    _replayDepth += 1
    try {
      handleMessage(data)
    } finally {
      _replayDepth -= 1
    }
  }

  const MULTI_VIZ_KEYS = ['outputs', 'payloads', 'visualizations', 'items']

  const isMultiPayloadArray = (value) => Array.isArray(value) && value.length > 0 &&
    value.every((item) => item && typeof item === 'object' && !Array.isArray(item))

  function handleVisualization(data) {
    if (!data) {
      return
    }

    if (Array.isArray(data)) {
      data.forEach((entry) => handleVisualization(entry))
      return
    }

    if (typeof data !== 'object') {
      return
    }

    const nestedCandidate = (data && typeof data.data === 'object' && data.data !== null && !Array.isArray(data.data))
      ? data.data
      : null

    let multiPayload = null

    for (const key of MULTI_VIZ_KEYS) {
      if (isMultiPayloadArray(data[key])) {
        multiPayload = { source: 'data', key, entries: data[key] }
        break
      }
      if (nestedCandidate && isMultiPayloadArray(nestedCandidate[key])) {
        multiPayload = { source: 'nested', key, entries: nestedCandidate[key] }
        break
      }
    }

    if (!multiPayload && isMultiPayloadArray(data.data)) {
      multiPayload = { source: 'data', key: 'data', entries: data.data }
    }

    if (!multiPayload && nestedCandidate && isMultiPayloadArray(nestedCandidate.data)) {
      multiPayload = { source: 'nested', key: 'data', entries: nestedCandidate.data }
    }

    if (multiPayload) {
      const baseEnvelope = { ...data }

      if (multiPayload.source === 'data') {
        if (multiPayload.key === 'data') {
          delete baseEnvelope.data
        } else {
          delete baseEnvelope[multiPayload.key]
        }
      } else if (nestedCandidate) {
        const cleanedNested = { ...nestedCandidate }
        delete cleanedNested[multiPayload.key]
        baseEnvelope.data = cleanedNested
      }

      multiPayload.entries.forEach((entry, index) => {
        const envelope = {
          ...baseEnvelope,
          data: entry.data !== undefined ? entry.data : entry,
          module: entry.module || entry.module_name || baseEnvelope.module,
          module_path: entry.module_path || entry.layer_path || baseEnvelope.module_path || baseEnvelope.layer_path,
          layer_path: entry.layer_path || baseEnvelope.layer_path || baseEnvelope.module_path,
          subtype: entry.subtype || entry.type || baseEnvelope.subtype || baseEnvelope.type,
          emit_id: entry.emit_id || (baseEnvelope.emit_id ? `${baseEnvelope.emit_id}:${index}` : baseEnvelope.emit_id || null),
          emit_mode: entry.emit_mode || baseEnvelope.emit_mode,
          forward_pass: entry.forward_pass ?? baseEnvelope.forward_pass ?? null,
          params: entry.params || baseEnvelope.params,
          event_type: entry.event_type || baseEnvelope.event_type
        }
        handleVisualization(envelope)
      })
      return
    }

    const nestedData = (data && typeof data.data === 'object' && data.data !== null && !Array.isArray(data.data))
      ? data.data
      : null

    const moduleName = data.module || data.module_name ||
      nestedData?.module || nestedData?.module_name || nestedData?.metadata?.module || 'unknown'

    const modulePath = data.module_path || data.layer_path ||
      nestedData?.module_path || nestedData?.layer_path || nestedData?.layer_info?.path

    const visualizationType = data.subtype || nestedData?.subtype || nestedData?.type || data.type

    const params = data.params || nestedData?.params
    const eventType = data.event_type || nestedData?.event_type

    const emitMode = nestedData?.emit_mode || data.data?.emit_mode || data.emit_mode || 'all'
    const forwardPass = nestedData?.forward_pass ?? data.data?.forward_pass ?? data.forward_pass ?? null
    const emitStride = nestedData?.emit_stride ?? data.data?.emit_stride ?? data.emit_stride ?? null
    const deferred = nestedData?.deferred ?? data.data?.deferred ?? data.deferred ?? false

    const visualizationData = (nestedData && Object.prototype.hasOwnProperty.call(nestedData, 'data'))
      ? nestedData.data
      : (data.data !== undefined ? data.data : nestedData ?? data)

    const visualization = {
      id: generateVisualizationId(),
      timestamp: new Date(),
      module: moduleName,
      type: visualizationType,
      data: visualizationData,
      params,
      modulePath,
      eventType,
      emitMode,
      forwardPass,
      emitStride,
      deferred,
      emitId: data.emit_id || nestedData?.emit_id || null
    }

    visualization.normalizedLayerPath = normalizeLayerPath(getVisualizationLayerPath(visualization))

    // A live visualization is a renderable artifact, not an event log entry.
    // Token-by-token updates of the same module/layer/subtype replace the
    // previous artifact instead of filling the UI with duplicate icons.
    // Explicit artifact replay remains lossless for audit/gallery use.
    const existingIndex = _replayDepth > 0 ? -1 : visualizations.value.findIndex((viz) => {
      if (!viz.normalizedLayerPath) {
        viz.normalizedLayerPath = normalizeLayerPath(getVisualizationLayerPath(viz))
      }
      return (
        viz.module === visualization.module &&
        viz.normalizedLayerPath === visualization.normalizedLayerPath &&
        viz.type === visualization.type
      )
    })

    if (existingIndex !== -1) {
      visualization.updateCount = (visualizations.value[existingIndex].updateCount || 1) + 1
      visualizations.value.splice(existingIndex, 1)
    } else {
      visualization.updateCount = 1
    }

    visualizations.value.unshift(visualization) // Add to beginning

    if (_replayDepth === 0) {
      visualizationEvents.value.unshift({ ...visualization })
      if (visualizationEvents.value.length > 200) {
        visualizationEvents.value = visualizationEvents.value.slice(0, 200)
      }
    }

    // Bound live history; replay retains all entries.
    const visualizationLimit = _replayDepth > 0 ? 1000 : 50
    if (visualizations.value.length > visualizationLimit) {
      visualizations.value = visualizations.value.slice(0, visualizationLimit)
    }
  }


  function handleModuleResult(data) {
    const key = `${data.module}_${data.token || 'default'}`
    moduleResults.value[key] = {
      ...data,
      timestamp: new Date()
    }
  }

  function cleanModuleName(raw) {
    if (!raw) {
      return ''
    }
    let name = String(raw)
    // Strip module-object repr like "<ltts_modules.foo.Bar object at 0x...>"
    const reprMatch = name.match(/<([^\s>]+)/)
    if (reprMatch) {
      name = reprMatch[1]
    }
    // Strip the ltts_modules. package prefix
    name = name.replace(/^ltts_modules\./, '')
    return name
  }

  function handleError(data) {
    const moduleName = cleanModuleName(data.module || data.module_name)
    const detail = data.message || data.error || 'error'
    let message = moduleName ? `[${moduleName}] ${detail}` : detail
    if (data.layer_path) {
      message += ` (layer: ${data.layer_path})`
    }

    error.value = message
    // Errors default to manual-close only; give them a finite duration so they auto-dismiss.
    toast.error(message, { duration: 8000 })

    if (modelLogStore && typeof modelLogStore.pushEntry === 'function') {
      modelLogStore.pushEntry({
        timestamp: new Date(),
        eventType: 'error',
        moduleName: moduleName || undefined,
        layerPath: data.layer_path || undefined,
        status: 'error',
        metrics: { errors: 1, last_error: detail },
        raw: { payload: data }
      })
    }
  }

  function send(data) {
    if (!isConnected.value) {
      error.value = 'Cannot send message: WebSocket is not connected'
      return
    }

    try {
      ws.value.send(JSON.stringify(data))
    } catch (err) {
      error.value = `Failed to send message: ${err.message} `
      toast.error('Failed to send WebSocket message')
    }
  }

  function disconnect() {
    _clearReconnectTimer()
    _clearHeartbeat()
    isReconnecting.value = false
    _reconnectAttempts = _maxReconnectAttempts // Prevent auto-reconnect
    if (ws.value) {
      ws.value.close(1000, 'Client disconnect')
      ws.value = null
      isConnected.value = false
    }
  }

  // Cleanup function that could be called when component is unmounted
  function cleanup() {
    disconnect()
    messages.value = []
    lastMessage.value = null
    error.value = null
  }

  function generateVisualizationId() {
    return `viz_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  function clearVisualizations() {
    visualizations.value = []
    visualizationEvents.value = []
  }

  function clearModuleResults() {
    moduleResults.value = {}
  }

  function getVisualizationsByModule(moduleName, layerPath = null) {
    return visualizations.value.filter(viz => {
      const moduleMatch = viz.module === moduleName

      // If no layer path specified, return all for this module
      if (!layerPath) return moduleMatch

      // If layer path specified, also match the layer
      const layerMatch = viz.modulePath === layerPath ||
        viz.data?.layer_info?.path === layerPath ||
        viz.data?.layer_path === layerPath

      return moduleMatch && layerMatch
    })
  }

  function getRecentVisualizationsByModule(moduleName, layerPath = null, maxAge = 30000) {
    const cutoffTime = Date.now() - maxAge // 30 seconds ago

    const filtered = visualizations.value.filter(viz => {
      const moduleMatch = viz.module === moduleName
      const isRecent = viz.timestamp && viz.timestamp.getTime() > cutoffTime

      if (!layerPath) {
        return moduleMatch && isRecent
      }

      const vizLayerPath = getVisualizationLayerPath(viz)


      const layerMatch = normalizeLayerPath(vizLayerPath) === normalizeLayerPath(layerPath)

      return moduleMatch && layerMatch && isRecent
    })

    return filtered
  }

  function getVisualizationLayerPath(viz) {
    return viz.modulePath ||
      viz.data?.layer_path ||
      viz.data?.layer_info?.path ||
      viz.data?.actual_layer_path ||
      viz.data?.layerName ||
      null
  }

  function normalizeLayerPath(path) {
    if (!path || typeof path !== 'string') {
      return null
    }

    let normalized = path
    normalized = normalized.replace(/^original_model\.model\./, '')
    normalized = normalized.replace(/^original_model\./, '')
    normalized = normalized.replace(/^model\./, '')
    normalized = normalized.replace(/^(transformer|encoder|decoder)\./, '')
    return normalized
  }

  return {
    // State
    ws,
    isConnected,
    isReconnecting,
    messages,
    error,
    lastMessage,
    clientToken,
    moduleResults,
    visualizations,
    visualizationEvents,
    currentProcessingLayer,
    isInferenceRunning,

    // Actions
    connect,
    send,
    disconnect,
    cleanup,
    clearVisualizations,
    clearModuleResults,
    getVisualizationsByModule,
    getRecentVisualizationsByModule,
    // Replay recorded server envelopes through the live normalization path.
    replayMessage,

    // Event system
    addEventListener,
    removeEventListener,
    emitEvent
  }
})
