<script setup>
import { ref, computed } from 'vue'
import { useLttsState } from '@/state'
import { useWebSocketStore } from '@/state/socket-store'
import { useUIStore } from '@/state/ui-store'
import { useModuleStore } from '@/state/module-store'
import { useToast } from '@/composables/useToast'
import { baseUrl } from '@/constants'
import { collectInferenceModules } from '@/utils/module-collection'
import axios from 'axios'

const state = useLttsState()
const socketStore = useWebSocketStore()
const uiStore = useUIStore()
const moduleStore = useModuleStore()
const toast = useToast()
const userText = computed({
  get: () => uiStore.userPrompt,
  set: (value) => { uiStore.userPrompt = value }
})
const isRunning = ref(false)
const isCheckingCompatibility = ref(false)
const resultText = ref('')
const maxNewTokens = ref(4)
const lastError = ref(null)
const compatibility = ref(null)
const lastRunExport = ref(null)
let _runRequestId = 0

const currentModelId = computed(() => {
  return state.models.selectedModel?.id || state.blueprints.loadedBlueprint?.source?.id
})

const canRunInference = computed(() => {
  return currentModelId.value && userText.value.trim().length > 0 && !isRunning.value
})

const onTextInput = (event) => {
}

const collectAttachedModules = () => {
  return collectInferenceModules({
    blueprint: state.blueprints.loadedBlueprint,
    moduleStore
  })
}

const checkCompatibility = async () => {
  if (!currentModelId.value || isCheckingCompatibility.value) return
  isCheckingCompatibility.value = true
  lastError.value = null
  try {
    const modules = collectAttachedModules()
    const response = await axios.post(`${baseUrl}/api/models/preflight`, {
      model_id: currentModelId.value,
      modules: Object.keys(modules).length > 0 ? modules : null
    })
    compatibility.value = response.data
    if (response.data.compatible) {
      toast.success('Model compatibility check passed')
    } else {
      toast.warning(`Compatibility issues: ${response.data.incompatible_modules.join(', ')}`)
    }
  } catch (error) {
    const message = error.response?.data?.detail || error.message || 'Compatibility check failed'
    lastError.value = message
    toast.error(message)
  } finally {
    isCheckingCompatibility.value = false
  }
}

const exportLastRun = () => {
  if (!lastRunExport.value) return
  const exportPayload = {
    ...lastRunExport.value,
    results: {
      ...(lastRunExport.value.results || {}),
      visualizations: socketStore.visualizations
    }
  }
  const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {
    type: 'application/json'
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  const safeModel = String(lastRunExport.value.model?.id || 'model').replace(/[^a-zA-Z0-9._-]+/g, '_')
  anchor.href = url
  anchor.download = `ltts-run-${safeModel}-${lastRunExport.value.run_id}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

const runInput = async () => {
  if (!canRunInference.value) return

  isRunning.value = true
  resultText.value = ''

  socketStore.isInferenceRunning = true

  lastError.value = null

  try {
    const requestId = ++_runRequestId
    const modules = collectAttachedModules()

    if (Object.keys(modules).length === 0) {
      console.debug('[Inference] No modules attached to any layers in blueprint')
    }

    const response = await axios.post(`${baseUrl}/api/inference`, {
      model_id: currentModelId.value,
      prompt: userText.value.trim(),
      client_token: socketStore.clientToken,
      modules: Object.keys(modules).length > 0 ? modules : null,
      parameters: {
        force_eager_attention: true,
        max_new_tokens: Math.max(1, Math.min(128, Number(maxNewTokens.value) || 4))
      }
    })

    if (requestId !== _runRequestId) {
      // A newer run has started — discard this stale response
      return
    }

    if (response.data.status === 'success') {
      resultText.value = response.data.result
      compatibility.value = response.data.compatibility || null
      lastRunExport.value = response.data.run_export || null

      window.dispatchEvent(new CustomEvent('inference-result', {
        detail: {
          result: response.data.result,
          model_id: response.data.model_id,
          output_info: response.data.output_info
        }
      }))

      toast.success('Inference complete')
    } else if (response.data.status === 'cancelled') {
      resultText.value = ''
      lastError.value = null
      toast.info({ title: 'Inference cancelled', message: 'Stopped by user request' })
    } else {
      throw new Error(response.data.message || 'Inference failed')
    }
  } catch (error) {
    console.error('Inference error:', error)
    const errorMessage = error.response?.data?.detail || error.message || 'Inference failed'
    lastError.value = errorMessage
    toast.error(`Inference failed: ${errorMessage}`)
  } finally {
    isRunning.value = false
    socketStore.isInferenceRunning = false
  }
}
</script>

<template>
  <div class="io-card input">
    <div class="io-header">
      <span class="io-title">Model Input</span>
      <div class="header-actions">
        <label class="token-limit" title="Maximum number of tokens generated in this run">
          <span>Tokens</span>
          <input
            v-model.number="maxNewTokens"
            type="number"
            min="1"
            max="128"
            step="1"
            :disabled="isRunning"
            aria-label="Maximum new tokens"
          />
        </label>
        <VaButton
          size="small"
          preset="secondary"
          @click="checkCompatibility"
          :disabled="!currentModelId || isRunning"
          :loading="isCheckingCompatibility"
          aria-label="Check model compatibility"
        >
          Preflight
        </VaButton>
        <VaButton
          v-if="lastRunExport"
          size="small"
          preset="secondary"
          @click="exportLastRun"
          aria-label="Export last run as JSON"
        >
          Export JSON
        </VaButton>
        <VaButton
          size="small"
          preset="primary"
          @click="runInput"
          :disabled="!canRunInference"
          :loading="isRunning"
          aria-label="Run inference"
        >
          {{ isRunning ? 'Running...' : 'Send' }}
        </VaButton>
      </div>
    </div>
    <div class="io-input-container">
      <VaTextarea
        :resize="false"
        v-model="userText"
        placeholder="Type prompt..."
        min-rows="2"
        max-rows="2"
        class="io-textarea"
        :disabled="isRunning"
        aria-label="Inference prompt"
        @input="onTextInput"
      />
      <div v-if="!currentModelId" class="warning-text">
        Please load a model first
      </div>
      <div v-if="socketStore.isReconnecting" class="reconnect-banner">
        Reconnecting to server...
      </div>
      <div v-if="!socketStore.isConnected && !socketStore.isReconnecting" class="disconnected-banner">
        Disconnected from server.
        <button class="retry-link" @click="socketStore.connect()">Reconnect</button>
      </div>
      <div v-if="lastError" class="error-banner">
        <span class="error-text">{{ lastError }}</span>
        <button class="retry-link" @click="runInput">Retry</button>
        <button class="dismiss-link" @click="lastError = null">Dismiss</button>
      </div>
      <div v-if="compatibility" class="compatibility-banner" :class="{ incompatible: !compatibility.compatible }">
        <strong>{{ compatibility.compatible ? 'Compatible' : 'Compatibility issues' }}</strong>
        <span>
          Hidden states {{ compatibility.capabilities?.hidden_states?.supported ? '✓' : '✗' }} ·
          Eager attention {{ compatibility.capabilities?.eager_attention?.supported ? '✓' : '✗' }} ·
          Gradients {{ compatibility.capabilities?.gradients?.supported ? '✓' : '✗' }} ·
          Tokenizer {{ compatibility.capabilities?.tokenizer_decoding?.supported ? '✓' : '✗' }}
        </span>
        <span v-if="compatibility.incompatible_modules?.length">
          Modules: {{ compatibility.incompatible_modules.join(', ') }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.io-card {
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 10px;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
  width: 100%;
  max-width: none;
}
.io-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}
.io-title {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--va-text-primary);
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.token-limit {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--va-text-secondary);
  font-size: 0.78rem;
  white-space: nowrap;
}
.token-limit input {
  width: 3.4rem;
  height: 2rem;
  padding: 0 0.35rem;
  border: 1px solid var(--va-background-border);
  border-radius: 6px;
  background: var(--va-background-element);
  color: var(--va-text-primary);
}
.io-input-container {
  width: 100%;
}

.io-textarea {
  width: 100%;
  resize: vertical;
}

.io-textarea :deep(textarea) {
  white-space: pre-wrap !important;
  word-wrap: break-word !important;
  font-family: inherit !important;
}

.warning-text {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: var(--va-warning);
  font-style: italic;
}

.debug-info {
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: var(--va-text-secondary);
  font-family: monospace;
  background: var(--va-background-secondary);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
}

.error-banner {
  margin-top: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: rgba(var(--va-danger-rgb, 220, 53, 69), 0.1);
  border: 1px solid var(--va-danger, #dc3545);
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
}

.error-text {
  flex: 1;
  color: var(--va-danger, #dc3545);
}

.reconnect-banner {
  margin-top: 0.5rem;
  padding: 0.4rem 0.75rem;
  background: rgba(var(--va-warning-rgb, 255, 193, 7), 0.1);
  border: 1px solid var(--va-warning, #ffc107);
  border-radius: 6px;
  font-size: 0.8rem;
  color: var(--va-warning, #ffc107);
  text-align: center;
}

.disconnected-banner {
  margin-top: 0.5rem;
  padding: 0.4rem 0.75rem;
  background: rgba(var(--va-danger-rgb, 220, 53, 69), 0.08);
  border: 1px solid var(--va-danger, #dc3545);
  border-radius: 6px;
  font-size: 0.8rem;
  color: var(--va-danger, #dc3545);
  text-align: center;
}

.retry-link, .dismiss-link {
  background: none;
  border: none;
  color: var(--va-primary);
  cursor: pointer;
  font-size: 0.8rem;
  text-decoration: underline;
  padding: 0;
}

.retry-link:hover, .dismiss-link:hover {
  opacity: 0.8;
}

.compatibility-banner {
  margin-top: 0.5rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--va-success, #2e7d32);
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  font-size: 0.78rem;
}

.compatibility-banner.incompatible {
  border-color: var(--va-warning, #ffc107);
}
</style>
