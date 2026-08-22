<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useModelLogStore } from '@/state/model-log-store'

const logStore = useModelLogStore()
const { logs, panelVisible } = storeToRefs(logStore)

const showOnlyIssues = ref(false)
const showEmissions = ref(true)
const searchTerm = ref('')

const sortedLogs = computed(() => logs.value ?? [])

const filteredLogs = computed(() => {
  const term = searchTerm.value.trim().toLowerCase()
  return sortedLogs.value.filter(entry => {
    if (showOnlyIssues.value && entry.severity === 'info' && !entry.emitted) {
      return false
    }
    if (!showEmissions.value && entry.emitted) {
      return false
    }
    if (!term) {
      return true
    }
    const haystack = [
      entry.moduleName,
      entry.layerPath,
      entry.status,
      entry.eventType,
      entry.moduleRepr
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
    return haystack.includes(term)
  })
})

function clearLogs() {
  logStore.clear()
}

function formatTimestamp(date) {
  if (!(date instanceof Date)) {
    date = new Date(date)
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  }).format(date)
}

function severityClass(entry) {
  switch (entry.severity) {
    case 'error':
      return 'severity-error'
    case 'warn':
      return 'severity-warn'
    case 'success':
      return 'severity-success'
    default:
      return 'severity-info'
  }
}

const handleKeyDown = event => {
  if (event.key === '`' && event.ctrlKey) {
    logStore.togglePanel()
  }
}

watch(searchTerm, newValue => {
  if (newValue && !showEmissions.value) {
    showEmissions.value = true
  }
})

watch(panelVisible, value => {
  if (value) {
    logStore.markEntriesSeen()
  }
})

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})
</script>

<template>
  <transition name="model-log-slide">
    <section v-if="panelVisible" class="model-log-wrapper">
      <div class="model-log-panel">
        <header class="model-log-header">
          <div class="model-log-title">
            <h3>Run Log</h3>
            <span class="model-log-subtitle">
              Latest module diagnostics
            </span>
          </div>
          <div class="model-log-controls">
            <VaSwitch
              v-model="showOnlyIssues"
              label="Issues only"
              class="control-switch"
            />
            <VaSwitch
              v-model="showEmissions"
              label="Include emissions"
              class="control-switch"
            />
            <VaInput
              v-model="searchTerm"
              placeholder="Search..."
              class="model-log-search"
              clearable
            />
            <VaButton
              preset="plain"
              class="control-button"
              @click="clearLogs"
            >
              Clear
            </VaButton>
            <VaButton
              preset="plain"
              class="control-button"
              icon="close"
              title="Close log panel"
              @click="logStore.closePanel"
            />
          </div>
        </header>
        <div class="model-log-content">
          <div
            v-if="filteredLogs.length === 0"
            class="model-log-empty"
          >
            No log entries yet. Attach modules and run inference to capture diagnostics.
          </div>
          <div
            v-for="entry in filteredLogs"
            :key="entry.id"
            class="model-log-entry"
            :class="severityClass(entry)"
          >
            <div class="model-log-entry__header">
              <span class="model-log-entry__time">{{ formatTimestamp(entry.timestamp) }}</span>
              <span class="model-log-entry__module">{{ entry.moduleName }}</span>
              <span v-if="entry.layerPath" class="model-log-entry__layer">{{ entry.layerPath }}</span>
              <VaBadge
                :color="entry.emitted ? 'success' : 'warning'"
                class="model-log-entry__badge"
              >
                {{ entry.emitted ? 'emitted' : 'no output' }}
              </VaBadge>
            </div>
            <div class="model-log-entry__body">
              <div class="model-log-entry__status">
                Event: <strong>{{ entry.eventType }}</strong>
                <span v-if="entry.requestedLayer"> · requested {{ entry.requestedLayer }}</span>
              </div>
              <div class="model-log-entry__metrics">
                <span>invocations: {{ entry.metrics?.invocations ?? 0 }}</span>
                <span>emissions: {{ entry.metrics?.emissions ?? 0 }}</span>
                <span v-if="entry.metrics?.last_status">status: {{ entry.metrics.last_status }}</span>
                <span v-if="entry.metrics?.last_error" class="error-text">error: {{ entry.metrics.last_error }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </transition>
</template>

<style scoped>
.model-log-wrapper {
  width: 100%;
  margin-bottom: 12px;
}

.model-log-panel {
  width: 100%;
  max-height: 260px;
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.model-log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--va-background-border);
}

.model-log-title {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.model-log-title h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
}

.model-log-subtitle {
  font-size: 11px;
  color: var(--va-text-secondary);
}

.model-log-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.model-log-controls :deep(.va-switch__label) {
  font-size: 11px;
}

.control-switch :deep(.va-switch) {
  transform: scale(0.85);
}

.control-button {
  min-height: 28px !important;
  padding: 0 8px !important;
  font-size: 12px !important;
}

.model-log-search {
  width: 150px;
}

.model-log-search :deep(.va-input-wrapper__field) {
  min-height: 28px !important;
  font-size: 12px !important;
}

.model-log-content {
  padding: 10px 16px 14px;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-log-empty {
  padding: 18px;
  text-align: center;
  color: var(--va-text-secondary);
  font-size: 12px;
}

.model-log-entry {
  border-radius: 8px;
  padding: 10px 12px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  background: rgba(255, 255, 255, 0.02);
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11.5px;
}

.severity-error {
  border-color: rgba(255, 99, 132, 0.45);
  background: rgba(255, 99, 132, 0.07);
}

.severity-warn {
  border-color: rgba(255, 193, 7, 0.45);
  background: rgba(255, 193, 7, 0.07);
}

.severity-success {
  border-color: rgba(76, 175, 80, 0.35);
  background: rgba(76, 175, 80, 0.06);
}

.severity-info {
  border-color: rgba(96, 125, 139, 0.25);
}

.model-log-entry__header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.model-log-entry__time {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: var(--va-text-secondary);
}

.model-log-entry__module {
  color: var(--va-primary);
}

.model-log-entry__layer {
  color: var(--va-text-secondary);
}

.model-log-entry__badge {
  margin-left: auto;
}

.model-log-entry__badge :deep(.va-badge) {
  font-size: 10px;
  padding: 2px 6px;
}

.model-log-entry__body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--va-text-secondary);
}

.model-log-entry__metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
}

.error-text {
  color: var(--va-danger);
}

.model-log-slide-enter-active,
.model-log-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.model-log-slide-enter-from,
.model-log-slide-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}
</style>
