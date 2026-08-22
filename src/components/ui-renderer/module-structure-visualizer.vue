<template>
  <div class="module-structure-visualizer">
    <div v-if="!module" class="no-module">
      <VaAlert color="warning" border="left">
        No module selected for visualization
      </VaAlert>
    </div>

    <div v-else class="module-structure">
      <!-- Module Header -->
      <div class="module-header">
        <div class="module-title-section">
          <h3 class="module-name">{{ module.metadata?.name || 'Unnamed Module' }}</h3>
          <div class="module-meta">
            <VaChip size="small" color="info">v{{ module.metadata?.version || '1.0.0' }}</VaChip>
            <VaChip size="small" color="secondary">{{ module.metadata?.author || 'Unknown Author' }}</VaChip>
          </div>
        </div>
        <div class="module-actions">
          <VaButton
            size="small"
            preset="primary"
            @click="previewModule"
            :disabled="!hasUISchema"
          >
            Preview UI
          </VaButton>
        </div>
      </div>

      <!-- Module Description -->
      <div class="module-description" v-if="module.metadata?.description">
        <p>{{ module.metadata.description }}</p>
      </div>

      <!-- Module Information Cards -->
      <div class="module-info-grid">
        <VaCard class="info-card">
          <VaCardTitle>Module Capabilities</VaCardTitle>
          <VaCardContent>
            <div class="capability-item" v-if="module.metadata?.event_types">
              <strong>Event Types:</strong>
              <div class="tags">
                <VaChip
                  v-for="eventType in module.metadata.event_types"
                  :key="eventType"
                  size="small"
                  color="primary"
                >
                  {{ eventType }}
                </VaChip>
              </div>
            </div>
            <div class="capability-item" v-if="module.metadata?.dependencies">
              <strong>Dependencies:</strong>
              <div class="tags">
                <VaChip
                  v-for="dep in module.metadata.dependencies"
                  :key="dep"
                  size="small"
                  color="success"
                >
                  {{ dep }}
                </VaChip>
              </div>
            </div>
            <div class="capability-item">
              <strong>Configurable:</strong>
              <VaBadge
                :text="hasUISchema ? 'Yes' : 'No'"
                :color="hasUISchema ? 'success' : 'secondary'"
              />
            </div>
          </VaCardContent>
        </VaCard>

        <!-- UI Schema Visualization -->
        <VaCard class="info-card" v-if="hasUISchema">
          <VaCardTitle>UI Schema Structure</VaCardTitle>
          <VaCardContent>
            <div class="ui-schema-info">
              <div class="schema-stat">
                <strong>Parameters:</strong>
                <span>{{ parameterCount }}</span>
              </div>
              <div class="schema-stat" v-if="module.ui_schema?.layout">
                <strong>Layout:</strong>
                <span>{{ module.ui_schema.layout.type }} ({{ module.ui_schema.layout.columns || 'N/A' }} cols)</span>
              </div>
              <div class="schema-stat" v-if="module.ui_schema?.real_time">
                <strong>Real-time:</strong>
                <span>{{ module.ui_schema.real_time.enabled ? 'Enabled' : 'Disabled' }}</span>
              </div>
            </div>

            <!-- Parameter List -->
            <div class="parameters-section" v-if="module.ui_schema?.parameters?.length">
              <h5>Parameters:</h5>
              <div class="parameter-list">
                <div
                  v-for="param in module.ui_schema.parameters"
                  :key="param.name"
                  class="parameter-item"
                  @click="selectParameter(param)"
                  role="button"
                  tabindex="0"
                  @keydown.enter.prevent="selectParameter(param)"
                  @keydown.space.prevent="selectParameter(param)"
                >
                  <div class="parameter-header">
                    <span class="parameter-name">{{ param.name }}</span>
                    <VaBadge :text="param.type" color="info" size="small" />
                  </div>
                  <div class="parameter-details">
                    <div class="parameter-label">{{ param.label }}</div>
                    <div class="parameter-default" v-if="param.default !== undefined">
                      Default: <code>{{ formatDefaultValue(param.default) }}</code>
                    </div>
                    <div class="parameter-help" v-if="param.help">
                      {{ param.help }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </VaCardContent>
        </VaCard>

        <!-- Method Mappings -->
        <VaCard class="info-card" v-if="module.metadata?.methods">
          <VaCardTitle>Method Mappings</VaCardTitle>
          <VaCardContent>
            <div class="methods-grid">
              <div
                v-for="(method, lifecycle) in module.metadata.methods"
                :key="lifecycle"
                class="method-item"
              >
                <strong>{{ formatLifecycleName(lifecycle) }}:</strong>
                <span class="method-name">{{ method }}</span>
              </div>
            </div>
          </VaCardContent>
        </VaCard>
      </div>

      <!-- Preview Modal -->
      <VaModal
        v-model="showPreview"
        title="Module UI Preview"
        size="large"
        hide-default-actions
      >
        <div class="preview-content">
          <DynamicForm
            v-if="hasUISchema"
            :schema="module.ui_schema"
            :initial-data="previewData"
            :show-actions="true"
            submit-text="Test Configuration"
            @submit="handlePreviewSubmit"
            @fieldChange="handleFieldChange"
          />
          <div v-else class="no-preview">
            <VaAlert color="warning" border="left">
              This module does not provide a UI schema for preview
            </VaAlert>
          </div>
        </div>
        <template #footer>
          <div class="preview-footer">
            <VaButton @click="showPreview = false">Close</VaButton>
          </div>
        </template>
      </VaModal>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import DynamicForm from './dynamic-form.vue'

const props = defineProps({
  module: {
    type: Object,
    default: null
  }
})

const showPreview = ref(false)
const previewData = ref({})
const selectedParameter = ref(null)

// Computed properties
const hasUISchema = computed(() => {
  return props.module?.ui_schema?.parameters?.length > 0
})

const parameterCount = computed(() => {
  return props.module?.ui_schema?.parameters?.length || 0
})

// Methods
const previewModule = () => {
  showPreview.value = true
  // Initialize preview data with default values
  if (props.module?.ui_schema?.parameters) {
    const data = {}
    props.module.ui_schema.parameters.forEach(param => {
      data[param.name] = param.default !== undefined ? param.default : getDefaultValueForType(param.type)
    })
    previewData.value = data
  }
}

const handlePreviewSubmit = (formData) => {
  // Preview submission is disabled.
}

const handleFieldChange = ({ field, value, formData }) => {
  }

const selectParameter = (param) => {
  selectedParameter.value = param
}

const formatDefaultValue = (value) => {
  if (typeof value === 'object') {
    return JSON.stringify(value)
  }
  return String(value)
}

const formatLifecycleName = (lifecycle) => {
  const names = {
    'ctor': 'Constructor',
    'dtor': 'Destructor',
    'ntor': 'Interceptor',
    'utor': 'UI Schema Provider'
  }
  return names[lifecycle] || lifecycle
}

const getDefaultValueForType = (type) => {
  const defaults = {
    'text': '',
    'number': 0,
    'boolean': false,
    'select': '',
    'range': 0
  }
  return defaults[type] || ''
}
</script>

<style scoped>
/* Keyboard focus must be visible on the clickable containers below. */
[role='button']:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.module-structure-visualizer {
  padding: 1rem;
}

.module-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--va-background-border);
}

.module-title-section {
  flex: 1;
}

.module-name {
  margin: 0 0 0.5rem 0;
  color: var(--va-primary);
  font-size: 1.5rem;
}

.module-meta {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.module-description {
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: var(--va-background-element);
  border-radius: 8px;
  border: 1px solid var(--va-background-border);
}

.module-description p {
  margin: 0;
  color: var(--va-text-secondary);
  line-height: 1.5;
}

.module-info-grid {
  column-count: auto;
  column-width: 300px;
  column-gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.info-card {
  height: fit-content;
  break-inside: avoid;
  margin-bottom: 1.5rem;
  display: inline-block;
  width: 100%;
}

.capability-item {
  margin-bottom: 1rem;
}

.capability-item:last-child {
  margin-bottom: 0;
}

.capability-item strong {
  display: block;
  margin-bottom: 0.25rem;
  color: var(--va-text-primary);
}

.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

.ui-schema-info {
  margin-bottom: 1rem;
}

.schema-stat {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--va-background-border);
}

.schema-stat:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.schema-stat strong {
  color: var(--va-text-primary);
}

.parameters-section h5 {
  margin: 1rem 0 0.5rem 0;
  color: var(--va-text-primary);
}

.parameter-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.parameter-item {
  padding: 0.75rem;
  background: var(--va-background-element);
  border-radius: 6px;
  border: 1px solid var(--va-background-border);
  cursor: pointer;
  transition: all 0.2s ease;
}

.parameter-item:hover {
  border-color: var(--va-primary);
  background: var(--va-background-secondary);
}

.parameter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.25rem;
}

.parameter-name {
  font-weight: 600;
  color: var(--va-text-primary);
}

.parameter-details {
  font-size: 0.875rem;
}

.parameter-label {
  color: var(--va-text-secondary);
  margin-bottom: 0.25rem;
}

.parameter-default {
  color: var(--va-text-secondary);
  margin-bottom: 0.25rem;
}

.parameter-default code {
  background: var(--va-background-border);
  padding: 0.125rem 0.25rem;
  border-radius: 3px;
  font-size: 0.75rem;
}

.parameter-help {
  color: var(--va-text-secondary);
  font-style: italic;
}

.methods-grid {
  display: grid;
  gap: 0.75rem;
}

.method-item {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem;
  background: var(--va-background-element);
  border-radius: 4px;
}

.method-item strong {
  color: var(--va-text-primary);
}

.method-name {
  font-family: monospace;
  background: var(--va-background-border);
  padding: 0.125rem 0.25rem;
  border-radius: 3px;
}

.preview-content {
  min-height: 300px;
}

.no-preview {
  padding: 2rem;
  text-align: center;
}

.preview-footer {
  display: flex;
  justify-content: flex-end;
  padding: 1rem 0;
}
</style>
