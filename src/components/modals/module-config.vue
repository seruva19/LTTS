<template>
  <VaModal
    v-model="isVisible"
    title="Configure Module"
    size="large"
    max-height="80vh"
    :ok-text="isExecuting ? 'Executing...' : 'Execute Module'"
    :cancel-text="'Cancel'"
    :disable-ok="!isFormValid || isExecuting"
    @ok="executeModule"
    @cancel="closeModal"
  >
    <div class="module-config">
      <div class="module-header">
        <h4 class="module-title">{{ selectedModule?.metadata?.name || 'Unknown Module' }}</h4>
        <p class="module-description">{{ selectedModule?.metadata?.description }}</p>
        <div class="module-meta">
          <VaChip size="small" color="info">v{{ selectedModule?.metadata?.version }}</VaChip>
          <VaChip size="small" color="secondary">{{ selectedModule?.metadata?.author }}</VaChip>
        </div>
      </div>

      <VaDivider />

      <div class="config-section" v-if="selectedModule?.ui_schema">
        <h5>Parameters</h5>
        <DynamicForm
          ref="dynamicForm"
          :schema="selectedModule.ui_schema"
          :initial-data="moduleParams"
          :real-time-updates="selectedModule.ui_schema.real_time?.enabled"
          :show-actions="false"
          @update="handleParameterUpdate"
          @fieldChange="handleFieldChange"
        />
      </div>

      <div class="config-section" v-else>
        <VaAlert color="warning" border="left">
          This module doesn't provide a configuration interface.
        </VaAlert>
      </div>

      <div class="config-section" v-if="selectedModel">
        <h5>Target Configuration</h5>
        <div class="target-config">
          <VaInput
            v-model="targetConfig.model_id"
            label="Model ID"
            :disabled="true"
            readonly
          />
          <VaSelect
            v-model="targetConfig.event_type"
            label="Event Type"
            :options="eventTypeOptions"
            placeholder="Select when to execute"
          />
          <VaInput
            v-model="targetConfig.layer_path"
            label="Layer Path (optional)"
            placeholder="e.g., transformer.layers.0"
            clearable
          />
        </div>
      </div>

      <div class="execution-preview" v-if="isFormValid">
        <h5>Execution Preview</h5>
        <VaCard stripe stripe-color="info">
          <VaCardContent>
            <pre class="preview-json">{{ executionPreview }}</pre>
          </VaCardContent>
        </VaCard>
      </div>
    </div>
  </VaModal>
</template>

<script setup>
import { ref, computed, watch, reactive } from 'vue'
import { useModuleStore } from '../../state/module-store'
import { useModelStore } from '../../state/model-store'
import { useWebSocketStore } from '../../state/socket-store'
import DynamicForm from '@/components/ui-renderer/dynamic-form.vue'
import axios from 'axios'
import { baseUrl } from '../../constants'

const props = defineProps({
  modelValue: Boolean,
  moduleId: String,
  modelId: String
})

const emit = defineEmits(['update:modelValue', 'moduleExecuted'])

// Use direct store imports to match blueprint manager
const moduleStore = useModuleStore()
const modelStore = useModelStore()
const wsStore = useWebSocketStore()

const dynamicForm = ref(null)
const isExecuting = ref(false)
const isFormValid = ref(true)

const moduleParams = reactive({})
const targetConfig = reactive({
  model_id: '',
  event_type: 'layer_after',
  layer_path: ''
})

const isVisible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

import { useToast } from '@/composables/useToast'

const toast = useToast()

// Debug when modal opens/closes
watch(isVisible, (newValue) => {
  if (newValue && selectedModule.value?.metadata?.name) {
    toast.info(`Configuring ${selectedModule.value.metadata.name} module`)
  }
})

const selectedModule = computed(() => {
  if (!props.moduleId) return null
  return moduleStore.modules[props.moduleId]
})

const selectedModel = computed(() => {
  if (!props.modelId) return null
  return modelStore.models[props.modelId]
})

const eventTypeOptions = computed(() => {
  // Only return event types declared by the module
  const moduleEventTypes = selectedModule.value?.metadata?.event_types || []

  if (moduleEventTypes.length === 0) {
    // If module doesn't declare event types, fall back to defaults for backward compatibility
    console.warn(`Module ${selectedModule.value?.metadata?.name} has no event_types declared, using defaults`)
    return ['layer_before', 'layer_after']
  }

  return moduleEventTypes
})

const executionPreview = computed(() => {
  if (!selectedModule.value || !isFormValid.value) return {}

  return {
    module: selectedModule.value.metadata.name,
    model_id: targetConfig.model_id,
    event_type: targetConfig.event_type,
    layer_path: targetConfig.layer_path || 'auto',
    parameters: { ...moduleParams },
    client_token: wsStore.clientToken
  }
})

// Initialize when module/model changes
watch([() => props.moduleId, () => props.modelId], () => {
  if (props.moduleId && props.modelId) {
    initializeConfig()
  }
}, { immediate: true })

const initializeConfig = () => {
  // Set target config
  targetConfig.model_id = props.modelId

  // Set default event type to the first available event type from the module
  const availableEventTypes = eventTypeOptions.value
  targetConfig.event_type = availableEventTypes.length > 0 ? availableEventTypes[0] : 'layer_after'

  targetConfig.layer_path = ''

  // Initialize module parameters with defaults
  if (selectedModule.value?.ui_schema?.parameters) {
    Object.keys(moduleParams).forEach(key => delete moduleParams[key])

    selectedModule.value.ui_schema.parameters.forEach(param => {
      moduleParams[param.name] = param.default
    })
  }
}

const handleParameterUpdate = (data) => {
  Object.assign(moduleParams, data)
}

const handleFieldChange = ({ field, value, formData }) => {
}

const validateForm = () => {
  // Validate event type compatibility
  const moduleEventTypes = selectedModule.value?.metadata?.event_types || []
  if (moduleEventTypes.length > 0 && !moduleEventTypes.includes(targetConfig.event_type)) {
    toast.error(`Event type "${targetConfig.event_type}" is not supported by this module. Supported types: ${moduleEventTypes.join(', ')}`)
    return false
  }

  if (dynamicForm.value) {
    return dynamicForm.value.validate()
  }
  return true
}

const executeModule = async () => {
  if (!validateForm()) {
    toast.error('Please fill in all required fields')
    isFormValid.value = false
    return
  }

  isExecuting.value = true
  const loadingToast = toast.loading('Executing module...')

  try {
    if (targetConfig.layer_path) {
      moduleParams.layer_path = targetConfig.layer_path
    }

    const payload = {
      module: selectedModule.value.metadata.name,
      model_id: targetConfig.model_id,
      engine: selectedModel.value?.engine || 'transformers',
      params: { ...moduleParams },
      module_path: targetConfig.layer_path || null,
      client_token: wsStore.clientToken
    }

    const response = await axios.post(`${baseUrl}/api/execute_module`, payload)

    if (response.data.status === 'ok') {
      toast.close(loadingToast)

      const attachmentInfo = {
        name: selectedModule.value.metadata.name,
        modelId: props.modelId,
        eventType: targetConfig.event_type,
        layerPath: targetConfig.layer_path || null,
        parameters: { ...moduleParams }
      }

      const result = moduleStore.attachModule(attachmentInfo)
      toast.success(`Module "${selectedModule.value.metadata.name}" finished`)

      emit('moduleExecuted', {
        module: selectedModule.value.metadata.name,
        model_id: targetConfig.model_id,
        params: moduleParams
      })

      closeModal()
    } else {
      toast.close(loadingToast)
      toast.error(`Module execution failed: ${response.data.message || 'Unknown error'}`)
    }
  } catch (error) {
    toast.close(loadingToast)
    toast.error(`Failed to execute module: ${error.response?.data?.detail || error.message}`)
  } finally {
    isExecuting.value = false
  }
}

const closeModal = () => {
  isVisible.value = false
  // Reset form state
  isExecuting.value = false
  isFormValid.value = true
}

</script>

<style scoped>
.module-config {
  padding: 1rem 0;
}

.module-header {
  margin-bottom: 1rem;
}

.module-title {
  margin: 0 0 0.5rem 0;
  color: var(--va-primary);
}

.module-description {
  margin: 0 0 0.5rem 0;
  color: var(--va-secondary);
  font-size: 0.9rem;
}

.module-meta {
  display: flex;
  gap: 0.5rem;
}

.config-section {
  margin: 1.5rem 0;
}

.config-section h5 {
  margin: 0 0 1rem 0;
  color: var(--va-text-primary);
}

.target-config {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1rem;
}

.target-config > :last-child {
  grid-column: 1 / -1;
}

.execution-preview {
  margin-top: 1.5rem;
}

.preview-json {
  font-family: 'Courier New', monospace;
  font-size: 0.875rem;
  background: var(--va-background-secondary);
  padding: 1rem;
  border-radius: 4px;
  overflow-x: auto;
  margin: 0;
}

@media (max-width: 768px) {
  .target-config {
    grid-template-columns: 1fr;
  }
}
</style>
