<template>
  <div class="dynamic-form">
    <VaForm ref="form" @submit.prevent="handleSubmit">
      <div v-if="schema.layout?.type === 'grid'"
           class="form-grid"
           :style="{ gridTemplateColumns: `repeat(${schema.layout.columns || 2}, 1fr)` }">
        <div v-for="param in schema.parameters" :key="param.name" class="form-field">
          <component
            :is="getFieldComponent(param.type)"
            v-model="formData[param.name]"
            :param="param"
            :error="errors[param.name]"
            @update:modelValue="updateField(param.name, $event)"
          />
        </div>
      </div>

      <div v-else class="form-vertical">
        <div v-for="param in schema.parameters" :key="param.name" class="form-field">
          <component
            :is="getFieldComponent(param.type)"
            v-model="formData[param.name]"
            :param="param"
            :error="errors[param.name]"
            @update:modelValue="updateField(param.name, $event)"
          />
        </div>
      </div>

      <div class="form-actions" v-if="showActions">
        <VaButton type="submit" :loading="loading">
          {{ submitText }}
        </VaButton>
        <VaButton preset="secondary" @click="resetForm" v-if="showReset">
          Reset
        </VaButton>
      </div>
    </VaForm>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import TextFieldRenderer from './field-renderers/text-field.vue'
import NumberFieldRenderer from './field-renderers/number-field.vue'
import SelectFieldRenderer from './field-renderers/select-field.vue'
import BooleanFieldRenderer from './field-renderers/boolean-field.vue'
import RangeFieldRenderer from './field-renderers/range-field.vue'

const props = defineProps({
  schema: {
    type: Object,
    required: true
  },
  initialData: {
    type: Object,
    default: () => ({})
  },
  loading: {
    type: Boolean,
    default: false
  },
  showActions: {
    type: Boolean,
    default: true
  },
  showReset: {
    type: Boolean,
    default: true
  },
  submitText: {
    type: String,
    default: 'Submit'
  },
  realTimeUpdates: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['submit', 'update', 'fieldChange'])

const form = ref(null)
const formData = reactive({})
const errors = reactive({})

// Component mapping for different field types
const fieldComponents = {
  text: TextFieldRenderer,
  number: NumberFieldRenderer,
  select: SelectFieldRenderer,
  boolean: BooleanFieldRenderer,
  range: RangeFieldRenderer
}

const getFieldComponent = (type) => {
  return fieldComponents[type] || TextFieldRenderer
}

// Initialize form data with defaults
const initializeFormData = () => {
  if (!props.schema?.parameters) return

  // Clear stale keys from a previous schema
  Object.keys(formData).forEach(key => delete formData[key])

  props.schema.parameters.forEach(param => {
    const initialValue = props.initialData[param.name] ?? param.default
    formData[param.name] = initialValue
  })
}

// Validate a single field
const validateField = (param, value) => {
  const fieldErrors = []

  // Required validation
  if (param.required && (value === null || value === undefined || value === '')) {
    fieldErrors.push(`${param.label || param.name} is required`)
  }

  // Type-specific validation
  if (value !== null && value !== undefined && value !== '') {
    switch (param.type) {
      case 'number':
        if (isNaN(value)) {
          fieldErrors.push('Must be a valid number')
        } else {
          if (param.min !== undefined && value < param.min) {
            fieldErrors.push(`Must be at least ${param.min}`)
          }
          if (param.max !== undefined && value > param.max) {
            fieldErrors.push(`Must be at most ${param.max}`)
          }
        }
        break

      case 'text':
        if (param.minLength && value.length < param.minLength) {
          fieldErrors.push(`Must be at least ${param.minLength} characters`)
        }
        if (param.maxLength && value.length > param.maxLength) {
          fieldErrors.push(`Must be at most ${param.maxLength} characters`)
        }
        if (param.pattern) {
          const regex = new RegExp(param.pattern)
          if (!regex.test(value)) {
            fieldErrors.push('Invalid format')
          }
        }
        break

      case 'select': {
        if (param.options) {
          const optionValues = param.options.map(option =>
            option && typeof option === 'object' && 'value' in option ? option.value : option
          )
          if (!optionValues.includes(value)) {
            fieldErrors.push('Invalid selection')
          }
        }
        break
      }
    }
  }

  return fieldErrors
}

// Validate entire form
const validateForm = () => {
  const newErrors = {}
  let isValid = true

  if (!props.schema?.parameters) return true

  props.schema.parameters.forEach(param => {
    const fieldErrors = validateField(param, formData[param.name])
    if (fieldErrors.length > 0) {
      newErrors[param.name] = fieldErrors[0] // Show first error
      isValid = false
    }
  })

  Object.keys(errors).forEach(key => delete errors[key])
  Object.assign(errors, newErrors)

  return isValid
}

// Update field value
const updateField = (fieldName, value) => {
  try {
    formData[fieldName] = value

    // Clear error for this field
    if (errors[fieldName]) {
      delete errors[fieldName]
    }

    // Real-time validation
    const param = props.schema.parameters?.find(p => p.name === fieldName)
    if (param) {
      const fieldErrors = validateField(param, value)
      if (fieldErrors.length > 0) {
        errors[fieldName] = fieldErrors[0]
      }
    }

    emit('fieldChange', { field: fieldName, value, formData: { ...formData } })

    // Real-time updates
    if (props.realTimeUpdates || props.schema.real_time?.enabled) {
      const interval = props.schema.real_time?.interval || 500
      clearTimeout(updateField.timeout)
      updateField.timeout = setTimeout(() => {
        emit('update', { ...formData })
      }, interval)
    }
  } catch (error) {
    console.error('Error in updateField:', error, { fieldName, value })
    // Set error state for this field
    errors[fieldName] = 'Field update error'
  }
}

// Handle form submission
const handleSubmit = () => {
  if (validateForm()) {
    emit('submit', { ...formData })
  }
}

// Reset form to defaults
const resetForm = () => {
  Object.keys(errors).forEach(key => delete errors[key])
  initializeFormData()
}

// Get current form data
const getFormData = () => ({ ...formData })

// Set form data programmatically
const setFormData = (data) => {
  Object.keys(formData).forEach(key => delete formData[key])
  Object.assign(formData, data)
}

// Watch for schema changes
watch(() => props.schema, () => {
  initializeFormData()
}, { deep: true })

// Watch for initial data changes
watch(() => props.initialData, (newData) => {
  if (newData) {
    Object.assign(formData, newData)
  }
}, { deep: true })

onMounted(() => {
  initializeFormData()
})

onUnmounted(() => {
  clearTimeout(updateField.timeout)
})

// Expose methods for parent components
defineExpose({
  validate: validateForm,
  getFormData,
  setFormData,
  resetForm
})
</script>

<style scoped>
.dynamic-form {
  width: 100%;
}

.form-grid {
  display: grid;
  gap: 1rem;
  margin-bottom: 1rem;
}

.form-vertical {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1rem;
}

.form-field {
  width: 100%;
}

.form-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  padding-top: 1rem;
  border-top: 1px solid var(--va-background-border);
}

@media (max-width: 768px) {
  .form-grid {
    grid-template-columns: 1fr !important;
  }
}
</style>
