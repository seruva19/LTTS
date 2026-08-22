<template>
  <VaCheckbox
    :model-value="modelValue"
    @update:model-value="handleUpdate"
    :label="param.label || param.name"
    :disabled="param.disabled"
    :readonly="param.readonly"
    :error="!!error"
    :error-messages="error ? [error] : []"
  >
    <template #appendInner v-if="param.help">
      <VaIcon name="help" size="small" class="help-icon" :title="param.help" />
    </template>
  </VaCheckbox>
</template>

<script setup>
defineProps({
  modelValue: Boolean,
  param: {
    type: Object,
    required: true
  },
  error: String
})

const emit = defineEmits(['update:modelValue'])

const handleUpdate = (value) => {
  try {
    emit('update:modelValue', value)
  } catch (error) {
    console.error('Error in boolean field update:', error, { value })
  }
}
</script>

<style scoped>
.help-icon {
  opacity: 0.6;
  cursor: help;
  margin-left: 0.5rem;
}
</style>
