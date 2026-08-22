<template>
  <VaInput
    :model-value="modelValue"
    @update:model-value="handleUpdate"
    type="number"
    :label="param.label || param.name"
    :placeholder="param.placeholder || param.help"
    :disabled="param.disabled"
    :readonly="param.readonly"
    :error="!!error"
    :error-messages="error ? [error] : []"
    :min="param.min"
    :max="param.max"
    :step="param.step || 1"
  >
    <template #appendInner v-if="param.help">
      <VaIcon name="help" size="small" class="help-icon" :title="param.help" />
    </template>
  </VaInput>
</template>

<script setup>
const props = defineProps({
  modelValue: [String, Number],
  param: {
    type: Object,
    required: true
  },
  error: String
})

const emit = defineEmits(['update:modelValue'])

const handleUpdate = (value) => {
  const numValue = value === '' ? null : Number(value)
  emit('update:modelValue', numValue)
}
</script>

<style scoped>
.help-icon {
  opacity: 0.6;
  cursor: help;
}
</style>
