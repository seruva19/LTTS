<template>
  <VaSelect
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :label="param.label || param.name"
    :placeholder="param.placeholder || `Select ${param.label || param.name}`"
    :disabled="param.disabled"
    :readonly="param.readonly"
    :error="!!error"
    :error-messages="error ? [error] : []"
    :options="param.options || []"
    :clearable="param.clearable !== false"
    :multiple="param.multiple"
  >
    <template #appendInner v-if="param.help">
      <VaIcon name="help" size="small" class="help-icon" :title="param.help" />
    </template>
  </VaSelect>
</template>

<script setup>
defineProps({
  modelValue: [String, Number, Array],
  param: {
    type: Object,
    required: true
  },
  error: String
})

defineEmits(['update:modelValue'])
</script>

<style scoped>
.help-icon {
  opacity: 0.6;
  cursor: help;
}
</style>
