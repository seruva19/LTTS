<template>
  <VaInput
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :label="param.label || param.name"
    :placeholder="param.placeholder || param.help"
    :disabled="param.disabled"
    :readonly="param.readonly"
    :error="!!error"
    :error-messages="error ? [error] : []"
    :clearable="param.clearable !== false"
  >
    <template #appendInner v-if="param.help">
      <VaIcon name="help" size="small" class="help-icon" :title="param.help" />
    </template>
  </VaInput>
</template>

<script setup>
defineProps({
  modelValue: [String, Number],
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
