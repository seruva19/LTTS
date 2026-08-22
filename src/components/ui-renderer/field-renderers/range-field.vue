<template>
  <div class="range-field">
    <VaSlider
      :model-value="modelValue"
      @update:model-value="$emit('update:modelValue', $event)"
      :label="param.label || param.name"
      :disabled="param.disabled"
      :readonly="param.readonly"
      :min="param.min || 0"
      :max="param.max || 100"
      :step="param.step || 1"
      :show-track-labels="true"
      :track-label-visible="true"
    />
    <div class="range-value">{{ modelValue }}</div>
    <div v-if="param.help" class="help-text">
      {{ param.help }}
    </div>
    <div v-if="error" class="error-text">
      {{ error }}
    </div>
  </div>
</template>

<script setup>
defineProps({
  modelValue: [Number],
  param: {
    type: Object,
    required: true
  },
  error: String
})

defineEmits(['update:modelValue'])
</script>

<style scoped>
.range-field {
  width: 100%;
}

.range-value {
  text-align: center;
  font-weight: 500;
  margin-top: 0.25rem;
  color: var(--va-primary);
}

.help-text {
  font-size: 0.875rem;
  color: var(--va-secondary);
  margin-top: 0.25rem;
}

.error-text {
  font-size: 0.875rem;
  color: var(--va-danger);
  margin-top: 0.25rem;
}
</style>
