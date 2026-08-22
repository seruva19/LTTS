<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const outputText = ref('')

const handleInferenceResult = (event) => {
  outputText.value = event.detail.result
}

onMounted(() => {
  window.addEventListener('inference-result', handleInferenceResult)
})

onUnmounted(() => {
  window.removeEventListener('inference-result', handleInferenceResult)
})
</script>

<template>
  <div class="io-card output">
    <div class="io-header">
      <span class="io-title">Model Output</span>
    </div>
    <div class="io-output" :class="{ 'empty': !outputText }">
      {{ outputText || 'No output yet' }}
    </div>
  </div>
</template>

<style scoped>
.io-card {
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 10px;
  padding: 0.75rem;
  margin-top: 0.75rem;
  width: 100%;
  max-width: none;
  margin-bottom: 20px;
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
.io-output {
  font-size: 0.9rem;
  color: var(--va-text-primary);
  min-height: 2.25rem;
  padding: 0.5rem;
  background: var(--va-background-secondary);
  border-radius: 6px;
  height: 200px;
}
.io-output.empty {
  color: var(--va-text-secondary);
  font-style: italic;
}
</style>