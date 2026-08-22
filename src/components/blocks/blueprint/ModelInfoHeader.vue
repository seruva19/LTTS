<script setup>
import { computed } from 'vue'
import { useLayerFormatting } from '@/composables/useLayerFormatting'

const props = defineProps({
  blueprint: {
    type: Object,
    required: true
  },
  totalModules: {
    type: Number,
    default: 0
  }
})

const { formatNumber } = useLayerFormatting()

const detailedLayerCount = computed(() => {
  if (!props.blueprint) {
    return 0
  }
  if (typeof props.blueprint.detailedLayerCount === 'number') {
    return props.blueprint.detailedLayerCount
  }
  return (props.blueprint.layers || []).length
})

const architecturalLayerCount = computed(() => {
  if (!props.blueprint) {
    return 0
  }
  if (typeof props.blueprint.architecturalLayerCount === 'number') {
    return props.blueprint.architecturalLayerCount
  }
  return detailedLayerCount.value
})

const showArchitecturalCount = computed(() => {
  const value = architecturalLayerCount.value
  return value > 0 && value !== detailedLayerCount.value
})

const totalParameters = computed(() => {
  if (!props.blueprint) {
    return 0
  }
  if (typeof props.blueprint.totalParameters === 'number') {
    return props.blueprint.totalParameters
  }
  return (props.blueprint.layers || []).reduce(
    (sum, layer) => sum + (typeof layer?.parameters === 'number' ? layer.parameters : 0),
    0
  )
})

const architectureName = computed(() => {
  if (props.blueprint.architecture) {
    return props.blueprint.architecture
  }
  const family = props.blueprint.architectureFamily || props.blueprint.modelType
  const topology = props.blueprint.architectureTopology
  return [family, topology]
    .filter(Boolean)
    .map(value => String(value).replaceAll('_', ' '))
    .join(' · ') || 'Transformer'
})

const capabilityLabels = {
  encoder: 'Encoder',
  decoder: 'Decoder',
  self_attention: 'Self-attention',
  cross_attention: 'Cross-attention',
  causal_lm: 'Causal LM',
  masked_lm: 'Masked LM',
  seq2seq_lm: 'Seq2seq LM',
  moe: 'MoE',
  multimodal: 'Multimodal'
}

const activeCapabilities = computed(() => Object.entries(props.blueprint.capabilities || {})
  .filter(([name, supported]) => supported && capabilityLabels[name])
  .map(([name]) => capabilityLabels[name]))
</script>

<template>
  <div class="model-info-compact">
    <div class="model-header-compact">
      <h3 class="model-title-compact">{{ blueprint.name || 'Transformer Model' }}</h3>
      <div class="model-stats-compact">
        <span v-if="showArchitecturalCount" class="stat-compact">
          {{ formatNumber(architecturalLayerCount) }} blocks
        </span>
        <span class="stat-compact">
          {{ formatNumber(detailedLayerCount) }} layers
        </span>
        <span class="stat-compact">
          {{ formatNumber(totalParameters) }} params
        </span>
        <span class="stat-compact architecture-stat">{{ architectureName }}</span>
        <span class="stat-compact">{{ blueprint.engine || 'HuggingFace' }} engine</span>
        <span class="stat-compact">{{ totalModules }} modules</span>
      </div>
    </div>
    <div v-if="activeCapabilities.length" class="capability-row" aria-label="Model capabilities">
      <span class="capability-label">Architecture capabilities</span>
      <span
        v-for="capability in activeCapabilities"
        :key="capability"
        class="capability-chip"
      >
        {{ capability }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.model-info-compact {
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin-bottom: 1rem;
}

.model-header-compact {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.model-title-compact {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--va-text-primary);
  margin: 0;
  flex-shrink: 0;
}

.model-stats-compact {
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
}

.stat-compact {
  font-size: 0.8rem;
  color: var(--va-text-secondary);
  font-weight: 500;
  background: var(--va-background-secondary);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  white-space: nowrap;
}

.architecture-stat {
  color: var(--va-primary);
  border: 1px solid color-mix(in srgb, var(--va-primary) 32%, transparent);
}

.capability-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.65rem;
  padding-top: 0.65rem;
  border-top: 1px solid var(--va-background-border);
}

.capability-label {
  color: var(--va-text-secondary);
  font-size: 0.72rem;
  font-weight: 600;
  margin-right: 0.2rem;
  text-transform: uppercase;
}

.capability-chip {
  background: color-mix(in srgb, var(--va-primary) 12%, var(--va-background-secondary));
  border-radius: 999px;
  color: var(--va-text-primary);
  font-size: 0.72rem;
  padding: 0.18rem 0.48rem;
}

@media (max-width: 768px) {
  .model-header-compact {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }

  .model-stats-compact {
    gap: 0.5rem;
  }
}
</style>
