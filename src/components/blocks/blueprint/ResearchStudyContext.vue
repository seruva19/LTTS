<script setup>
import { computed } from 'vue'
import { useResearchStore } from '@/state/research-store'
import { scrollToLayerByName } from '@/composables/useLayerNavigation'

const research = useResearchStore()
const selectedCandidate = computed(() => research.candidateForLayer(research.selectedLayerName))
const selectedComparison = computed(() => research.comparisonForLayer(research.selectedLayerName))
const selectedCausal = computed(() => research.causalForLayer(research.selectedLayerName))
const selectedResult = computed(() => selectedCausal.value || selectedComparison.value || selectedCandidate.value)
const selectedLayer = computed(() => selectedCausal.value?.layer || selectedComparison.value?.layer_path || selectedCandidate.value?.layerName)
const selectedValue = computed(() => {
  if (selectedCausal.value) {
    const delta = selectedCausal.value.delta_from_baseline
    return `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`
  }
  if (selectedComparison.value) {
    const delta = selectedComparison.value.delta
    return delta === null || delta === undefined
      ? selectedComparison.value.status
      : `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`
  }
  const score = selectedCandidate.value?.score
  return Number.isFinite(score) ? `${score >= 0 ? '+' : ''}${score.toFixed(3)}` : ''
})

const focusSelection = () => {
  if (research.selectedLayerName) scrollToLayerByName(research.selectedLayerName)
}
</script>

<template>
  <section v-if="research.isOpen" class="research-context" aria-label="Active Research Study">
    <div class="research-context-title">
      <VaIcon name="science" size="small" />
      <strong>{{ research.activeStage === 'compare' ? 'Run comparison' : research.activeStage === 'intervene' ? 'Causal scan' : 'Research Study' }}</strong>
    </div>
    <span class="research-context-copy">
      {{ research.manifest || 'Choose a dataset manifest' }}
      <template v-if="research.manifest"> · {{ research.split }} · {{ research.seeds }} seeds</template>
    </span>
    <button
      v-if="selectedResult"
      type="button"
      class="research-selection"
      @click="focusSelection"
    >
      {{ selectedLayer }}
      <strong>{{ selectedValue }}</strong>
    </button>
    <button type="button" class="research-context-close" title="Close Research Study" @click="research.close">
      <VaIcon name="close" size="small" />
    </button>
  </section>
</template>

<style scoped>
.research-context { position: sticky; top: -1rem; z-index: 6; display: flex; align-items: center; gap: 0.6rem; min-height: 38px; margin: -1rem -1rem 0.8rem; padding: 0.35rem 0.7rem; background: color-mix(in srgb, var(--va-background-primary) 94%, transparent); border-bottom: 1px solid var(--va-background-border); backdrop-filter: blur(6px); }
.research-context-title { display: flex; align-items: center; gap: 0.3rem; white-space: nowrap; }
.research-context-title strong { font-size: 0.76rem; font-weight: 600; }
.research-context-copy { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--va-text-secondary); font-size: 0.68rem; }
.research-selection { margin-left: auto; display: flex; align-items: center; gap: 0.35rem; max-width: 36%; border: 1px solid var(--va-background-border); border-radius: 2px; background: var(--va-background-secondary); color: var(--va-text-primary); padding: 0.3rem 0.4rem; cursor: pointer; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-size: 0.68rem; }
.research-selection strong { color: var(--va-primary); }
.research-context-close { border: 0; border-radius: 2px; background: transparent; color: var(--va-text-primary); padding: 0.25rem; cursor: pointer; }
@media (max-width: 760px) { .research-context-copy { display: none; } .research-selection { max-width: 52%; } }
</style>
