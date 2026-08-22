<template>
  <VaModal
    v-model="isVisible"
    class="advanced-tuning-modal"
    max-width="600px"
    max-height="80vh"
    fixed-layout
    hide-default-actions
  >
    <template #header>
      <h3 class="modal-title">Advanced Section Tuning</h3>
    </template>

    <div class="tuning-content">
      <div class="section-controls">
        <h4>Section Visibility</h4>
        <div class="control-group">
          <VaCheckbox
            v-model="sectionVisibility.embeddings"
            label="Embeddings"
            @update:model-value="updateSectionVisibility"
          />
          <VaCheckbox
            v-model="sectionVisibility.attention_mechanisms"
            label="Attention Mechanisms"
            @update:model-value="updateSectionVisibility"
          />
          <VaCheckbox
            v-model="sectionVisibility.transformer_blocks"
            label="Transformer Blocks"
            @update:model-value="updateSectionVisibility"
          />
          <VaCheckbox
            v-model="sectionVisibility.feed_forward"
            label="Feed Forward"
            @update:model-value="updateSectionVisibility"
          />
          <VaCheckbox
            v-model="sectionVisibility.normalization"
            label="Normalization"
            @update:model-value="updateSectionVisibility"
          />
          <VaCheckbox
            v-model="sectionVisibility.output_processing"
            label="Output Processing"
            @update:model-value="updateSectionVisibility"
          />
        </div>
      </div>

      <div class="display-options">
        <h4>Display Options</h4>
        <div class="control-group">
          <VaCheckbox
            v-model="displayOptions.showParameters"
            label="Show Parameter Counts"
            @update:model-value="updateDisplayOptions"
          />
          <VaCheckbox
            v-model="displayOptions.showModules"
            label="Show Module Indicators"
            @update:model-value="updateDisplayOptions"
          />
          <VaCheckbox
            v-model="displayOptions.compactMode"
            label="Compact Mode"
            @update:model-value="updateDisplayOptions"
          />
        </div>
      </div>

      <div class="preset-actions">
        <h4>Quick Presets</h4>
        <div class="preset-buttons">
          <VaButton
            size="small"
            preset="secondary"
            @click="applyPreset('all')"
            icon="visibility"
          >
            Show All
          </VaButton>
          <VaButton
            size="small"
            preset="secondary"
            @click="applyPreset('minimal')"
            icon="visibility_off"
          >
            Minimal View
          </VaButton>
          <VaButton
            size="small"
            preset="secondary"
            @click="applyPreset('attention-only')"
            icon="center_focus_strong"
          >
            Attention Only
          </VaButton>
          <VaButton
            size="small"
            preset="secondary"
            @click="resetToDefaults"
            icon="refresh"
          >
            Reset
          </VaButton>
        </div>
      </div>
    </div>

    <template #footer>
      <div class="modal-actions">
        <VaButton
          preset="secondary"
          @click="closeModal"
        >
          Close
        </VaButton>
        <VaButton
          preset="primary"
          @click="applyChanges"
        >
          Apply Changes
        </VaButton>
      </div>
    </template>
  </VaModal>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const isVisible = ref(false)

// Section visibility controls
const sectionVisibility = ref({
  embeddings: true,
  attention_mechanisms: true,
  transformer_blocks: true,
  feed_forward: true,
  normalization: true,
  output_processing: true
})

// Display options
const displayOptions = ref({
  showParameters: true,
  showModules: true,
  compactMode: false
})

// Handle advanced tuning toggle event
const handleAdvancedTuningToggle = (event) => {
  isVisible.value = event.detail.isOpen
}

// Update functions
const updateSectionVisibility = () => {
  // Emit event to update section visibility
  const event = new CustomEvent('section-visibility-update', {
    detail: { visibility: sectionVisibility.value }
  })
  document.dispatchEvent(event)
}

const updateDisplayOptions = () => {
  // Emit event to update display options
  const event = new CustomEvent('display-options-update', {
    detail: { options: displayOptions.value }
  })
  document.dispatchEvent(event)
}

// Preset functions
const applyPreset = (presetType) => {
  switch (presetType) {
    case 'all':
      Object.keys(sectionVisibility.value).forEach(key => {
        sectionVisibility.value[key] = true
      })
      break
    case 'minimal':
      Object.keys(sectionVisibility.value).forEach(key => {
        sectionVisibility.value[key] = key === 'embeddings' || key === 'output_processing'
      })
      break
    case 'attention-only':
      Object.keys(sectionVisibility.value).forEach(key => {
        sectionVisibility.value[key] = key === 'attention_mechanisms' || key === 'embeddings' || key === 'output_processing'
      })
      break
  }
  updateSectionVisibility()
}

const resetToDefaults = () => {
  Object.keys(sectionVisibility.value).forEach(key => {
    sectionVisibility.value[key] = true
  })
  displayOptions.value = {
    showParameters: true,
    showModules: true,
    compactMode: false
  }
  updateSectionVisibility()
  updateDisplayOptions()
}

const closeModal = () => {
  isVisible.value = false
}

const applyChanges = () => {
  updateSectionVisibility()
  updateDisplayOptions()
  closeModal()
}

// Event listeners
onMounted(() => {
  document.addEventListener('advanced-tuning-toggle', handleAdvancedTuningToggle)
})

onUnmounted(() => {
  document.removeEventListener('advanced-tuning-toggle', handleAdvancedTuningToggle)
})
</script>

<style scoped>
.advanced-tuning-modal {
  z-index: 2000;
}

.modal-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--va-text-primary);
}

.tuning-content {
  padding: 1rem 0;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.section-controls h4,
.display-options h4,
.preset-actions h4 {
  margin: 0 0 0.75rem 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--va-text-primary);
  border-bottom: 1px solid var(--va-background-border);
  padding-bottom: 0.5rem;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.preset-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding-top: 1rem;
  border-top: 1px solid var(--va-background-border);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .preset-buttons {
    flex-direction: column;
  }

  .modal-actions {
    flex-direction: column;
  }
}
</style>