<template>
  <div
    :class="['blueprint-card', { 'selected': isSelected, 'loading': isLoading }]"
    @click="handleSelect"
    role="button"
    tabindex="0"
    @keydown.enter.prevent="handleSelect"
    @keydown.space.prevent="handleSelect"
  >
    <div class="card-inner">
      <div class="card-header">
        <div class="header-content">
          <h3 class="card-title">{{ blueprint.name }}</h3>
          <div class="header-meta">
            <span class="meta-item">
              <VaIcon name="layers" size="12px" />
              {{ previewData.layerCount }}
            </span>
            <span class="meta-item">
              <VaIcon name="extension" size="12px" />
              {{ previewData.moduleCount }}
            </span>
            <span class="meta-date">{{ formatDate(blueprint.created_at || blueprint.createdAt) }}</span>
          </div>
        </div>
      </div>

      <div v-if="blueprint.description" class="card-description">
        {{ truncateText(blueprint.description, 65) }}
      </div>

      <div class="card-preview">
        <!-- Show thumbnail image if available -->
        <img
          v-if="blueprint.thumbnail"
          :src="blueprint.thumbnail"
          :alt="blueprint.name + ' thumbnail'"
          class="preview-thumbnail"
        />
        <!-- Otherwise show SVG visualization -->
        <BlueprintPreview
          v-else
          :layerCount="previewData.layerCount"
          :moduleCount="previewData.moduleCount"
          :moduleDistribution="previewData.moduleDistribution"
          :width="180"
          :height="60"
          displayMode="distribution"
        />
      </div>

      <div v-if="previewData.moduleTypes && previewData.moduleTypes.length > 0" class="module-badges">
        <div
          v-for="moduleType in previewData.moduleTypes.slice(0, 3)"
          :key="moduleType"
          :class="['module-badge', getModuleBadgeClass(moduleType)]"
        >
          {{ getModuleDisplayName(moduleType) }}
        </div>
        <div v-if="previewData.moduleTypes.length > 3" class="module-badge more">
          +{{ previewData.moduleTypes.length - 3 }}
        </div>
      </div>

      <div class="card-actions">
        <VaButton
          preset="plain"
          size="small"
          icon="download"
          @click.stop="$emit('load', blueprint)"
          title="Load"
        />
        <VaButton
          preset="plain"
          size="small"
          icon="content_copy"
          @click.stop="$emit('duplicate', blueprint)"
          title="Duplicate"
        />
        <VaButton
          preset="plain"
          size="small"
          icon="delete"
          color="danger"
          @click.stop="$emit('delete', blueprint)"
          title="Delete"
        />
      </div>
    </div>

    <div v-if="isLoading" class="loading-overlay">
      <VaProgressCircle indeterminate size="small" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useBlueprintStore } from '../../state/blueprint-store'
import BlueprintPreview from './BlueprintPreview.vue'

const props = defineProps({
  blueprint: {
    type: Object,
    required: true
  },
  isSelected: {
    type: Boolean,
    default: false
  },
  isLoading: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['select', 'load', 'delete', 'duplicate'])
const blueprintStore = useBlueprintStore()

function handleSelect() {
  emit('select', props.blueprint)

  // Update store selection
  if (props.isSelected) {
    blueprintStore.clearSelection()
  } else {
    blueprintStore.setSelectedBlueprint(props.blueprint.id)
  }
}

const previewData = computed(() => {
  return props.blueprint.preview || {
    layerCount: props.blueprint.layers?.length || 0,
    moduleCount: props.blueprint.module_count || props.blueprint.moduleCount || 0,
    moduleTypes: extractModuleTypes(props.blueprint),
    architecture: props.blueprint.architecture || 'Transformer',
    parameterCount: calculateTotalParams(props.blueprint),
    moduleDistribution: calculateModuleDistribution(props.blueprint)
  }
})

function extractModuleTypes(blueprint) {
  const types = new Set()

  // Extract from modules array
  if (blueprint.modules) {
    blueprint.modules.forEach(mod => {
      const name = mod.name || mod.module?.name || mod.metadata?.name
      if (name) types.add(name)
    })
  }

  // Extract from layer modules
  if (blueprint.layers) {
    blueprint.layers.forEach(layer => {
      ;(layer.preModules || []).forEach(mod => {
        const name = mod.name || mod.module?.name
        if (name) types.add(name)
      })
      ;(layer.postModules || []).forEach(mod => {
        const name = mod.name || mod.module?.name
        if (name) types.add(name)
      })
    })
  }

  return Array.from(types)
}

function calculateModuleDistribution(blueprint) {
  const distribution = {}

  if (blueprint.layers) {
    blueprint.layers.forEach((layer, index) => {
      const count = (layer.preModules?.length || 0) + (layer.postModules?.length || 0)
      if (count > 0) {
        distribution[`layer_${index}`] = count
      }
    })
  }

  return distribution
}

function calculateTotalParams(blueprint) {
  if (blueprint.preview?.parameterCount) {
    return blueprint.preview.parameterCount
  }

  if (blueprint.layers) {
    const total = blueprint.layers.reduce((sum, layer) => {
      return sum + (layer.parameters || layer.param_count || 0)
    }, 0)
    return total
  }

  return 0
}

function formatParams(count) {
  if (!count || count === 0) return '0'
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`
  return count.toString()
}

function formatDate(dateString) {
  if (!dateString) return 'Unknown'
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now - date
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return `${diffDays} days ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`

  return date.toLocaleDateString()
}

function truncateText(text, maxLength) {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

function getModuleBadgeClass(moduleType) {
  const name = moduleType.toLowerCase()
  if (name.includes('attn') || name.includes('attention')) return 'badge-attention'
  if (name.includes('stat') || name.includes('monitor')) return 'badge-stats'
  if (name.includes('token') || name.includes('contributor')) return 'badge-token'
  if (name.includes('gradient') || name.includes('importance')) return 'badge-gradient'
  if (name.includes('neuron') || name.includes('tracker')) return 'badge-neuron'
  if (name.includes('inspector') || name.includes('layer')) return 'badge-inspector'
  return 'badge-default'
}

function getModuleDisplayName(moduleType) {
  // Shorten common module names for display
  const name = moduleType.toLowerCase()
  if (name.includes('attn_visualizer')) return 'Attn'
  if (name.includes('layer_statistics')) return 'Stats'
  if (name.includes('token_contributor')) return 'Token'
  if (name.includes('gradient_importance')) return 'Gradient'
  if (name.includes('neuron_tracker')) return 'Neuron'
  if (name.includes('layer_inspector')) return 'Inspector'

  // Default: take first word and capitalize
  return moduleType.split('_')[0].charAt(0).toUpperCase() +
         moduleType.split('_')[0].slice(1)
}
</script>

<style scoped>
/* Keyboard focus must be visible on the clickable containers below. */
[role='button']:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.blueprint-card {
  position: relative;
  background: var(--va-background-element);
  border: 1.5px solid var(--va-background-border);
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  height: 100%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.blueprint-card:hover {
  border-color: var(--va-primary);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

.blueprint-card.selected {
  border-color: var(--va-primary);
  box-shadow: 0 0 0 2px rgba(var(--va-primary-rgb), 0.15), 0 2px 8px rgba(0, 0, 0, 0.1);
  background: linear-gradient(135deg, var(--va-background-element) 0%, rgba(var(--va-primary-rgb), 0.03) 100%);
}

.blueprint-card.selected::before {
  content: '✓';
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  width: 20px;
  height: 20px;
  background: var(--va-primary);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  z-index: 10;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.blueprint-card.loading {
  opacity: 0.6;
  pointer-events: none;
}

.card-inner {
  padding: 0.875rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  height: 100%;
}

.card-header {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.header-content {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.card-title {
  margin: 0;
  font-size: 0.9375rem;
  font-weight: 600;
  color: var(--va-text-primary);
  line-height: 1.3;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  font-size: 0.75rem;
  color: var(--va-text-secondary);
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-weight: 500;
}

.meta-item .va-icon {
  color: var(--va-primary);
  opacity: 0.7;
}

.meta-date {
  margin-left: auto;
  font-size: 0.6875rem;
  opacity: 0.7;
}

.card-description {
  font-size: 0.8125rem;
  color: var(--va-text-secondary);
  line-height: 1.4;
  margin-top: -0.125rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-preview {
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--va-background-secondary);
  border-radius: 6px;
  padding: 0.5rem;
  margin: 0.125rem 0;
  min-height: 70px;
  position: relative;
  overflow: hidden;
}

.preview-thumbnail {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 4px;
  transition: transform 0.2s ease;
}

.blueprint-card:hover .preview-thumbnail {
  transform: scale(1.02);
}

.module-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.375rem;
  margin-top: auto;
}

.module-badge {
  font-size: 0.6875rem;
  padding: 0.1875rem 0.4375rem;
  border-radius: 3px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.badge-attention {
  background: var(--va-info-10);
  color: var(--va-info);
}

.badge-stats {
  background: var(--va-success-10);
  color: var(--va-success);
}

.badge-token {
  background: var(--va-warning-10);
  color: var(--va-warning);
}

.badge-gradient {
  background: var(--va-danger-10);
  color: var(--va-danger);
}

.badge-neuron {
  background: #e1bee7;
  color: #7b1fa2;
}

.badge-inspector {
  background: #b3e5fc;
  color: #0277bd;
}

.badge-default {
  background: var(--va-background-border);
  color: var(--va-text-secondary);
}

.badge-more {
  background: var(--va-background-border);
  color: var(--va-text-primary);
  font-weight: 600;
}

.card-actions {
  display: flex;
  gap: 0.25rem;
  justify-content: flex-end;
  opacity: 0;
  transition: opacity 0.2s ease;
  padding-top: 0.25rem;
  border-top: 1px solid var(--va-background-border);
}

.blueprint-card:hover .card-actions {
  opacity: 1;
}

.card-actions .va-button {
  min-width: unset;
  padding: 0.25rem;
}

.loading-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 20;
}

@media (max-width: 768px) {
  .card-inner {
    padding: 0.75rem;
  }

  .card-title {
    font-size: 0.875rem;
  }

  .card-actions {
    opacity: 1;
  }
}
</style>
