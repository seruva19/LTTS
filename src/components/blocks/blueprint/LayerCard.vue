<script setup>
import { computed, inject, unref } from 'vue'
import { useLttsState } from '@/state'
import { useLayerFormatting } from '@/composables/useLayerFormatting'
import { useLayerState } from '@/composables/useLayerState'
import LayerBreadcrumb from './LayerBreadcrumb.vue'
import Module from '@/components/blocks/module.vue'

const props = defineProps({
  layer: {
    type: Object,
    required: true
  },
  isInferenceRunning: {
    type: Boolean,
    default: false
  },
  currentProcessingLayer: {
    type: Number,
    default: -1
  }
})

const state = useLttsState()
const research = state.research || {
  selectedLayerName: null,
  candidateForLayer: () => null,
  selectLayer: () => {}
}
const { formatNumber, formatMemory, getLayerTypeInfo } = useLayerFormatting()
const layerState = inject('layerState')
const architectureContext = inject('architectureContext', computed(() => ({
  topology: 'unknown',
  capabilities: {}
})))

const preModuleOptions = inject('preModuleOptions')
const postModuleOptions = inject('postModuleOptions')
const handleLayerHeaderClick = inject('handleLayerHeaderClick')
const showContextMenu = inject('showContextMenu')
const getLayerIndex = inject('getLayerIndex')
const filterByBreadcrumb = inject('filterByBreadcrumb')

const isExpanded = computed(() => layerState.isExpanded(props.layer.name))
const isSelected = computed(() => layerState.isSelected(props.layer.name))
const researchCandidate = computed(() => research.candidateForLayer(props.layer.name))
const researchComparison = computed(() => research.comparisonForLayer?.(props.layer.name) || null)
const researchCausal = computed(() => research.causalForLayer?.(props.layer.name) || null)
const researchResult = computed(() => researchCausal.value || researchComparison.value || researchCandidate.value)
const isResearchSelected = computed(() => research.selectedLayerName === props.layer.name)
const researchResultLabel = computed(() => {
  const result = researchResult.value
  if (!result) return ''
  if (researchCausal.value) {
    const delta = result.delta_from_baseline
    return `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`
  }
  if (researchComparison.value) {
    if (result.delta === null || result.delta === undefined) return result.status
    return `${result.delta >= 0 ? '+' : ''}${result.delta.toFixed(3)}`
  }
  return `${result.score >= 0 ? '+' : ''}${result.score.toFixed(3)}`
})
const isProcessing = computed(() => {
  const layerIndex = getLayerIndex(props.layer)
  return props.currentProcessingLayer === layerIndex && props.isInferenceRunning
})

const layerModuleCount = computed(() => {
  const preCount = props.layer.preModules?.length || 0
  const postCount = props.layer.postModules?.length || 0
  return preCount + postCount
})

const layerModuleCounts = computed(() => ({
  pre: props.layer.preModules?.length || 0,
  post: props.layer.postModules?.length || 0
}))

const isAttachable = computed(() => props.layer.attachable !== false)
const roleLabel = computed(() => String(props.layer.moduleRole || props.layer.blockType || 'other')
  .replaceAll('_', ' '))
const stageLabel = computed(() => String(props.layer.architectureStage || 'model')
  .replaceAll('_', ' '))
const filterOptionsForLayer = (options) => {
  const role = props.layer.moduleRole
  const architecture = unref(architectureContext) || {}
  const topology = architecture.topology || 'unknown'
  const capabilities = architecture.capabilities || {}
  return (unref(options) || []).filter(option => {
    const targetRoles = option?.targetRoles || []
    const supportedTopologies = option?.supportedTopologies || []
    const requiredCapabilities = option?.requiredCapabilities || []
    const optionalDependencies = option?.optionalDependencies || []
    const roleCompatible = !role || targetRoles.length === 0 || targetRoles.includes(role)
    const topologyCompatible = supportedTopologies.length === 0 ||
      supportedTopologies.includes(topology)
    // Fail closed: a capability the model does not report is not a capability
    // the model has. Treating `undefined` as satisfied let the UI offer modules
    // the backend then refused at run time with a 422.
    const capabilitiesCompatible = requiredCapabilities.every(capability =>
      capabilities[capability] === true
    )
    const dependenciesAvailable = optionalDependencies.every(dependency =>
      dependency?.available !== false
    )
    return roleCompatible && topologyCompatible &&
      capabilitiesCompatible && dependenciesAvailable
  })
}
const incompatibleOptionsForLayer = (options) => {
  const role = props.layer.moduleRole
  const architecture = unref(architectureContext) || {}
  const topology = architecture.topology || 'unknown'
  const capabilities = architecture.capabilities || {}
  return (unref(options) || []).flatMap(option => {
    const targetRoles = option?.targetRoles || []
    if (role && targetRoles.length > 0 && !targetRoles.includes(role)) return []
    const reasons = []
    const supportedTopologies = option?.supportedTopologies || []
    if (supportedTopologies.length > 0 && !supportedTopologies.includes(topology)) {
      reasons.push(`supports ${supportedTopologies.join(', ')}`)
    }
    const missingCapabilities = (option?.requiredCapabilities || [])
      .filter(capability => capabilities[capability] !== true)
    if (missingCapabilities.length > 0) {
      reasons.push(`missing ${missingCapabilities.join(', ')}`)
    }
    const missingPackages = (option?.optionalDependencies || [])
      .filter(dependency => dependency?.available === false)
      .map(dependency => dependency.package)
    if (missingPackages.length > 0) {
      reasons.push(`install ${missingPackages.join(', ')}`)
    }
    return reasons.length > 0
      ? [{ name: option.text || option.value, reason: reasons.join('; ') }]
      : []
  })
}
const compatiblePreModuleOptions = computed(() => filterOptionsForLayer(preModuleOptions))
const compatiblePostModuleOptions = computed(() => filterOptionsForLayer(postModuleOptions))
const unavailableModuleOptions = computed(() => [
  ...incompatibleOptionsForLayer(preModuleOptions),
  ...incompatibleOptionsForLayer(postModuleOptions)
])
const hasCompatibleModules = computed(() =>
  compatiblePreModuleOptions.value.length > 0 ||
  compatiblePostModuleOptions.value.length > 0
)

const removeModule = (moduleType, moduleIndex) => {
  const collection = moduleType === 'pre' ? props.layer.preModules : props.layer.postModules
  if (!Array.isArray(collection)) return

  const targetModule = collection[moduleIndex]
  if (!targetModule?.id) return

  state.blueprints.removeModuleFromLayer(
    state.blueprints.loadedBlueprint,
    props.layer,
    moduleType,
    targetModule.id
  )
}
</script>

<template>
  <div
    class="layer-card"
    :class="{
      'selected': isSelected,
      'expanded': isExpanded,
      'processing': isProcessing,
      'has-modules': layerModuleCount > 0,
      'inference-active': isInferenceRunning,
      'not-attachable': !isAttachable,
      'research-candidate': Boolean(researchResult),
      'research-comparison': Boolean(researchComparison),
      'research-causal': Boolean(researchCausal),
      'research-causal-positive': researchCausal?.delta_from_baseline > 0,
      'research-causal-negative': researchCausal?.delta_from_baseline < 0,
      [`research-${researchComparison?.status || 'candidate'}`]: Boolean(researchResult),
      'research-selected': isResearchSelected
    }"
    :style="researchCausal ? { '--causal-intensity': researchCausal.intensity } : null"
    :data-layer-name="layer.name"
    :data-layer-id="layer.id"
    @contextmenu.prevent="showContextMenu($event, layer.name, getLayerIndex(layer), layer.id)"
  >
    <div
      class="layer-header"
      role="button"
      tabindex="0"
      :aria-expanded="isExpanded"
      :title="isExpanded ? 'Click to collapse' : 'Click to expand'"
      @click="handleLayerHeaderClick($event, layer.name)"
      @keydown.enter.prevent="handleLayerHeaderClick($event, layer.name)"
      @keydown.space.prevent="handleLayerHeaderClick($event, layer.name)"
    >
      <div class="layer-header-left">
        <div class="layer-badge-container">
          <VaBadge
            :text="getLayerTypeInfo(layer).badge"
            :color="getLayerTypeInfo(layer).color"
            class="layer-type-badge"
          />
          <span class="layer-icon">{{ getLayerTypeInfo(layer).icon }}</span>
        </div>
        <div class="layer-title-container">
          <LayerBreadcrumb
            v-if="layer.breadcrumb && layer.breadcrumb.length > 0"
            :breadcrumb="layer.breadcrumb"
            @filter="filterByBreadcrumb"
          />
          <h3 v-else class="layer-title">{{ layer.name }}</h3>
          <p class="layer-subtitle">{{ layer.type }}</p>
          <div class="architecture-tags">
            <span class="architecture-tag role-tag">{{ roleLabel }}</span>
            <span class="architecture-tag">{{ stageLabel }}</span>
            <span v-if="layer.layerIndex !== null && layer.layerIndex !== undefined" class="architecture-tag">
              block {{ layer.layerIndex }}
            </span>
            <span v-if="layer.attentionKind" class="architecture-tag">
              {{ layer.attentionKind }} attention
            </span>
            <span class="architecture-tag" :class="isAttachable ? 'attachable-tag' : 'structural-tag'">
              {{ isAttachable ? 'attachable' : 'structure only' }}
            </span>
          </div>
        </div>

        <div class="layer-indicators" @click.stop>
          <div v-if="isProcessing" class="processing-indicator">
            <div class="processing-dot"></div>
            <span class="processing-text">Processing</span>
          </div>
          <div v-if="layerModuleCounts.pre > 0" class="module-indicator pre-module">
            <VaIcon name="input" size="small" />
            <span class="module-count">{{ layerModuleCounts.pre }}</span>
            <span class="module-type">PRE</span>
          </div>
          <div v-if="layerModuleCounts.post > 0" class="module-indicator post-module">
            <VaIcon name="output" size="small" />
            <span class="module-count">{{ layerModuleCounts.post }}</span>
            <span class="module-type">POST</span>
          </div>
        </div>
      </div>

      <div class="layer-header-right" @click.stop>
        <button
          v-if="researchResult"
          type="button"
          class="research-candidate-score"
          :title="researchCausal ? 'Causal effect from the paired intervention scan' : researchComparison ? `Research comparison: ${researchComparison.status}` : 'Select this Research Study candidate'"
          @click.stop="research.selectLayer(layer.name)"
        >
          <VaIcon name="analytics" size="small" />
          {{ researchResultLabel }}
        </button>
        <div class="layer-quick-stats">
          <span v-if="layer.parameters" class="param-count">{{ formatNumber(layer.parameters) }}</span>
          <span v-if="layer.numHeads" class="heads-count">{{ layer.numHeads }}H</span>
        </div>
        <div class="expand-collapse-hint">
          <span class="hint-text">{{ isExpanded ? 'Collapse' : 'Expand' }}</span>
        </div>
        <VaButton
          preset="plain"
          size="small"
          :icon="isExpanded ? 'expand_less' : 'expand_more'"
          @click.stop="layerState.toggleExpanded(layer.name)"
        />
      </div>
    </div>

    <div v-if="isExpanded" class="layer-details">
      <div class="layer-info-grid">
        <div v-if="layer.inputShape" class="info-item">
          <span class="info-label">Input Shape:</span>
          <span class="info-value">{{ layer.inputShape.join(' × ') }}</span>
        </div>
        <div v-if="layer.outputShape" class="info-item">
          <span class="info-label">Output Shape:</span>
          <span class="info-value">{{ layer.outputShape.join(' × ') }}</span>
        </div>
        <div v-if="layer.parameters" class="info-item">
          <span class="info-label">Parameters:</span>
          <span class="info-value">{{ layer.parameters.toLocaleString() }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Architecture role:</span>
          <span class="info-value">{{ roleLabel }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Stage:</span>
          <span class="info-value">{{ stageLabel }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Module attachment:</span>
          <span class="info-value">{{ isAttachable ? 'Supported' : 'Not an interception target' }}</span>
        </div>

        <template v-if="layer.inspector_info">
          <div v-if="layer.inspector_info.input_numel" class="info-item">
            <span class="info-label">Input Elements:</span>
            <span class="info-value">{{ layer.inspector_info.input_numel.toLocaleString() }}</span>
          </div>
          <div v-if="layer.inspector_info.output_numel" class="info-item">
            <span class="info-label">Output Elements:</span>
            <span class="info-value">{{ layer.inspector_info.output_numel.toLocaleString() }}</span>
          </div>
          <div v-if="layer.inspector_info.input_memory_mb" class="info-item">
            <span class="info-label">Input Memory:</span>
            <span class="info-value memory-badge">{{ formatMemory(layer.inspector_info.input_memory_mb) }}</span>
          </div>
          <div v-if="layer.inspector_info.output_memory_mb" class="info-item">
            <span class="info-label">Output Memory:</span>
            <span class="info-value memory-badge">{{ formatMemory(layer.inspector_info.output_memory_mb) }}</span>
          </div>
        </template>

        <template v-if="layer.type?.toLowerCase().includes('attention')">
          <div v-if="layer.numHeads" class="info-item">
            <span class="info-label">Attention Heads:</span>
            <span class="info-value">{{ layer.numHeads }}</span>
          </div>
          <div v-if="layer.attentionType" class="info-item">
            <span class="info-label">Attention Type:</span>
            <span class="info-value">{{ layer.attentionType }}</span>
          </div>
        </template>

        <template v-if="layer.type?.toLowerCase().includes('transformer')">
          <div v-if="layer.blockType" class="info-item">
            <span class="info-label">Block Type:</span>
            <span class="info-value">{{ layer.blockType }}</span>
          </div>
          <div v-if="layer.hasResidual" class="info-item">
            <span class="info-label">Residual:</span>
            <span class="info-value">✓ Yes</span>
          </div>
        </template>
      </div>

      <div v-if="isAttachable" class="module-controls">
        <div v-if="state.modules.moduleList.length === 0 && !state.modules.isLoading" class="no-modules-message">
          <VaAlert color="info" border="left" size="small">
            No modules available. Load modules from the Modules menu.
          </VaAlert>
        </div>
        <div v-if="!hasCompatibleModules && state.modules.moduleList.length > 0" class="no-modules-message">
          <VaAlert color="warning" border="left" size="small">
            Installed modules do not declare support for the
            <strong>{{ roleLabel }}</strong> role. Choose a compatible architecture node.
          </VaAlert>
        </div>
        <VaAlert
          v-if="unavailableModuleOptions.length > 0"
          color="info"
          border="left"
          size="small"
          class="compatibility-alert"
        >
          <strong>Unavailable here:</strong>
          <span
            v-for="item in unavailableModuleOptions"
            :key="`${item.name}:${item.reason}`"
            class="compatibility-reason"
          >
            {{ item.name }} ({{ item.reason }})
          </span>
        </VaAlert>
        <div class="control-group control-pre">
          <label class="control-label">Add Pre-layer Module:</label>
          <VaSelect
            v-model="layerState.preModuleSelection[layer.name]"
            :options="compatiblePreModuleOptions"
            text-by="text"
            value-by="value"
            track-by="value"
            placeholder="Select module..."
            clearable
            @update:modelValue="(value) => { if (value) { state.blueprints.addPreModuleToLayer(state.blueprints.loadedBlueprint, state.modules.moduleList, layer, value); layerState.preModuleSelection[layer.name] = ''; } }"
            class="module-select"
            style="z-index: 9999 !important;"
          />
        </div>
        <div class="control-group control-post">
          <label class="control-label">Add Post-layer Module:</label>
          <VaSelect
            v-model="layerState.postModuleSelection[layer.name]"
            :options="compatiblePostModuleOptions"
            text-by="text"
            value-by="value"
            track-by="value"
            placeholder="Select module..."
            clearable
            @update:modelValue="(value) => { if (value) { state.blueprints.addPostModuleToLayer(state.blueprints.loadedBlueprint, state.modules.moduleList, layer, value); layerState.postModuleSelection[layer.name] = ''; } }"
            class="module-select"
            style="z-index: 9999 !important;"
          />
        </div>
      </div>
      <VaAlert v-else color="warning" border="left" size="small" class="attachment-alert">
        This node is shown to explain the architecture, but it is not a safe hook target.
        Attach analysis modules to a block, attention, MLP, MoE, normalization, embedding,
        or activation node instead.
      </VaAlert>

      <div class="modules-section">
        <div class="modules-header">
          <h4>Attached Modules</h4>
        </div>

        <div v-if="layer.preModules && layer.preModules.length > 0" class="module-group">
          <div class="module-group-header">
            <VaBadge text="PRE" color="info" class="module-type-badge" />
            <span class="module-group-title">Pre-layer Modules</span>
          </div>
          <div class="module-list">
            <Module
              v-for="(preModule, idx) in layer.preModules"
              :key="preModule.id || preModule.name + '_' + idx"
              :module-attachment="preModule"
              :layer="layer"
              position="pre"
              @remove="removeModule('pre', idx)"
            />
          </div>
        </div>

        <div v-if="layer.postModules && layer.postModules.length > 0" class="module-group">
          <div class="module-group-header">
            <VaBadge text="POST" color="warning" class="module-type-badge" />
            <span class="module-group-title">Post-layer Modules</span>
          </div>
          <div class="module-list">
            <Module
              v-for="(postModule, idx) in layer.postModules"
              :key="postModule.id || postModule.name + '_' + idx"
              :module-attachment="postModule"
              :layer="layer"
              position="post"
              @remove="removeModule('post', idx)"
            />
          </div>
        </div>

        <div v-if="(!layer.preModules || layer.preModules.length === 0) && (!layer.postModules || layer.postModules.length === 0)" class="no-modules-attached">
          <VaAlert color="secondary" border="left" size="small">
            No modules attached to this layer. Use the controls above to add modules.
          </VaAlert>
        </div>

        <div class="layer-notes">
          <div class="layer-notes-header">
            <VaIcon name="sticky_note_2" size="small" />
            <span>Layer Notes</span>
          </div>
          <div class="layer-notes-content">
            <p class="notes-line">
              <strong>{{ layerModuleCount }}</strong> attached modules ·
              <span v-if="layer.parameters">{{ formatNumber(layer.parameters) }} params</span>
              <span v-else>— params</span>
            </p>
            <p class="notes-line">Use pre/post modules to analyze this layer.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
@import '../../../styles/blueprint-layers.css';
@import '../../../styles/blueprint-modules.css';

.architecture-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin-top: 0.3rem;
}

.architecture-tag {
  background: var(--va-background-secondary);
  border: 1px solid var(--va-background-border);
  border-radius: 999px;
  color: var(--va-text-secondary);
  font-size: 0.68rem;
  line-height: 1.2;
  padding: 0.15rem 0.4rem;
  text-transform: capitalize;
}

.role-tag,
.attachable-tag {
  border-color: color-mix(in srgb, var(--va-primary) 35%, transparent);
  color: var(--va-primary);
}

.structural-tag {
  color: var(--va-warning);
}

.not-attachable {
  border-style: dashed;
  opacity: 0.92;
}

.attachment-alert {
  margin: 0.75rem 0;
}

.compatibility-alert {
  margin-bottom: 0.75rem;
}

.compatibility-reason {
  display: block;
  margin-top: 0.2rem;
}

.layer-card.research-candidate {
  background: color-mix(in srgb, var(--va-primary) 5%, var(--va-background-element));
}

.layer-card.research-selected {
  border-color: var(--va-primary);
  box-shadow: inset 0 0 0 1px var(--va-primary);
}

.layer-card.research-comparison {
  background: color-mix(in srgb, var(--va-primary) 3%, var(--va-background-element));
}

.layer-card.research-causal {
  background: color-mix(
    in srgb,
    var(--causal-color, var(--va-primary)) calc(5% + var(--causal-intensity, 0) * 17%),
    var(--va-background-element)
  );
  border-color: color-mix(in srgb, var(--causal-color, var(--va-primary)) 55%, var(--va-background-border));
}

.layer-card.research-causal-positive { --causal-color: var(--va-success); }
.layer-card.research-causal-negative { --causal-color: var(--va-danger); }

.layer-card.research-appeared .research-candidate-score {
  color: var(--va-success);
}

.layer-card.research-disappeared .research-candidate-score {
  color: var(--va-danger);
}

.research-candidate-score {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  border: 1px solid var(--va-background-border);
  border-radius: 2px;
  background: var(--va-background-secondary);
  color: var(--va-primary);
  padding: 0.25rem 0.4rem;
  cursor: pointer;
  font-size: 0.7rem;
}
</style>
