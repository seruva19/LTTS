<template>
  <div class="model-level-modules">
    <div class="model-modules-header">
      <h3>Model-Level Modules</h3>
      <VaButton
        icon="add"
        size="small"
        @click="showAddModule = true"
      >
        Add Model Module
      </VaButton>
    </div>

    <div v-if="modelModules.length > 0" class="model-modules-list">
      <div
        v-for="module in modelModules"
        :key="module.id"
        class="model-module-item"
      >
        <div class="module-info">
          <div class="module-name">{{ module.name }}</div>
          <div class="module-description">{{ module.description }}</div>
        </div>

        <div class="module-controls">
          <VaButton
            icon="settings"
            size="small"
            @click="openModuleSettings(module)"
          >
            Settings
          </VaButton>
          <VaButton
            icon="delete"
            size="small"
            color="danger"
            @click="removeModule(module)"
          >
            Remove
          </VaButton>
        </div>
      </div>
    </div>

    <div v-else class="empty-state">
      <div class="empty-icon">🎛️</div>
      <div class="empty-text">
        No model-level modules attached
      </div>
      <div class="empty-description">
        Model-level modules affect the entire model behavior, not specific layers
      </div>
    </div>

    <VaModal
      v-model="showAddModule"
      title="Add Model-Level Module"
      size="medium"
    >
      <div class="add-module-form">
        <VaSelect
          v-model="selectedModule"
          :options="availableModelModules"
          label="Select Module"
          placeholder="Choose a model-level module"
        />

        <VaAlert
          v-if="unavailableModelModules.length > 0"
          color="info"
          border="left"
          size="small"
        >
          <strong>Unavailable here:</strong>
          <span
            v-for="item in unavailableModelModules"
            :key="`${item.name}:${item.reason}`"
            class="compatibility-reason"
          >
            {{ item.name }} ({{ item.reason }})
          </span>
        </VaAlert>

        <div v-if="selectedModule" class="module-preview">
          <h4>{{ selectedModule.text }}</h4>
          <p>{{ selectedModule.description }}</p>
        </div>
      </div>

      <template #footer>
        <VaButton @click="showAddModule = false">Cancel</VaButton>
        <VaButton
          @click="addModelModule"
          :disabled="!selectedModule"
          color="primary"
        >
          Add Module
        </VaButton>
      </template>
    </VaModal>

    <VaModal
      v-model="showModuleSettings"
      :title="`${selectedModuleForSettings?.name || 'Module'} Settings`"
      size="large"
    >
      <div v-if="selectedModuleForSettings" class="module-settings">
        <div v-if="moduleSettingsUI" class="settings-container">
          <div
            v-for="section in moduleSettingsUI.sections"
            :key="section.id"
            class="settings-section"
          >
            <h4>{{ section.title }}</h4>

            <div
              v-for="control in section.controls"
              :key="control.name"
              class="control-item"
            >
              <VaSlider
                v-if="control.type === 'slider'"
                v-model="moduleSettings[control.name]"
                :label="control.label"
                :min="control.min"
                :max="control.max"
                :step="control.step"
                :default="control.default"
              >
                <template #append>
                  <span class="slider-value">{{ moduleSettings[control.name] }}</span>
                </template>
              </VaSlider>

              <VaSwitch
                v-else-if="control.type === 'toggle'"
                v-model="moduleSettings[control.name]"
                :label="control.label"
                :default="control.default"
              />

              <VaTextarea
                v-else-if="control.type === 'textarea'"
                v-model="moduleSettings[control.name]"
                :label="control.label"
                :placeholder="control.placeholder"
                :rows="4"
              />

              <VaSelect
                v-else-if="control.type === 'select'"
                v-model="moduleSettings[control.name]"
                :label="control.label"
                :options="control.options"
                :default="control.default"
              />

              <VaInput
                v-else-if="control.type === 'number'"
                v-model="moduleSettings[control.name]"
                :label="control.label"
                type="number"
                :min="control.min"
                :max="control.max"
                :default="control.default"
              />

              <VaButton
                v-else-if="control.type === 'button'"
                @click="handleModuleAction(control.action, control.name)"
                :color="control.color || 'primary'"
              >
                {{ control.label }}
              </VaButton>

              <div v-if="control.description" class="control-description">
                {{ control.description }}
              </div>
            </div>
          </div>
        </div>

        <div v-else class="no-settings">
          No additional settings available for this module
        </div>
      </div>

      <template #footer>
        <VaButton @click="showModuleSettings = false">Close</VaButton>
        <VaButton
          @click="saveModuleSettings"
          color="primary"
        >
          Save Settings
        </VaButton>
      </template>
    </VaModal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useLttsState } from '@/state'

const state = useLttsState()

const showAddModule = ref(false)
const showModuleSettings = ref(false)
const selectedModule = ref(null)
const selectedModuleForSettings = ref(null)
const moduleSettings = ref({})
const moduleSettingsUI = ref(null)

const modelModules = ref([])

const modelArchitecture = computed(() => ({
  topology: state.blueprints.loadedBlueprint?.architectureTopology || 'unknown',
  capabilities: state.blueprints.loadedBlueprint?.capabilities || {}
}))

const modelModuleCompatibility = (name, module) => {
  const contract = module.attachment_contract || {}
  const reasons = []
  const topologies = contract.supported_topologies || []
  if (
    topologies.length > 0 &&
    !topologies.includes(modelArchitecture.value.topology)
  ) {
    reasons.push(`supports ${topologies.join(', ')}`)
  }
  // Fail closed, like the layer cards: a capability the model does not report
  // is not one it has.
  const missingCapabilities = (contract.required_capabilities || [])
    .filter(capability => modelArchitecture.value.capabilities[capability] !== true)
  if (missingCapabilities.length > 0) {
    reasons.push(`missing ${missingCapabilities.join(', ')}`)
  }
  const missingPackages = (contract.optional_dependencies || [])
    .filter(dependency => dependency?.available === false)
    .map(dependency => dependency.package)
  if (missingPackages.length > 0) {
    reasons.push(`install ${missingPackages.join(', ')}`)
  }
  return {
    name,
    compatible: reasons.length === 0,
    reason: reasons.join('; ')
  }
}

const availableModelModules = computed(() => {
  if (!state.modules.modules) return []

  return Object.entries(state.modules.modules)
    .filter(([, module]) =>
      module.attachment_contract?.model_level ||
      module.metadata?.module_type === 'model_level'
    )
    .filter(([name, module]) => modelModuleCompatibility(name, module).compatible)
    .map(([name, module]) => ({
      text: name,
      value: name,
      description: module.metadata?.description || 'Model-level module'
    }))
})

const unavailableModelModules = computed(() => {
  if (!state.modules.modules) return []
  return Object.entries(state.modules.modules)
    .filter(([, module]) =>
      module.attachment_contract?.model_level ||
      module.metadata?.module_type === 'model_level'
    )
    .map(([name, module]) => modelModuleCompatibility(name, module))
    .filter(item => !item.compatible)
})

const addModelModule = () => {
  if (!selectedModule.value) return

  const newModule = {
    id: Date.now(),
    name: selectedModule.value.value,
    description: selectedModule.value.description,
    settings: {}
  }

  modelModules.value.push(newModule)

  if (state.blueprints.loadedBlueprint) {
    state.blueprints.addModelModule(selectedModule.value.value, {})
  }

  showAddModule.value = false
  selectedModule.value = null
}

const removeModule = (module) => {
  modelModules.value = modelModules.value.filter(m => m.id !== module.id)

  if (state.blueprints.loadedBlueprint) {
    state.blueprints.removeModelModule(module.name)
  }
}

const openModuleSettings = (module) => {
  selectedModuleForSettings.value = module
  moduleSettings.value = { ...module.settings }

  const moduleInfo = state.modules.modules[module.name]
  if (moduleInfo && moduleInfo.ui) {
    moduleSettingsUI.value = moduleInfo.ui
  }

  showModuleSettings.value = true
}

const handleModuleAction = (action, controlName) => {
  if (!selectedModuleForSettings.value) return
  moduleSettings.value = {
    ...moduleSettings.value,
    action,
    _action_name: controlName,
    _action_timestamp: Date.now()
  }
}

const saveModuleSettings = () => {
  if (!selectedModuleForSettings.value) return

  const moduleIndex = modelModules.value.findIndex(m => m.id === selectedModuleForSettings.value.id)
  if (moduleIndex !== -1) {
    modelModules.value[moduleIndex].settings = { ...moduleSettings.value }
  }

  if (state.blueprints.loadedBlueprint) {
    state.blueprints.updateModelModuleSettings(selectedModuleForSettings.value.name, moduleSettings.value)
  }

  showModuleSettings.value = false
}

onMounted(() => {
  if (state.blueprints.loadedBlueprint && state.blueprints.loadedBlueprint.modelModules) {
    modelModules.value = state.blueprints.loadedBlueprint.modelModules.map((module, index) => ({
      id: index + 1,
      name: module.name,
      description: getModelDescription(module.name),
      settings: module.settings || {}
    }))
  }
})

function getModelDescription(moduleName) {
  const moduleInfo = state.modules.modules[moduleName]
  return moduleInfo?.metadata?.description || 'Model-level module'
}
</script>

<style scoped>
.model-level-modules {
  background: var(--va-background);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.model-modules-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.model-modules-header h3 {
  margin: 0;
  color: var(--va-primary);
}

.model-modules-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-module-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: var(--va-background-secondary);
  border-radius: 6px;
  border: 1px solid var(--va-border);
}

.module-info {
  flex: 1;
}

.module-name {
  font-weight: 600;
  color: var(--va-text-primary);
  margin-bottom: 4px;
}

.module-description {
  font-size: 14px;
  color: var(--va-text-secondary);
}

.module-controls {
  display: flex;
  gap: 8px;
}

.empty-state {
  text-align: center;
  padding: 32px;
  color: var(--va-text-secondary);
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.empty-text {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
}

.empty-description {
  font-size: 14px;
}

.add-module-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.module-preview {
  padding: 12px;
  background: var(--va-background-secondary);
  border-radius: 6px;
}

.module-preview h4 {
  margin: 0 0 8px 0;
  color: var(--va-text-primary);
}

.module-preview p {
  margin: 0;
  color: var(--va-text-secondary);
  font-size: 14px;
}

.compatibility-reason {
  display: block;
  margin-top: 4px;
}

.settings-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.settings-section {
  border: 1px solid var(--va-border);
  border-radius: 6px;
  padding: 16px;
}

.settings-section h4 {
  margin: 0 0 16px 0;
  color: var(--va-text-primary);
}

.control-item {
  margin-bottom: 16px;
}

.control-description {
  font-size: 12px;
  color: var(--va-text-secondary);
  margin-top: 4px;
}

.slider-value {
  min-width: 40px;
  text-align: center;
}

.no-settings {
  text-align: center;
  padding: 32px;
  color: var(--va-text-secondary);
}
</style>
