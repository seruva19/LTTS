<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useLttsState } from '../../state'
import { useSettingsStore } from '../../state/settings-store'
import { useToast } from '@/composables/useToast'
import { useColors } from 'vuestic-ui'
import { getCurrentTheme } from '@/theme'
import { createDefaultSettings, deepMerge, MODULE_REGISTRY_URL } from '@/state/settings-defaults'

const state = useLttsState()
const settingsStore = useSettingsStore()
const toast = useToast()
const { applyPreset } = useColors()

const applySelectedTheme = (theme) => {
  if (theme === 'light' || theme === 'dark') {
    applyPreset(theme)
    return
  }
  applyPreset(getCurrentTheme())
}

// Editable settings state
const settings = ref(createDefaultSettings())

// Original settings to track changes
const originalSettings = ref({})

// Dirty state tracking
const isDirty = computed(() => {
  return JSON.stringify(settings.value) !== JSON.stringify(originalSettings.value)
})

// Confirmation dialog
const showConfirmDialog = ref(false)

const saveSettings = async () => {
  try {
    settingsStore.settings = JSON.parse(JSON.stringify(settings.value))
    settingsStore.isDirty = true

    await settingsStore.saveSettings(true)
    originalSettings.value = JSON.parse(JSON.stringify(settings.value))
    applySelectedTheme(settings.value.ui?.theme)
    toast.success('Settings saved')
    state.ui.showSettingsPanel = false
  } catch (error) {
    toast.error('Error saving settings: ' + (error?.message || String(error)))
  }
}

const loadSettings = async () => {
  try {
    // Don't reload settings from server, just use the current settings from the store
    // Ensure we have a complete settings structure by merging with defaults
    const storeSettings = settingsStore.settings || {}
    const defaultStructure = createDefaultSettings()

    // Deep merge to ensure all properties exist
    settings.value = deepMerge(defaultStructure, storeSettings)
    originalSettings.value = JSON.parse(JSON.stringify(settings.value))
  } catch (error) {
    console.warn('Could not load settings:', error)
    // Keep the existing default settings structure
    originalSettings.value = JSON.parse(JSON.stringify(settings.value))
  }
}

const resetToDefaults = async () => {
  try {
    await settingsStore.resetSettings()
    settings.value = JSON.parse(JSON.stringify(settingsStore.settings))
    originalSettings.value = JSON.parse(JSON.stringify(settings.value))
    applySelectedTheme(settings.value.ui?.theme)
    toast.success('Settings reset to defaults')
  } catch (error) {
    toast.error('Error resetting settings: ' + (error?.message || String(error)))
  }
}

const discardChanges = () => {
  settings.value = JSON.parse(JSON.stringify(originalSettings.value))
  showConfirmDialog.value = false
  state.ui.showSettingsPanel = false
}

const attemptClose = () => {
  if (isDirty.value) {
    showConfirmDialog.value = true
  } else {
    state.ui.showSettingsPanel = false
  }
}

// Watch for modal close attempts
watch(() => state.ui.showSettingsPanel, (newVal, oldVal) => {
  if (oldVal && !newVal && isDirty.value) {
    // Prevent closing if dirty
    state.ui.showSettingsPanel = true
    showConfirmDialog.value = true
  }
})

onMounted(() => {
  loadSettings()
})
</script>

<template>
  <VaModal
    class="settings-modal"
    v-model="state.ui.showSettingsPanel"
    size="large"
    hide-default-actions
    no-padding
    :before-close="attemptClose"
  >
    <div class="modal-header">
      <h3 class="modal-title">Settings</h3>
    </div>

    <div class="settings-content">
      <!-- Every control here is read by the application. Options that only
           round-tripped through storage were removed rather than left to look
           functional; their keys still exist in the settings schema. -->
      <div class="settings-section">
        <h4>User Interface</h4>
        <div class="settings-grid">
          <div class="setting-item">
            <label>Theme</label>
            <VaSelect
              v-model="settings.ui.theme"
              :options="[
                { text: 'Auto', value: 'auto' },
                { text: 'Light', value: 'light' },
                { text: 'Dark', value: 'dark' }
              ]"
              text-by="text"
              value-by="value"
            />
          </div>
          <div class="setting-item">
            <VaCheckbox
              v-model="settings.controls.auto_scroll_to_target"
              label="Auto Scroll to Target"
            />
          </div>
        </div>
      </div>

      <div class="settings-section">
        <h4>Modules</h4>
        <div class="settings-grid">
          <div class="setting-item full-width">
            <label>Module Registry URL</label>
            <VaInput
              v-model="settings.modules.registry_url"
              :placeholder="MODULE_REGISTRY_URL"
            />
            <small class="setting-hint">
              Leave empty to resolve automatically: LTTS_MODULES_REGISTRY, then a
              sibling LTTS-modules checkout, then the published registry.
            </small>
          </div>
          <div class="setting-item">
            <label>Request Timeout (ms)</label>
            <VaInput
              v-model="settings.api.timeout"
              type="number"
              min="1000"
              max="300000"
            />
          </div>
        </div>
      </div>

      <div class="settings-section">
        <h4>Storage</h4>
        <div class="settings-grid">
          <div class="setting-item">
            <label>Save Location</label>
            <VaSelect
              v-model="settings.storage.save_location"
              :options="[
                { text: 'Local Storage (Browser)', value: 'localStorage' },
                { text: 'Disk (Server)', value: 'disk' }
              ]"
              text-by="text"
              value-by="value"
            />
          </div>
          <div class="setting-item">
            <VaCheckbox
              v-model="settings.storage.auto_save_settings"
              label="Auto Save Settings"
            />
          </div>
          <div class="setting-item">
            <VaCheckbox
              v-model="settings.blueprint.load_last_blueprint_on_launch"
              label="Load Last Blueprint on Launch"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Warning message for unsaved changes -->
    <div v-if="isDirty" class="warning-bar">
      <VaIcon name="warning" size="small" />
      <span>You have unsaved changes. Please save or discard your changes before closing.</span>
    </div>

    <div class="modal-footer">
      <VaButton preset="secondary" @click="resetToDefaults">Reset to Defaults</VaButton>
      <VaButton preset="secondary" @click="attemptClose">Cancel</VaButton>
      <VaButton preset="primary" @click="saveSettings" :class="{ 'unsaved-changes': isDirty }">
        Save Settings{{ isDirty ? ' *' : '' }}
      </VaButton>
    </div>
  </VaModal>

  <!-- Confirmation Dialog -->
  <VaModal
    v-model="showConfirmDialog"
    size="small"
    hide-default-actions
    no-padding
    class="confirm-modal"
  >
    <div class="confirm-header">
      <h4>Unsaved Changes</h4>
    </div>
    <div class="confirm-content">
      <p>You have unsaved changes. What would you like to do?</p>
    </div>
    <div class="confirm-footer">
      <VaButton preset="secondary" @click="showConfirmDialog = false">Continue Editing</VaButton>
      <VaButton preset="secondary" @click="discardChanges">Discard Changes</VaButton>
      <VaButton preset="primary" @click="saveSettings">Save & Close</VaButton>
    </div>
  </VaModal>
</template>

<style scoped>
.settings-modal {
  z-index: 10000 !important;
}

.settings-modal :deep(.va-modal) {
  max-width: 800px;
  max-height: 90vh;
  border-radius: 12px;
  overflow: hidden;
}

.modal-header {
  background: var(--va-background-element);
  border-bottom: 1px solid var(--va-background-border);
  padding: 1.5rem;
}

.modal-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--va-text-primary);
}

.settings-content {
  padding: 1.5rem;
  max-height: 60vh;
  overflow-y: auto;
  background: var(--va-background-primary);
}

.setting-hint {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  opacity: 0.7;
  line-height: 1.35;
}

.settings-section {
  margin-bottom: 2rem;
}

.settings-section:last-child {
  margin-bottom: 0;
}

.settings-section h4 {
  margin: 0 0 1rem 0;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--va-primary);
  border-bottom: 2px solid var(--va-primary);
  padding-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.setting-item label {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--va-text-primary);
}

.setting-item.full-width {
  grid-column: 1 / -1;
}

.modal-footer {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  padding: 1.5rem;
  border-top: 1px solid var(--va-background-border);
  background: var(--va-background-element);
}

.modal-footer .va-button {
  min-width: 100px;
}

.unsaved-changes {
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* Confirmation Modal Styles */
.confirm-modal {
  z-index: 10001 !important;
}

.confirm-header {
  background: var(--va-background-element);
  border-bottom: 1px solid var(--va-background-border);
  padding: 1rem 1.5rem;
}

.confirm-header h4 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--va-text-primary);
}

.confirm-content {
  padding: 1.5rem;
  background: var(--va-background-primary);
}

.confirm-content p {
  margin: 0;
  color: var(--va-text-primary);
  font-size: 1rem;
  line-height: 1.5;
}

.confirm-footer {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--va-background-border);
  background: var(--va-background-element);
}

.confirm-footer .va-button {
  min-width: 120px;
}

/* Warning Bar Styles */
.warning-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  background: var(--va-warning);
  color: var(--va-on-warning);
  border-top: 1px solid var(--va-background-border);
  font-size: 0.9rem;
  font-weight: 500;
}

.warning-bar .va-icon {
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .settings-grid {
    grid-template-columns: 1fr;
  }

  .settings-content {
    padding: 1rem;
  }

  .modal-footer {
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem;
  }

  .confirm-footer {
    flex-direction: column;
    gap: 0.75rem;
  }

  .confirm-footer .va-button {
    min-width: auto;
  }
}
</style>
