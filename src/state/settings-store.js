import { defineStore } from 'pinia'
import { ref, computed, watch, nextTick } from 'vue'
import axios from 'axios'
import { baseUrl } from '../constants'
import { useToast } from '@/composables/useToast'
import { createDefaultSettings, deepMerge } from './settings-defaults'

function normalizeSelectableValue(value, fallback = null) {
  if (typeof value === 'string') {
    return value
  }
  if (value && typeof value === 'object' && typeof value.value === 'string') {
    return value.value
  }
  return fallback
}

function normalizeSettingsShape(input = {}) {
  const normalized = { ...(input || {}) }

  if (normalized.blueprints && !normalized.blueprint) {
    normalized.blueprint = { ...normalized.blueprints }
  }

  normalized.blueprint = {
    ...(normalized.blueprint || {}),
    ...(normalized.blueprints || {})
  }

  normalized.ui = {
    ...(normalized.ui || {}),
    theme: normalizeSelectableValue(normalized.ui?.theme, normalized.ui?.theme ?? 'auto')
  }

  normalized.visualization = {
    ...(normalized.visualization || {}),
    default_attention_view: normalizeSelectableValue(
      normalized.visualization?.default_attention_view,
      normalized.visualization?.default_attention_view ?? 'heatmap'
    ),
    color_scheme: normalizeSelectableValue(
      normalized.visualization?.color_scheme,
      normalized.visualization?.color_scheme ?? 'viridis'
    )
  }

  normalized.storage = {
    ...(normalized.storage || {}),
    save_location: normalizeSelectableValue(
      normalized.storage?.save_location,
      normalized.storage?.save_location ?? 'localStorage'
    )
  }

  normalized.blueprint = {
    ...(normalized.blueprint || {}),
    save_location: normalizeSelectableValue(
      normalized.blueprint?.save_location,
      normalized.blueprint?.save_location ?? 'localStorage'
    )
  }

  normalized.controls = {
    ...(normalized.controls || {})
  }
  if (
    normalized.controls.auto_scroll_to_target === undefined &&
    normalized.controls.auto_scroll_to_selection !== undefined
  ) {
    normalized.controls.auto_scroll_to_target = normalized.controls.auto_scroll_to_selection
  }

  normalized.modules = {
    registry_url: 'https://raw.githubusercontent.com/seruva19/LTTS-modules/main/registry.json',
    ...(normalized.modules || {})
  }

  return normalized
}

export const useSettingsStore = defineStore('settings', () => {
  // Initialize with basic structure to prevent undefined errors
  // Shape placeholder until loadSettings() runs; see settings-defaults.js.
  const settings = ref(createDefaultSettings())
  const defaultSettings = ref({})
  const isLoading = ref(false)
  const error = ref(null)
  const isDirty = ref(false)
  const toast = useToast()

  // Storage location preference
  const storageLocation = computed(() => {
    return settings.value?.storage?.save_location || 'localStorage'
  })

  // Auto-save preference
  const autoSave = computed(() => {
    return settings.value?.storage?.auto_save_settings !== false
  })

  // Load last blueprint on launch preference
  const loadLastBlueprintOnLaunch = computed(() => {
    return settings.value?.blueprint?.load_last_blueprint_on_launch === true
  })

  // Deep merge function for settings
  // Load default settings
  async function loadDefaultSettings() {
    try {
      const response = await axios.get(`${baseUrl}/api/settings/defaults`)
      if (response.data.status === 'success') {
        defaultSettings.value = normalizeSettingsShape(response.data.settings)
        return defaultSettings.value
      } else {
        throw new Error(response.data.message || 'Failed to load default settings')
      }
    } catch (err) {
      error.value = `Failed to load default settings: ${err.message}`
      console.error('Error loading default settings:', err)
      // Return hardcoded fallback defaults
      const fallbackDefaults = createDefaultSettings()
      defaultSettings.value = normalizeSettingsShape(fallbackDefaults)
      return defaultSettings.value
    }
  }

  // Load settings from server or localStorage
  let suppressDirty = false

  function readPersistedSettings() {
    try {
      const raw = localStorage.getItem('ltts-settings')
      return raw ? JSON.parse(raw) : null
    } catch (e) {
      console.warn('Failed to parse localStorage settings', e)
      return null
    }
  }

  // Assign settings without the deep watcher flagging the change as a user
  // edit. The flag is cleared again on nextTick, after the watcher has flushed.
  function applyWithoutDirtying(value) {
    suppressDirty = true
    settings.value = value
    isDirty.value = false
    nextTick(() => {
      suppressDirty = false
      isDirty.value = false
    })
  }

  async function loadSettings(showToast = true) {
    isLoading.value = true
    error.value = null

    try {
      // First load defaults
      await loadDefaultSettings()

      // Try to load from localStorage first if that's the preferred storage.
      // `storageLocation` is derived from `settings`, which is only assigned at
      // the end of this function - so read the persisted preference directly
      // instead of the still-initial in-memory value.
      let userSettings = {}
      const persisted = readPersistedSettings()
      const preferredStorage =
        persisted?.storage?.save_location ??
        defaultSettings.value?.storage?.save_location ??
        storageLocation.value

      if (preferredStorage === 'localStorage' && persisted) {
        userSettings = persisted
      }

      // If no localStorage settings or disk storage preferred, load from server
      if (Object.keys(userSettings).length === 0 || preferredStorage === 'disk') {
        const response = await axios.get(`${baseUrl}/api/settings`)
        if (response.data.status === 'success') {
          userSettings = response.data.settings
        }
      }

      // Merge default and user settings
      applyWithoutDirtying(deepMerge(
        normalizeSettingsShape(defaultSettings.value),
        normalizeSettingsShape(userSettings)
      ))

      if (showToast) {
        toast.success('Settings loaded')
      }
    } catch (err) {
      error.value = `Failed to load settings: ${err.message}`
      if (showToast) {
        toast.error(`Failed to load settings: ${err.message}`)
      }
      // Fall back to defaults
      applyWithoutDirtying(normalizeSettingsShape({ ...defaultSettings.value }))
    } finally {
      isLoading.value = false
    }
  }

  // Save settings
  async function saveSettings(showToast = true) {
    if (!isDirty.value) {
      if (showToast) toast.info('No changes to save')
      return
    }

    isLoading.value = true
    error.value = null

    try {
      // Save to localStorage if preferred
      if (storageLocation.value === 'localStorage') {
        localStorage.setItem('ltts-settings', JSON.stringify(normalizeSettingsShape(settings.value)))
      }

      // Always save to server as backup
      const response = await axios.post(`${baseUrl}/api/settings`, {
        settings: normalizeSettingsShape(settings.value),
        storage_location: storageLocation.value
      })

      if (response.data.status === 'success') {
        isDirty.value = false
        if (showToast) toast.success('Settings saved')
      } else {
        throw new Error(response.data.message || 'Failed to save settings')
      }
    } catch (err) {
      error.value = `Failed to save settings: ${err.message}`
      if (showToast) toast.error(`Failed to save settings: ${err.message}`)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  // Update a specific setting
  function updateSetting(path, value) {
    const keys = path.split('.')
    let current = settings.value

    // Navigate to the parent object
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {}
      }
      current = current[keys[i]]
    }

    // Set the value
    current[keys[keys.length - 1]] = value
    isDirty.value = true

    // Auto-save if enabled
    if (autoSave.value) {
      saveSettings(false) // Don't show toast for auto-save
    }
  }

  // Get a specific setting
  function getSetting(path, defaultValue = null) {
    const keys = path.split('.')
    let current = settings.value

    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key]
      } else {
        return defaultValue
      }
    }

    return current
  }

  // Reset settings to defaults
  async function resetSettings() {
    try {
      await loadDefaultSettings()
      settings.value = { ...defaultSettings.value }
      isDirty.value = true

      // Clear localStorage
      localStorage.removeItem('ltts-settings')

      // Save the reset settings
      await saveSettings()

      toast.success('Settings reset to defaults')
    } catch (err) {
      error.value = `Failed to reset settings: ${err.message}`
      toast.error(`Failed to reset settings: ${err.message}`)
    }
  }

  // Export settings
  function exportSettings() {
    const dataStr = JSON.stringify(settings.value, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'ltts-settings.json'
    link.click()
    URL.revokeObjectURL(url)
    toast.success('Settings exported')
  }

  // Import settings
  async function importSettings(file) {
    try {
      const text = await file.text()
      const importedSettings = JSON.parse(text)

      // Validate imported settings by merging with defaults
      settings.value = deepMerge(defaultSettings.value, importedSettings)
      isDirty.value = true

      await saveSettings()
      toast.success('Settings imported')
    } catch (err) {
      error.value = `Failed to import settings: ${err.message}`
      toast.error(`Failed to import settings: ${err.message}`)
    }
  }

  // Watch for changes and mark as dirty. Programmatic loads go through
  // applyWithoutDirtying(), which suppresses this - assigning `settings` and
  // then clearing `isDirty` synchronously does not work, because the deep
  // watcher flushes afterwards and re-sets the flag.
  watch(settings, () => {
    if (suppressDirty) return
    isDirty.value = true
  }, { deep: true })

  return {
    settings,
    defaultSettings,
    isLoading,
    error,
    isDirty,
    storageLocation,
    autoSave,
    loadLastBlueprintOnLaunch,

    loadSettings,
    saveSettings,
    updateSetting,
    getSetting,
    resetSettings,
    exportSettings,
    importSettings,
    loadDefaultSettings
  }
})
