import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useConfigStore } from './state/config-store'
import { useModelStore } from './state/model-store'
import { useModuleStore } from './state/module-store'
import { useBlueprintStore } from './state/blueprint-store'
import { useWebSocketStore } from './state/socket-store'
import { useSettingsStore } from './state/settings-store'

export const useLauncher = defineStore('launcher', () => {
  const isInitializing = ref(false)
  const error = ref(null)

  async function initializeApp() {
    const configStore = useConfigStore()
    const modelStore = useModelStore()
    const moduleStore = useModuleStore()
    const blueprintStore = useBlueprintStore()
    const wsStore = useWebSocketStore()
    const settingsStore = useSettingsStore()

    isInitializing.value = true
    error.value = null

    try {
      // Load settings first as they control other initialization behavior
      // Use true for showToast parameter to show the notification during initial load
      await settingsStore.loadSettings(true)

      // Check if we should load last blueprint on launch
      const shouldLoadLastBlueprint = settingsStore.loadLastBlueprintOnLaunch

      let blueprintLoaded = false

      if (shouldLoadLastBlueprint) {
        // Try to load blueprint from local storage or last saved blueprint
        blueprintLoaded = blueprintStore.loadBlueprintFromLocalStorage()
      }

      const results = await Promise.all([
        configStore.loadConfig(),
        modelStore.fetchModels(),
        moduleStore.fetchModules(),
        blueprintStore.fetchBlueprints()
      ])

      // Initialize WebSocket connection after other stores are loaded
      wsStore.connect()

      return true
    } catch (err) {
      error.value = `Failed to initialize application: ${err.message}`
      console.error('Application initialization failed:', err)
      throw err
    } finally {
      isInitializing.value = false
    }
  }

  return {
    isInitializing,
    error,
    initialize: initializeApp
  }
})