import { defineStore } from 'pinia'
import { useConfigStore } from './config-store'
import { useModelStore } from './model-store'
import { useModuleStore } from './module-store'
import { useBlueprintStore } from './blueprint-store'
import { useUIStore } from './ui-store'
import { useWebSocketStore } from './socket-store'
import { useSettingsStore } from './settings-store'
import { useResearchStore } from './research-store'

export const useLttsState = defineStore('lttsState', () => {
  const configStore = useConfigStore()
  const modelStore = useModelStore()
  const moduleStore = useModuleStore()
  const blueprintStore = useBlueprintStore()
  const uiStore = useUIStore()
  const wsStore = useWebSocketStore();
  const settingsStore = useSettingsStore()
  const researchStore = useResearchStore()

  return {
    config: configStore,
    models: modelStore,
    modules: moduleStore,
    blueprints: blueprintStore,
    ui: uiStore,
    ws: wsStore,
    settings: settingsStore,
    research: researchStore,
  }
})
