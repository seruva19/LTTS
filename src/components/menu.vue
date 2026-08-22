<script setup>
import { useLttsState } from '../state'
import { useColors } from "vuestic-ui"

import { storeToRefs } from 'pinia'
import { ref, onMounted, onUnmounted, computed } from 'vue'
const state = useLttsState()
const { isConnected, messages, error } = storeToRefs(state.ws)
const { applyPreset, currentPresetName } = useColors()
const switchValue = computed({
  get: () => (currentPresetName.value),
  set: (value) => applyPreset(value)
})

import { useToast } from '@/composables/useToast'

const toast = useToast()

const loadLastBlueprint = () => {
  try {
    const hasBlueprint = state.blueprints.loadBlueprintFromLocalStorage()
    if (hasBlueprint) {
      toast.success('Blueprint loaded from local storage')
    } else {
      toast.info('No saved blueprint found')
    }
  } catch (error) {
    toast.error('Failed to load blueprint')
    console.error('Error loading blueprint:', error)
  }
}

const fastSave = () => {
  if (!state.blueprints.loadedBlueprint) {
    toast.warning('No blueprint loaded to save')
    return
  }

  try {
    state.blueprints.saveBlueprintToLocalStorage()
    toast.success('Blueprint saved to local storage')
  } catch (error) {
    toast.error('Failed to save blueprint')
    console.error('Error saving blueprint:', error)
  }
}

const fastLoad = () => {
  loadLastBlueprint()
}

const alertPlaceholder = () => {
  toast.info('This feature is not implemented yet')
}

// Run and Stop functions
const runBlueprint = () => state.blueprints.runLoadedBlueprint()

const stopBlueprint = () => state.blueprints.stopLoadedBlueprint()

// Modules browser function
const showModulesBrowser = () => {
  // Emit event to open modules browser
  const event = new CustomEvent('open-modules-browser')
  document.dispatchEvent(event)
}

// Scoped module adding helpers
const addInModelModule = () => {
  const event = new CustomEvent('open-modules-browser', {
    detail: { filter: { scope: 'model', eventType: 'model_before' } }
  })
  document.dispatchEvent(event)
}

const addOutModelModule = () => {
  const event = new CustomEvent('open-modules-browser', {
    detail: { filter: { scope: 'model', eventType: 'model_after' } }
  })
  document.dispatchEvent(event)
}

const addInLayerModule = () => {
  // Get the currently selected layer from the UI state
  const selectedLayer = state.ui.selectedLayer

  if (!selectedLayer) {
    toast.warning('Please select a layer first')
    return
  }

  const layerPath = selectedLayer.path || selectedLayer.name || selectedLayer.id

  const event = new CustomEvent('open-modules-browser', {
    detail: {
      filter: {
        scope: 'layer',
        eventType: 'layer_before',
        layerPath: layerPath
      }
    }
  })
  document.dispatchEvent(event)
}

const addOutLayerModule = () => {
  // Get the currently selected layer from the UI state
  const selectedLayer = state.ui.selectedLayer

  if (!selectedLayer) {
    toast.warning('Please select a layer first')
    return
  }

  const layerPath = selectedLayer.path || selectedLayer.name || selectedLayer.id

  const event = new CustomEvent('open-modules-browser', {
    detail: {
      filter: {
        scope: 'layer',
        eventType: 'layer_after',
        layerPath: layerPath
      }
    }
  })
  document.dispatchEvent(event)
}

// Menu hover behavior using refs
const menuRefs = ref({})
const isMenuModeActive = ref(false)
const openMenuName = ref(null)

const handleMenuModelUpdate = (menuName, value) => {
  if (value) {
    openMenuName.value = menuName
    isMenuModeActive.value = true
  } else if (openMenuName.value === menuName) {
    openMenuName.value = null
    isMenuModeActive.value = false
  }
}

const closeMenus = () => {
  openMenuName.value = null
  isMenuModeActive.value = false
}

const onMenuSelected = (callback) => {
  try {
    if (typeof callback === 'function') {
      callback()
    }
  } finally {
    closeMenus()
  }
}

const setMenuRef = (menuName, ref) => {
  if (ref) {
    menuRefs.value[menuName] = ref
  }
}

const handleMenuButtonClick = (menuName) => {
  if (openMenuName.value === menuName) {
    openMenuName.value = null
    isMenuModeActive.value = false
    return
  }
  isMenuModeActive.value = true
  openMenuName.value = menuName
}

const handleMenuHover = (menuName) => {
  if (isMenuModeActive.value) {
    if (openMenuName.value !== menuName) {
      openMenuName.value = menuName
    }
  }
}

// Reset menu mode when clicking outside menubar area
const handleGlobalClick = (event) => {
  // Consider the entire top menu bar as inside area
  if (!event.target.closest('.menu') && !event.target.closest('.menu-content')) {
    setTimeout(() => {
      isMenuModeActive.value = false
      openMenuName.value = null
    }, 100)
  }
}

onMounted(() => {
  document.addEventListener('click', handleGlobalClick)
})

onUnmounted(() => {
  document.removeEventListener('click', handleGlobalClick)
})


</script>

<template>
  <div class="menu">
    <div class="menu-bar">
      <VaMenu
        :ref="(el) => setMenuRef('blueprints', el)"
        :model-value="openMenuName === 'blueprints'"
        @update:modelValue="(v) => handleMenuModelUpdate('blueprints', v)"
        :stateful="false"
        trigger="click"
        :close-on-click-outside="false"
        :close-on-focus-outside="false"
        :close-on-anchor-click="false"
        :close-on-content-click="true"
        :is-content-hoverable="true"
        content-class="menu-content blueprints">
        <template #anchor>
          <VaButton
            preset="plain"
            class="menu-root"
            @click="handleMenuButtonClick('blueprints')"
            @mouseenter="handleMenuHover('blueprints')">
            Blueprints
          </VaButton>
        </template>
        <VaMenuItem @selected="onMenuSelected(state.ui.openBlueprintManager)">
          Blueprint manager...
        </VaMenuItem>
        <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(loadLastBlueprint)">
          Load last blueprint
        </VaMenuItem>
        <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(fastSave)">
          Fast save
        </VaMenuItem>
        <VaMenuItem @selected="onMenuSelected(fastLoad)">
          Fast load
        </VaMenuItem>
      </VaMenu>

      <VaMenu
        :ref="(el) => setMenuRef('models', el)"
        :model-value="openMenuName === 'models'"
        @update:modelValue="(v) => handleMenuModelUpdate('models', v)"
        :stateful="false"
        trigger="click"
        :close-on-click-outside="false"
        :close-on-focus-outside="false"
        :close-on-anchor-click="false"
        :close-on-content-click="true"
        :is-content-hoverable="true"
        content-class="menu-content models">
        <template #anchor>
          <VaButton
            preset="plain"
            class="menu-root"
            @click="handleMenuButtonClick('models')"
            @mouseenter="handleMenuHover('models')">
            Models
          </VaButton>
        </template>
        <VaMenuItem @selected="onMenuSelected(state.models.showModels)">
          Load model...
        </VaMenuItem>
      </VaMenu>

      <VaMenu
        :ref="(el) => setMenuRef('modules', el)"
        :model-value="openMenuName === 'modules'"
        @update:modelValue="(v) => handleMenuModelUpdate('modules', v)"
        :stateful="false"
        trigger="click"
        :close-on-click-outside="false"
        :close-on-focus-outside="false"
        :close-on-anchor-click="false"
        :close-on-content-click="true"
        :is-content-hoverable="true"
        content-class="menu-content modules">
        <template #anchor>
          <VaButton
            preset="plain"
            class="menu-root"
            @click="handleMenuButtonClick('modules')"
            @mouseenter="handleMenuHover('modules')">
            Modules
          </VaButton>
        </template>
        <VaMenuItem @selected="onMenuSelected(state.modules.showModules)">
          Module manager...
        </VaMenuItem>
        <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(showModulesBrowser)">
          Explore modules...
        </VaMenuItem>
        <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(addInModelModule)">
          Add in-model module...
        </VaMenuItem>
        <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(addOutModelModule)">
           Add out-model module...
         </VaMenuItem>
         <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(addInLayerModule)">
           Add in-layer module...
         </VaMenuItem>
        <VaMenuItem @selected="onMenuSelected(addOutLayerModule)">
           Add out-layer module...
         </VaMenuItem>
      </VaMenu>

      <VaMenu
        :ref="(el) => setMenuRef('run', el)"
        :model-value="openMenuName === 'run'"
        @update:modelValue="(v) => handleMenuModelUpdate('run', v)"
        :stateful="false"
        trigger="click"
        :close-on-click-outside="false"
        :close-on-focus-outside="false"
        :close-on-anchor-click="false"
        :close-on-content-click="true"
        :is-content-hoverable="true"
        content-class="menu-content run">
        <template #anchor>
          <VaButton
            preset="plain"
            class="menu-root"
            @click="handleMenuButtonClick('run')"
            @mouseenter="handleMenuHover('run')">
            Run
          </VaButton>
        </template>
        <VaMenuItem @selected="onMenuSelected(runBlueprint)">
          Run
        </VaMenuItem>
        <VaDivider class="menu-divider" />
        <VaMenuItem @selected="onMenuSelected(stopBlueprint)">
          Stop
        </VaMenuItem>
      </VaMenu>

      <VaMenu
        :ref="(el) => setMenuRef('view', el)"
        :model-value="openMenuName === 'view'"
        @update:modelValue="(v) => handleMenuModelUpdate('view', v)"
        :stateful="false"
        trigger="click"
        :close-on-click-outside="false"
        :close-on-focus-outside="false"
        :close-on-anchor-click="false"
        :close-on-content-click="true"
        :is-content-hoverable="true"
        content-class="menu-content view">
        <template #anchor>
          <VaButton
            preset="plain"
            class="menu-root"
            @click="handleMenuButtonClick('view')"
            @mouseenter="handleMenuHover('view')">
            View
          </VaButton>
        </template>
        <VaMenuItem @selected="() => onMenuSelected(() => applyPreset(switchValue == 'dark' ? 'light' : 'dark'))">
          Switch theme
        </VaMenuItem>
      </VaMenu>

      <VaMenu
        :ref="(el) => setMenuRef('settings', el)"
        :model-value="openMenuName === 'settings'"
        @update:modelValue="(v) => handleMenuModelUpdate('settings', v)"
        :stateful="false"
        trigger="click"
        :close-on-click-outside="false"
        :close-on-focus-outside="false"
        :close-on-anchor-click="false"
        :close-on-content-click="true"
        :is-content-hoverable="true"
        content-class="menu-content settings">
        <template #anchor>
          <VaButton
            preset="plain"
            class="menu-root"
            @click="handleMenuButtonClick('settings')"
            @mouseenter="handleMenuHover('settings')">
            Settings
          </VaButton>
        </template>
        <VaMenuItem @selected="() => onMenuSelected(() => state.ui.showSettings())">
          Settings...
        </VaMenuItem>
      </VaMenu>

      <div class="status" :class="{ connected: isConnected }">
        {{ isConnected ? 'connected' : 'disconnected' }}
      </div>
    </div>

    <div class="color-mode-switch">
      <VaSwitch v-model="switchValue" true-value="dark" false-value="light" size="small"  true-inner-label="dark" false-inner-label="light">
      </VaSwitch>
    </div>
  </div>
</template>

<style scoped>
  .menu {
    height: 40px;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background:  var(--va-background-border);
  }

  .menu-bar {
    width: calc(100% - 100px);
  }

  .menu-divider {
    margin: 5px 0;
  }

  .menu-root {
    color: var(--va-text-primary) !important;
    background: none !important;
    margin: 5px;
    font-weight: normal;
    padding: 5px;
    border-radius: 0;
  }

  .menu-root:hover {
    background-color: var(--va-focus) !important;
    color: var(--va-text-inverted) !important;
  }

  .status {
    float: right;
    color: var(--va-danger);
    font-size: 12px;
    font-family: monospace;
    margin-top: 14px;
  }

  .status.connected {
    color: var(--va-success);
  }

  .color-mode-switch {
    position: absolute;
    right: 5px;
    top: 5px;
    z-index: 1000;
    background-color:  var(--va-background-border);
    padding: 3px;
  }
</style>
