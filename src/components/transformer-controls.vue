<template>
  <aside class="transformer-controls-sidebar" :class="{ expanded: !isCollapsed }" :style="{ width: `${isCollapsed ? 50 : panelWidth}px`, '--sidebar-width': `${isCollapsed ? 50 : panelWidth}px` }" aria-label="Model navigation and visibility">
    <div class="sidebar-resize-handle" @mousedown="startResize" />
    <div class="sidebar-header">
      <button class="sidebar-toggle" :title="isCollapsed ? 'Expand navigation panel' : 'Collapse navigation panel'" @click="toggleSidebar">
        <VaIcon :name="isCollapsed ? 'chevron_right' : 'chevron_left'" />
      </button>
      <div v-if="!isCollapsed"><strong>Model explorer</strong><small>Navigate and filter layers</small></div>
    </div>
    <!-- Compressed content -->
    <ResearchStudyPanel v-if="research.isOpen && !isCollapsed" @close="closeResearchStudy" />
    <div v-else class="compressed-content">
      <!-- Layer Categories Section -->
      <div class="layer-categories-section">
        <div v-if="!isCollapsed" class="section-caption">Navigate</div>
        <div v-if="navigationItems.length === 0" class="no-categories">
          <VaIcon name="category" size="1rem" color="secondary" />
        </div>

        <div v-else class="categories-list">
          <!-- Jump to Input - Only show when model is loaded -->
          <div
            v-if="state.blueprints.loadedBlueprint"
            class="category-item jump-button"
            role="button"
            tabindex="0"
            title="Jump to Input"
            @click="jumpToInput"
            @keydown.enter.prevent="jumpToInput"
            @keydown.space.prevent="jumpToInput"
          >
            <div class="category-icon">
              <span class="icon">⬆️</span>
            </div>
            <span v-if="!isCollapsed" class="jump-label">Model input</span>
          </div>

          <div
            v-for="item in navigationItems"
            :key="item.id"
            class="category-item"
            :class="{ 'selected': selectedCategory === item.id }"
            @click="scrollToCategory(item.id)"
            @mouseenter="selectedCategory = item.id"
            :title="`${item.name} (${item.count} layers)`"
            role="button"
            tabindex="0"
            @keydown.enter.prevent="scrollToCategory(item.id)"
            @keydown.space.prevent="scrollToCategory(item.id)"
          >
            <div class="category-icon">
              <span class="icon">{{ item.icon }}</span>
            </div>
            <div v-if="!isCollapsed" class="category-copy"><strong>{{ item.name }}</strong><small>{{ item.count }} layers</small></div>
          </div>

          <!-- Jump to Output - Only show when model is loaded -->
          <div
            v-if="state.blueprints.loadedBlueprint"
            class="category-item jump-button"
            role="button"
            tabindex="0"
            title="Jump to Output"
            @click="jumpToOutput"
            @keydown.enter.prevent="jumpToOutput"
            @keydown.space.prevent="jumpToOutput"
          >
            <div class="category-icon">
              <span class="icon">⬇️</span>
            </div>
            <span v-if="!isCollapsed" class="jump-label">Model output</span>
          </div>
        </div>
      </div>

      <!-- Component Controls Section -->
      <div class="component-section">
        <div v-if="!isCollapsed" class="section-caption">Section visibility</div>
        <div class="component-icons">
          <div
            v-for="component in transformerComponents"
            :key="component.id"
            class="component-icon"
            :class="{ active: component.enabled, disabled: !component.enabled }"
            @click="toggleComponent(component)"
            :title="component.name"
            role="button"
            tabindex="0"
            @keydown.enter.prevent="toggleComponent(component)"
            @keydown.space.prevent="toggleComponent(component)"
          >
            <VaIcon :name="component.icon" size="small" />
            <span v-if="!isCollapsed">{{ component.name }}</span>
            <span v-if="!isCollapsed" class="visibility-state">{{ component.enabled ? 'Shown' : 'Hidden' }}</span>
          </div>
        </div>
      </div>

      <!-- Preset Controls Section -->
      <div class="preset-section">
        <div v-if="!isCollapsed" class="section-caption">Tools</div>
        <div class="preset-icons">
          <div
            v-for="preset in presets"
            :key="preset.id"
            class="preset-icon"
            @click="applyPreset(preset.id)"
            :title="preset.name"
            role="button"
            tabindex="0"
            @keydown.enter.prevent="applyPreset(preset.id)"
            @keydown.space.prevent="applyPreset(preset.id)"
          >
            <VaIcon :name="preset.icon" size="small" />
            <span v-if="!isCollapsed">{{ preset.name }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Search Bar Overlay -->
    <div v-if="showSearchBar" class="search-overlay">
      <div class="search-container">
        <div class="search-header">
          <VaIcon name="search" size="small" />
          <span>Search Layers</span>
          <VaIcon name="close" size="small" @click="toggleSearch" class="close-btn" />
        </div>
        <VaInput
          v-model="searchQuery"
          placeholder="Search by name, type, or module..."
          class="search-input"
          @input="performSearch"
        />
        <div v-if="searchResults.length > 0" class="search-results">
          <div
            v-for="result in searchResults"
            :key="result.id"
            class="search-result-item"
            @click="jumpToSearchResult(result)"
            role="button"
            tabindex="0"
            @keydown.enter.prevent="jumpToSearchResult(result)"
            @keydown.space.prevent="jumpToSearchResult(result)"
          >
            <div class="result-name">{{ result.name || result.type }}</div>
            <div class="result-details">{{ result.module || 'No module' }}</div>
          </div>
        </div>
        <div v-else-if="searchQuery.trim()" class="no-results">
          No layers found matching "{{ searchQuery }}"
        </div>
      </div>
    </div>

    <!-- Bookmarks Panel -->
    <div v-if="showBookmarks" class="bookmarks-overlay">
      <div class="bookmarks-container">
        <div class="bookmarks-header">
          <VaIcon name="bookmark" size="small" />
          <span>Bookmarks</span>
          <VaIcon name="close" size="small" @click="toggleBookmarks" class="close-btn" />
        </div>
        <div v-if="bookmarks.length > 0" class="bookmarks-list">
          <div
            v-for="bookmark in bookmarks"
            :key="bookmark.id"
            class="bookmark-item"
          >
            <div
              class="bookmark-content"
              role="button"
              tabindex="0"
              @click="jumpToBookmark(bookmark)"
              @keydown.enter.prevent="jumpToBookmark(bookmark)"
              @keydown.space.prevent="jumpToBookmark(bookmark)"
            >
              <div class="bookmark-name">{{ bookmark.name }}</div>
              <div class="bookmark-time">{{ new Date(bookmark.timestamp).toLocaleTimeString() }}</div>
            </div>
            <VaIcon
              name="delete"
              size="small"
              @click="removeBookmark(bookmark.id)"
              class="delete-btn"
            />
          </div>
        </div>
        <div v-else class="no-bookmarks">
          No bookmarks saved yet. Right-click on layers to add bookmarks.
        </div>
      </div>
    </div>

    <!-- Module Filter Panel -->
    <div v-if="showModuleFilter" class="module-filter-overlay">
      <div class="module-filter-container">
        <div class="module-filter-header">
          <VaIcon name="filter_list" size="small" />
          <span>Filter by Module</span>
          <VaIcon name="close" size="small" @click="toggleModuleFilter" class="close-btn" />
        </div>
        <div class="module-filter-options">
          <div
            class="module-option"
            :class="{ active: selectedModuleFilter === 'all' }"
            @click="applyModuleFilter('all')"
            role="button"
            tabindex="0"
            @keydown.enter.prevent="applyModuleFilter('all')"
            @keydown.space.prevent="applyModuleFilter('all')"
          >
            <VaIcon name="layers" size="small" />
            <span>All Layers</span>
          </div>
          <div
            v-for="module in availableModules"
            :key="module"
            class="module-option"
            :class="{ active: selectedModuleFilter === module }"
            @click="applyModuleFilter(module)"
            role="button"
            tabindex="0"
            @keydown.enter.prevent="applyModuleFilter(module)"
            @keydown.space.prevent="applyModuleFilter(module)"
          >
            <VaIcon name="extension" size="small" />
            <span>{{ module }}</span>
          </div>
        </div>
        <div v-if="availableModules.length === 0" class="no-modules">
          No modules found in current blueprint.
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useLttsState } from '../state'
import { scrollToLayerIndex } from '@/composables/useLayerNavigation'
import ResearchStudyPanel from '@/components/research-study-panel.vue'
import { useResearchStore } from '@/state/research-store'

const state = useLttsState()
const research = useResearchStore()

// Track the current layer index for index-based keyboard navigation.
const currentLayerIndex = ref(-1)

// Resolve a layer's index in the ordered blueprint layers array.
const resolveLayerIndex = ({ id, name } = {}) => {
  const orderedLayers = state.blueprints.loadedBlueprint?.layers || []
  if (name != null) {
    const byName = orderedLayers.findIndex(l => l.name === name)
    if (byName >= 0) return byName
  }
  if (id != null) {
    return orderedLayers.findIndex(l => String(l.id) === String(id))
  }
  return -1
}

// Sidebar state
const isCollapsed = ref(true) // Start collapsed (just a bar)
const panelWidth = ref(250)
let resizing = false

const emitPanelSize = () => {
  document.dispatchEvent(new CustomEvent('controls-panel-resize', {
    detail: { width: isCollapsed.value ? 50 : panelWidth.value }
  }))
}
const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
  emitPanelSize()
}
const closeResearchStudy = () => {
  research.close()
  isCollapsed.value = true
  emitPanelSize()
}
const startResize = (event) => {
  if (isCollapsed.value) return
  resizing = true
  event.preventDefault()
}
const resizeSidebar = (event) => {
  if (!resizing) return
  panelWidth.value = Math.min(420, Math.max(220, event.clientX))
  emitPanelSize()
}
const stopResize = () => { resizing = false }

// Search functionality
const showSearchBar = ref(false)
const searchQuery = ref('')
const searchResults = ref([])

// Bookmarks system
const bookmarks = ref([])
const showBookmarks = ref(false)

// Module filter
const showModuleFilter = ref(false)
const selectedModuleFilter = ref('all')

// Layer categories functionality
// Map category IDs to emoji icons
const categoryIconMap = {
  'embeddings': '🔤',
  'input_processing': '📥',
  'attention_mechanisms': '👁️',
  'transformer_blocks': '🔄',
  'feed_forward': '⚡',
  'normalization': '⚖️',
  'activation_functions': '🔥',
  'moe_layers': '🧠',
  'output_processing': '📤',
  // Fallbacks for any other categories
  'other': '🔹',
  'unknown': '❓'
}

const navigationItems = computed(() => {
  if (!state.blueprints.loadedBlueprint?.categorization?.navigation) {
    return []
  }

  // Get navigation items and ensure they have emoji icons
  const items = [...state.blueprints.loadedBlueprint.categorization.navigation]

  // Apply emoji icons based on category ID
  return items.map(item => {
    // Create a new object to avoid mutating the original
    return {
      ...item,
      // Use the mapped emoji if available, otherwise use the original icon or a fallback
      icon: categoryIconMap[item.id] || item.icon || categoryIconMap['unknown']
    }
  })
})

const categories = computed(() => {
  if (!state.blueprints.loadedBlueprint?.categorization?.categories) {
    return {}
  }
  return state.blueprints.loadedBlueprint.categorization.categories
})

const architecture = computed(() => {
  return state.blueprints.loadedBlueprint?.categorization?.architecture || 'Unknown'
})

const totalLayers = computed(() => {
  return state.blueprints.loadedBlueprint?.categorization?.totalLayers || 0
})

const selectedCategory = ref(null)

// Auto-select active category based on scroll position
let categoryObserver = null
const getScrollableRoot = () => document.querySelector('.layers-container')
const setupCategoryObserver = () => {
  const root = getScrollableRoot()
  const targets = document.querySelectorAll('.category-section')
  if (!root || targets.length === 0) return
  categoryObserver = new IntersectionObserver((entries) => {
    const visible = entries.filter(e => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
    if (visible?.target?.dataset?.category) {
      selectedCategory.value = visible.target.dataset.category
    }
  }, { root, threshold: [0.3, 0.5, 0.7] })
  targets.forEach(t => categoryObserver.observe(t))
}
const teardownCategoryObserver = () => { if (categoryObserver) { categoryObserver.disconnect(); categoryObserver = null } }

const scrollToCategory = (categoryId) => {
  const element = document.querySelector(`[data-category="${categoryId}"]`)
  if (element) {
    element.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    })
    selectedCategory.value = categoryId
  }
}

const formatNumber = (num) => {
  if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B'
  if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M'
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K'
  return num?.toLocaleString() || '0'
}

// Performance controls
const skipDisabledLayers = ref(false)
const enableGradientCheckpointing = ref(false)
const enableMixedPrecision = ref(false)

// Component controls used by the Component Controls Section
const transformerComponents = ref([
  { id: 'embedding', name: 'Embedding', icon: 'text_fields', enabled: true },
  { id: 'attention', name: 'Attention', icon: 'visibility', enabled: true },
  { id: 'transformer_blocks', name: 'Transformer Blocks', icon: 'view_stream', enabled: true },
  { id: 'feed_forward', name: 'Feed Forward', icon: 'bolt', enabled: true },
  { id: 'normalization', name: 'Normalization', icon: 'tune', enabled: true },
  { id: 'output', name: 'Output', icon: 'outbox', enabled: true }
])

const toggleComponent = (component) => {
  component.enabled = !component.enabled
  applyComponentChanges()
}

// Navigation presets
const presets = ref([
  { id: 'research-study', name: 'Research Study', icon: 'science' },
  { id: 'search', name: 'Search Layers', icon: 'search' },
  { id: 'bookmarks', name: 'Bookmarks', icon: 'bookmark' },
  { id: 'filter-modules', name: 'Filter by Modules', icon: 'filter_list' },
  { id: 'expand-all', name: 'Expand All Timeline', icon: 'unfold_more' },
  { id: 'collapse-all', name: 'Collapse All Timeline', icon: 'unfold_less' }
])

// Search functionality methods
const toggleSearch = () => {
  showSearchBar.value = !showSearchBar.value
  if (showSearchBar.value) {
    showBookmarks.value = false
    showModuleFilter.value = false
    // Focus search input after DOM update
    setTimeout(() => {
      const searchInput = document.querySelector('.search-input')
      if (searchInput) searchInput.focus()
    }, 100)
  } else {
    searchQuery.value = ''
    searchResults.value = []
  }
}

const performSearch = () => {
  if (!searchQuery.value.trim()) {
    searchResults.value = []
    return
  }

  const layers = state.blueprints.loadedBlueprint?.layers || []
  searchResults.value = layers.filter(layer =>
    layer.name?.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
    layer.type?.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
    layer.module?.toLowerCase().includes(searchQuery.value.toLowerCase())
  ).slice(0, 10) // Limit to 10 results
}

const jumpToSearchResult = (layer) => {
  const index = resolveLayerIndex({ id: layer.id, name: layer.name })
  if (index >= 0) {
    currentLayerIndex.value = index
    scrollToLayerIndex(index)
    showSearchBar.value = false
    searchQuery.value = ''
    searchResults.value = []
  }
}

// Bookmarks functionality
const toggleBookmarks = () => {
  showBookmarks.value = !showBookmarks.value
  if (showBookmarks.value) {
    showSearchBar.value = false
    showModuleFilter.value = false
  }
}

const addBookmark = (layerId, layerName) => {
  const existingBookmark = bookmarks.value.find(b => b.id === layerId)
  if (!existingBookmark) {
    bookmarks.value.push({
      id: layerId,
      name: layerName,
      timestamp: Date.now()
    })
    // Save to localStorage
    localStorage.setItem('ltts-bookmarks', JSON.stringify(bookmarks.value))
  }
}

const removeBookmark = (layerId) => {
  bookmarks.value = bookmarks.value.filter(b => b.id !== layerId)
  saveBookmarksToLocalStorage()
}

const jumpToBookmark = (bookmark) => {
  const index = resolveLayerIndex({ id: bookmark.id, name: bookmark.layerName || bookmark.name })
  if (index >= 0) {
    currentLayerIndex.value = index
    scrollToLayerIndex(index)
    showBookmarks.value = false
  }
}

// Module filter functionality
const toggleModuleFilter = () => {
  showModuleFilter.value = !showModuleFilter.value
  if (showModuleFilter.value) {
    showSearchBar.value = false
    showBookmarks.value = false
  }
}

const availableModules = computed(() => {
  const layers = state.blueprints.loadedBlueprint?.layers || []
  const modules = new Set()
  layers.forEach(layer => {
    ;[...(layer.preModules || []), ...(layer.postModules || [])].forEach((module) => {
      const name = typeof module === 'string'
        ? module
        : module?.name || module?.module?.name || module?.metadata?.name || null
      if (name) {
        modules.add(name)
      }
    })
  })
  ;(state.blueprints.loadedBlueprint?.modelModules || []).forEach((module) => {
    const name = typeof module === 'string'
      ? module
      : module?.name || module?.module?.name || module?.metadata?.name || null
    if (name) {
      modules.add(name)
    }
  })
  return Array.from(modules)
})

const applyModuleFilter = (moduleId) => {
  selectedModuleFilter.value = moduleId
  // Dispatch event to filter layers
  const event = new CustomEvent('layer-module-filter', {
    detail: { moduleId }
  })
  document.dispatchEvent(event)
  showModuleFilter.value = false
}

const jumpToInput = () => {
  const inputElement = document.querySelector('.io-card.input')
  if (inputElement) {
    inputElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

const jumpToOutput = () => {
  const outputElement = document.querySelector('.io-card.output')
  if (outputElement) {
    outputElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

const applyPreset = (presetId) => {

  switch (presetId) {
    case 'research-study':
      research.open()
      if (isCollapsed.value) {
        isCollapsed.value = false
        emitPanelSize()
      }
      break
    case 'search':
      toggleSearch()
      break
    case 'bookmarks':
      toggleBookmarks()
      break
    case 'filter-modules':
      toggleModuleFilter()
      break
    case 'expand-all':
      expandAllTimelineGroups()
      break
    case 'collapse-all':
      collapseAllTimelineGroups()
      break
    case 'full':
      enableAll()
      break
    case 'attention-only':
      transformerComponents.value.forEach(component => {
        component.enabled = component.id === 'attention' || component.id === 'embedding' || component.id === 'output'
      })
      break
    case 'minimal':
      transformerComponents.value.forEach(component => {
        component.enabled = component.id === 'embedding' || component.id === 'output'
      })
      break
    case 'inference':
      enableAll()
      skipDisabledLayers.value = true
      enableMixedPrecision.value = true
      break
  }

  applyComponentChanges()
}

// Expand all timeline groups
const expandAllTimelineGroups = () => {
  // Dispatch custom event to timeline component
  const event = new CustomEvent('expand-all-timeline-groups', {
    detail: { expand: true }
  })
  document.dispatchEvent(event)
}

// Collapse all timeline groups
const collapseAllTimelineGroups = () => {
  // Dispatch custom event to timeline component
  const event = new CustomEvent('expand-all-timeline-groups', {
    detail: { expand: false }
  })
  document.dispatchEvent(event)
}

const enableAll = () => {
  transformerComponents.value.forEach(component => {
    component.enabled = true
  })
}

const applyComponentChanges = () => {
  // Build visibility map for blueprint sections
  const comp = transformerComponents.value.reduce((acc, c) => {
    acc[c.id] = c.enabled
    return acc
  }, {})

  const visibility = {
    embeddings: comp.embedding !== false,
    attention_mechanisms: comp.attention !== false,
    transformer_blocks: comp.transformer_blocks !== false,
    feed_forward: comp.feed_forward !== false,
    normalization: comp.normalization !== false,
    output_processing: comp.output !== false
  }

  const event = new CustomEvent('section-visibility-update', {
    detail: { visibility }
  })
  document.dispatchEvent(event)
}

// Listen for bookmark events from layer cards
const handleAddBookmark = (event) => {
  const { layerName, layerIndex, layerId } = event.detail
  if (!bookmarks.value.some(b => b.layerName === layerName)) {
    const newBookmark = {
      id: layerId || layerName,
      layerName,
      layerIndex,
      timestamp: new Date().toISOString()
    }
    bookmarks.value.push(newBookmark)
    saveBookmarksToLocalStorage()
  }
}

const handleRemoveBookmark = (event) => {
  const { layerName } = event.detail
  bookmarks.value = bookmarks.value.filter(b => b.layerName !== layerName)
  saveBookmarksToLocalStorage()
}

const saveBookmarksToLocalStorage = () => {
    localStorage.setItem('ltts-bookmarks', JSON.stringify(bookmarks.value))
    // Dispatch bookmark update event
    const event = new CustomEvent('bookmark-update', {
      detail: { bookmarkedLayers: bookmarks.value.map(b => b.layerName) }
    })
    document.dispatchEvent(event)
  }

// Keep scroll position in sync with selected layer
watch(() => state.ui?.selectedLayerId, (id) => {
  if (!id) return
  const index = resolveLayerIndex({ id, name: state.ui?.selectedLayerName })
  if (index >= 0) {
    currentLayerIndex.value = index
    scrollToLayerIndex(index)
  }
})

// Visual pulse on programmatic jumps
const pulseElement = (el) => {
  if (!el) return
  el.classList.add('jump-highlight')
  setTimeout(() => el.classList.remove('jump-highlight'), 950)
}

// Keyboard shortcuts and helpers
const navigateCategory = (delta) => {
  const ids = navigationItems.value.map(i => i.id)
  if (ids.length === 0) return
  let idx = ids.indexOf(selectedCategory.value)
  if (idx < 0) idx = 0
  idx = Math.min(ids.length - 1, Math.max(0, idx + delta))
  scrollToCategory(ids[idx])
}

const navigateLayer = (delta) => {
  // Step through the ordered blueprint layers by a tracked current index, so
  // navigation works even when target cards are unmounted by windowing.
  const orderedLayers = state.blueprints.loadedBlueprint?.layers || []
  if (orderedLayers.length === 0) return

  // Sync the tracked index with the current global selection if available.
  let current = currentLayerIndex.value
  if (current < 0) {
    current = resolveLayerIndex({
      id: state.ui?.selectedLayerId,
      name: state.ui?.selectedLayerName
    })
  }
  if (current < 0) current = delta > 0 ? -1 : orderedLayers.length

  const next = Math.min(orderedLayers.length - 1, Math.max(0, current + delta))
  currentLayerIndex.value = next
  scrollToLayerIndex(next)

  // Update global selection so other components stay in sync
  const layer = orderedLayers[next]
  try {
    if (state.ui?.setSelectedLayer && layer) {
      state.ui.setSelectedLayer(layer)
    }
  } catch {}
}

const handleKeydown = (e) => {
  const tag = (document.activeElement?.tagName || '').toLowerCase()
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return
  if (e.key === '/') { e.preventDefault(); toggleSearch(); return }
  if (e.key === 'b') { toggleBookmarks(); return }
  if (e.key === 'f') { toggleModuleFilter(); return }
  if (e.key === 'ArrowDown') { navigateLayer(1); return }
  if (e.key === 'ArrowUp') { navigateLayer(-1); return }
  if (e.key === 'PageDown') { navigateCategory(1); return }
  if (e.key === 'PageUp') { navigateCategory(-1); return }
  if (e.key === '[') { navigateCategory(-1); return }
  if (e.key === ']') { navigateCategory(1); return }
  if (e.key === 'j') { navigateLayer(1); return }
  if (e.key === 'k') { navigateLayer(-1); return }
}

// Load bookmarks from localStorage on mount
onMounted(() => {
const savedBookmarks = localStorage.getItem('ltts-bookmarks')
if (savedBookmarks) {
  try {
    bookmarks.value = JSON.parse(savedBookmarks)
  } catch (e) {
    console.warn('Failed to load bookmarks:', e)
  }
}
// Initial bookmark update dispatch
const event = new CustomEvent('bookmark-update', {
  detail: { bookmarkedLayers: bookmarks.value.map(b => b.layerName) }
})
document.dispatchEvent(event)

document.addEventListener('add-bookmark', handleAddBookmark)
document.addEventListener('remove-bookmark', handleRemoveBookmark)
// Initialize observer and keyboard shortcuts
setupCategoryObserver()
window.addEventListener('keydown', handleKeydown)
document.addEventListener('keydown', handleKeydown)
window.addEventListener('mousemove', resizeSidebar)
window.addEventListener('mouseup', stopResize)
emitPanelSize()
})

onUnmounted(() => {
document.removeEventListener('add-bookmark', handleAddBookmark)
document.removeEventListener('remove-bookmark', handleRemoveBookmark)
teardownCategoryObserver()
window.removeEventListener('keydown', handleKeydown)
document.removeEventListener('keydown', handleKeydown)
window.removeEventListener('mousemove', resizeSidebar)
window.removeEventListener('mouseup', stopResize)
})

</script>

<style scoped>
/* Keyboard focus must be visible on the clickable containers below. */
[role='button']:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.transformer-controls-sidebar {
  width: 50px; /* Very narrow width */
  background: var(--va-background-element);
  border-right: 1px solid var(--va-background-border);
  height: calc(100vh - 40px);
  overflow: hidden;
  position: fixed;
  top: 40px;
  left: 0;
  z-index: 1001;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
}

.compressed-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 0.5rem 0;
}

.layer-categories-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-bottom: 1px solid var(--va-background-border);
  padding-bottom: 0.5rem;
  margin-bottom: 0.5rem;
}

.no-categories {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem 0;
  color: var(--va-text-secondary);
}

.categories-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  padding: 0 0.25rem;
}

.category-item {
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border-radius: 4px;
  padding: 0.25rem;
}

.category-item:hover {
  /* No background change on hover - same as other buttons */
}

.category-item.selected {
  /* No background change when selected - same as other buttons */
}

.category-icon {
  width: 28px;
  height: 28px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.category-icon .icon {
  font-size: 0.875rem;
  color: #606060;
}

/* Jump Controls Styles */
.jump-controls {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--va-background-border);
}

.jump-button:focus-visible,
.bookmark-content:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.jump-button {
  cursor: pointer;
}

.jump-button:hover .category-icon {
  background: var(--va-background-element-hover, #f0f0f0);
}

.component-section {
  border-bottom: 1px solid var(--va-background-border);
  padding-bottom: 0.5rem;
  margin-bottom: 0.5rem;
}

.component-icons {
  padding: 0 0.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  align-items: center;
}

.component-icon {
  width: 32px;
  height: 32px;
  background: color-mix(in srgb, var(--va-primary) 5%, var(--va-background-element));
  border: 1px solid transparent;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: var(--va-primary);
  transition: background-color .15s ease, border-color .15s ease, color .15s ease;
}

.component-icon:hover {
  background: color-mix(in srgb, var(--va-primary) 12%, var(--va-background-element));
  border-color: color-mix(in srgb, var(--va-primary) 22%, transparent);
  color: var(--va-primary);
}

.component-icon.active {
  background: color-mix(in srgb, var(--va-primary) 9%, var(--va-background-element));
  border-color: color-mix(in srgb, var(--va-primary) 18%, transparent);
  color: var(--va-primary);
}

.component-icon.disabled {
  background: var(--va-background-secondary);
  border-color: transparent;
  color: var(--va-text-secondary);
  opacity: 0.6;
}

.preset-section {
  /* No border for last section */
}

.preset-icons {
  padding: 0 0.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  align-items: center;
}

.preset-icon {
  width: 32px;
  height: 32px;
  background: color-mix(in srgb, var(--va-primary) 5%, var(--va-background-element));
  border: 1px solid transparent;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: var(--va-primary);
  transition: background-color .15s ease, border-color .15s ease, color .15s ease;
}

.preset-icon:hover {
  background: color-mix(in srgb, var(--va-primary) 12%, var(--va-background-element));
  border-color: color-mix(in srgb, var(--va-primary) 22%, transparent);
  color: var(--va-primary);
}

.preset-icon.jump-button .category-icon {
  font-size: 14px;
}

/* Overlay Styles */
.search-overlay,
.bookmarks-overlay,
.module-filter-overlay {
  position: fixed;
  left: 50px;
  top: 40px;
  width: 300px;
  max-height: calc(100vh - 80px);
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1002;
  overflow: hidden;
}

.search-container,
.bookmarks-container,
.module-filter-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.search-header,
.bookmarks-header,
.module-filter-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: var(--va-background-secondary);
  border-bottom: 1px solid var(--va-background-border);
  font-weight: 500;
}

.close-btn {
  margin-left: auto;
  cursor: pointer;
  color: var(--va-text-secondary);
}

.close-btn:hover {
  color: var(--va-text-primary);
}

/* Search Styles */
.search-input {
  margin: 0.75rem;
  margin-bottom: 0;
}

.search-results {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.search-result-item {
  padding: 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  border-bottom: 1px solid var(--va-background-border);
}

.search-result-item:hover {
  background: var(--va-background-secondary);
}

.result-name {
  font-weight: 500;
  color: var(--va-text-primary);
}

.result-details {
  font-size: 0.75rem;
  color: var(--va-text-secondary);
  margin-top: 0.25rem;
}

.no-results {
  padding: 1rem;
  text-align: center;
  color: var(--va-text-secondary);
  font-style: italic;
}

/* Bookmarks Styles */
.bookmarks-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.bookmark-item {
  display: flex;
  align-items: center;
  padding: 0.5rem;
  border-radius: 4px;
  border-bottom: 1px solid var(--va-background-border);
}

.bookmark-item:hover {
  background: var(--va-background-secondary);
}

.bookmark-content {
  flex: 1;
  cursor: pointer;
}

.bookmark-name {
  font-weight: 500;
  color: var(--va-text-primary);
}

.bookmark-time {
  font-size: 0.75rem;
  color: var(--va-text-secondary);
  margin-top: 0.25rem;
}

.delete-btn {
  cursor: pointer;
  color: var(--va-text-secondary);
  margin-left: 0.5rem;
}

.delete-btn:hover {
  color: var(--va-danger);
}

.no-bookmarks {
  padding: 1rem;
  text-align: center;
  color: var(--va-text-secondary);
  font-style: italic;
}

/* Module Filter Styles */
.module-filter-options {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.module-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  border-bottom: 1px solid var(--va-background-border);
}

.module-option:hover {
  background: var(--va-background-secondary);
}

.module-option.active {
  background: var(--va-primary);
  color: white;
}

.no-modules {
  padding: 1rem;
  text-align: center;
  color: var(--va-text-secondary);
  font-style: italic;
}

/* Scrollbar styling */
.transformer-controls-sidebar::-webkit-scrollbar {
  width: 2px;
}

.transformer-controls-sidebar::-webkit-scrollbar-track {
  background: rgba(248, 250, 252, 0.5);
}

.transformer-controls-sidebar::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.4);
  border-radius: 1px;
}

.transformer-controls-sidebar::-webkit-scrollbar-thumb:hover {
  background: rgba(100, 116, 139, 0.6);
}

/* Expandable navigation and explicit visibility controls */
.transformer-controls-sidebar { box-shadow: 8px 0 24px rgba(15, 23, 42, .07); transition: width 160ms ease; }
.sidebar-resize-handle { position: absolute; right: -3px; top: 0; bottom: 0; width: 6px; cursor: ew-resize; z-index: 4; }
.sidebar-resize-handle:hover { background: color-mix(in srgb, var(--va-primary) 35%, transparent); }
.sidebar-header { min-height: 52px; padding: 8px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--va-background-border); }
.sidebar-header > div { min-width: 0; display: flex; flex-direction: column; }
.sidebar-header strong { font-size: 13px; white-space: nowrap; }
.sidebar-header small { font-size: 10px; color: var(--va-text-secondary); white-space: nowrap; }
.sidebar-toggle { border: 0; background: transparent; color: var(--va-text-primary); cursor: pointer; border-radius: 6px; padding: 5px; }
.sidebar-toggle:hover { background: var(--va-background-secondary); }
.section-caption { padding: 8px 12px 4px; color: var(--va-text-secondary); font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.expanded .compressed-content { overflow-y: auto; }
.expanded .layer-categories-section { flex: none; }
.expanded .category-item { justify-content: flex-start; gap: 8px; min-height: 38px; padding: 5px 9px; }
.expanded .category-item:hover, .expanded .category-item.selected { background: var(--va-background-secondary); }
.category-copy { min-width: 0; display: flex; flex-direction: column; }
.category-copy strong, .category-copy small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.category-copy strong { font-size: 12px; }
.category-copy small { font-size: 10px; color: var(--va-text-secondary); }
.jump-label { font-size: 12px; font-weight: 600; }
.expanded .component-icons, .expanded .preset-icons { align-items: stretch; padding: 0 8px; }
.expanded .component-icon, .expanded .preset-icon { width: 100%; min-height: 36px; height: auto; justify-content: flex-start; gap: 9px; padding: 7px 9px; background: transparent; border-color: transparent; }
.expanded .component-icon:hover, .expanded .preset-icon:hover { background: var(--va-background-secondary); border-color: var(--va-background-border); }
.expanded .component-icon.active { background: color-mix(in srgb, var(--va-primary) 9%, transparent); border-color: color-mix(in srgb, var(--va-primary) 25%, transparent); color: var(--va-primary); }
.expanded .component-icon.disabled { background: transparent; opacity: .58; }
.expanded .component-icon > span:not(.visibility-state), .expanded .preset-icon > span { font-size: 12px; font-weight: 600; }
.visibility-state { margin-left: auto; font-size: 9px; padding: 2px 5px; border-radius: 999px; background: var(--va-background-secondary); color: var(--va-text-secondary); }
.expanded .search-overlay, .expanded .bookmarks-overlay, .expanded .module-filter-overlay { left: var(--sidebar-width); }
.transformer-controls-sidebar:not(.expanded) .sidebar-resize-handle { display: none; }
</style>
