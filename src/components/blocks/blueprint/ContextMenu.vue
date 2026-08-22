<script setup>

const props = defineProps({
  contextMenu: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['addBookmark', 'removeBookmark', 'selectLayer', 'toggleExpanded'])
</script>

<template>
  <div
    v-if="contextMenu.visible"
    class="context-menu"
    :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }"
    role="menu"
    @click.stop
  >
    <div
      class="context-menu-item"
      role="menuitem"
      tabindex="0"
      @click="emit('addBookmark')"
      @keydown.enter.prevent="emit('addBookmark')"
      @keydown.space.prevent="emit('addBookmark')"
    >
      <VaIcon name="bookmark_add" size="small" />
      <span>Add Bookmark</span>
    </div>
    <div
      class="context-menu-item"
      role="menuitem"
      tabindex="0"
      @click="emit('removeBookmark')"
      @keydown.enter.prevent="emit('removeBookmark')"
      @keydown.space.prevent="emit('removeBookmark')"
    >
      <VaIcon name="bookmark_remove" size="small" />
      <span>Remove Bookmark</span>
    </div>
    <div class="context-menu-divider" role="separator"></div>
    <div
      class="context-menu-item"
      role="menuitem"
      tabindex="0"
      @click="emit('selectLayer', contextMenu.layerName)"
      @keydown.enter.prevent="emit('selectLayer', contextMenu.layerName)"
      @keydown.space.prevent="emit('selectLayer', contextMenu.layerName)"
    >
      <VaIcon name="visibility" size="small" />
      <span>Select Layer</span>
    </div>
    <div
      class="context-menu-item"
      role="menuitem"
      tabindex="0"
      @click="emit('toggleExpanded', contextMenu.layerName)"
      @keydown.enter.prevent="emit('toggleExpanded', contextMenu.layerName)"
      @keydown.space.prevent="emit('toggleExpanded', contextMenu.layerName)"
    >
      <VaIcon name="expand_more" size="small" />
      <span>Toggle Details</span>
    </div>
  </div>
</template>

<style scoped>
.context-menu {
  position: fixed;
  background: var(--va-background-element);
  border: 1px solid var(--va-background-border);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 9999;
  min-width: 180px;
  padding: 0.5rem 0;
}

.context-menu-item:focus-visible {
  outline: 2px solid var(--va-primary, #3b82f6);
  outline-offset: -2px;
}

.context-menu-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  cursor: pointer;
  transition: background-color 0.15s ease;
  font-size: 0.875rem;
  color: var(--va-text-primary);
}

.context-menu-item:hover {
  background-color: var(--va-background-secondary);
}

.context-menu-divider {
  height: 1px;
  background-color: var(--va-background-border);
  margin: 0.5rem 0;
}
</style>
