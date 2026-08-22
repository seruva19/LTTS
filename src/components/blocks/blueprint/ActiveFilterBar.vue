<script setup>

const props = defineProps({
  activeFilter: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['clear'])
</script>

<template>
  <div v-if="activeFilter" class="active-filter-bar">
    <div class="filter-info">
      <VaIcon name="filter_alt" size="small" />
      <span>Filtering by: <strong>{{ activeFilter }}</strong></span>
    </div>
    <VaButton
      preset="plain"
      size="small"
      icon="close"
      @click="emit('clear')"
      class="clear-filter-btn"
    />
  </div>
</template>

<style scoped>
.active-filter-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 6px;
  margin: 1rem 0;
}

.filter-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: rgba(6, 78, 59, 0.9);
  font-size: 0.875rem;
  font-weight: 500;
}

.clear-filter-btn {
  color: rgba(6, 78, 59, 0.7);
}

.clear-filter-btn:hover {
  color: rgba(6, 78, 59, 0.9);
  background: rgba(16, 185, 129, 0.1);
}
</style>
