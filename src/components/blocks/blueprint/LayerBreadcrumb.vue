<script setup>

const props = defineProps({
  breadcrumb: {
    type: Array,
    required: true
  }
})

const emit = defineEmits(['filter'])
</script>

<template>
  <div class="layer-breadcrumb">
    <span
      v-for="(crumb, index) in breadcrumb"
      :key="index"
      class="breadcrumb-segment"
      :class="{ 'is-leaf': crumb.isLeaf }"
      @click.stop="emit('filter', crumb.path)"
      :title="'Click to filter by: ' + crumb.path"
    >
      {{ crumb.name }}
      <VaIcon v-if="!crumb.isLeaf" name="chevron_right" size="x-small" class="breadcrumb-separator" />
    </span>
  </div>
</template>

<style scoped>
.layer-breadcrumb {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  margin-bottom: 0.25rem;
  font-size: 0.875rem;
  line-height: 1.2;
}

.breadcrumb-segment {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.125rem 0.375rem;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.15s ease;
  background: rgba(248, 250, 252, 0.8);
  border: 1px solid rgba(203, 213, 225, 0.4);
  color: rgba(51, 65, 85, 0.9);
  font-weight: 500;
}

.breadcrumb-segment:hover {
  background: rgba(241, 245, 249, 0.9);
  border-color: rgba(148, 163, 184, 0.6);
  color: rgba(30, 41, 59, 0.9);
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.breadcrumb-segment.is-leaf {
  background: rgba(59, 130, 246, 0.1);
  border-color: rgba(59, 130, 246, 0.3);
  color: rgba(30, 58, 138, 0.9);
  font-weight: 600;
}

.breadcrumb-segment.is-leaf:hover {
  background: rgba(59, 130, 246, 0.15);
  border-color: rgba(59, 130, 246, 0.5);
}

.breadcrumb-separator {
  opacity: 0.5;
  margin: 0 0.125rem;
}
</style>
