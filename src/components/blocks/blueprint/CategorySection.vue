<script setup>
import { useLayerFormatting } from '@/composables/useLayerFormatting'
import LayerCard from './LayerCard.vue'
import LazyMount from './LazyMount.vue'

const props = defineProps({
  category: {
    type: Object,
    required: true
  },
  categoryId: {
    type: String,
    required: true
  },
  areSectionLabelsVisible: {
    type: Boolean,
    default: true
  },
  isInferenceRunning: {
    type: Boolean,
    default: false
  },
  currentProcessingLayer: {
    type: Number,
    default: -1
  }
})

const { formatNumber } = useLayerFormatting()
</script>

<template>
  <div class="category-section" :data-category="categoryId">
    <div v-show="areSectionLabelsVisible" class="category-header" :style="{ backgroundColor: category.color + '20', borderColor: category.color }">
      <div class="category-title">
        <span class="category-icon">{{ category.icon }}</span>
        <h3>{{ category.name }}</h3>
      </div>
      <div class="category-stats">
        <span class="category-count">{{ category.totalCount }} layers</span>
        <span class="category-params">{{ formatNumber(category.totalParameters) }} params</span>
      </div>
    </div>

    <div v-show="areSectionLabelsVisible" class="category-description">
      {{ category.description }}
    </div>

    <div class="layers-grid">
      <LazyMount v-for="layer in category.layers" :key="layer.name">
        <LayerCard
          :layer="layer"
          :is-inference-running="isInferenceRunning"
          :current-processing-layer="currentProcessingLayer"
        />
      </LazyMount>
    </div>
  </div>
</template>

<style scoped>
.category-section {
  scroll-margin-top: 100px;
}

.category-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  border-radius: 12px;
  border: none;
  margin-bottom: 0.5rem;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.category-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.category-title .category-icon {
  font-size: 1.5rem;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

.category-title h3 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--va-text-primary);
}

.category-stats {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.category-count, .category-params {
  background: rgba(255, 255, 255, 0.8);
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--va-text-primary);
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.category-description {
  margin-bottom: 1rem;
  padding: 0 0.5rem;
  color: var(--va-text-secondary);
  font-size: 0.875rem;
  line-height: 1.4;
  font-style: italic;
}

.layers-grid {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.category-section[data-category="embeddings"] .category-header {
  background: #B0C4DE;
}

.category-section[data-category="attention_mechanisms"] .category-header {
  background: #B0E0E6;
}

.category-section[data-category="transformer_blocks"] .category-header {
  background: #D3D3D3;
}

.category-section[data-category="feed_forward"] .category-header {
  background: #C0C0C0;
}

.category-section[data-category="normalization"] .category-header {
  background: #DCDCDC;
}

.category-section[data-category="output_processing"] .category-header {
  background: #E6E6FA;
}

@media (max-width: 768px) {
  .category-header {
    padding: 0.75rem 1rem;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }

  .category-title h3 {
    font-size: 1.1rem;
  }

  .category-stats {
    gap: 0.5rem;
  }

  .category-count, .category-params {
    font-size: 0.8rem;
    padding: 0.4rem 0.6rem;
  }
}
</style>
