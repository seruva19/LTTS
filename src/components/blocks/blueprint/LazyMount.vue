<script setup>
/*
 * Windowed rendering wrapper. Large models expand to thousands of layer cards;
 * mounting them all at once is the dominant GUI cost. LazyMount renders its slot
 * only while the wrapper is near the viewport (within `rootMargin`) and unmounts
 * it otherwise, collapsing to a placeholder of `minHeight`. Unmounting is safe
 * because per-card UI state (expand/collapse) lives in a store keyed by layer
 * name, so it is preserved across remounts.
 */
import { ref, onMounted, onBeforeUnmount } from 'vue'

const props = defineProps({
  // Approximate collapsed card height, used for the placeholder so scrollbar
  // length stays roughly stable for unmounted cards.
  minHeight: { type: Number, default: 72 },
  // Mount cards this far outside the viewport so they are ready before visible.
  rootMargin: { type: String, default: '800px' },
})

const el = ref(null)
const visible = ref(false)
let observer = null

onMounted(() => {
  if (typeof IntersectionObserver === 'undefined') {
    // Render all cards when IntersectionObserver is unavailable.
    visible.value = true
    return
  }
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        visible.value = entry.isIntersecting
      }
    },
    { rootMargin: props.rootMargin },
  )
  if (el.value) observer.observe(el.value)
})

onBeforeUnmount(() => {
  if (observer) {
    observer.disconnect()
    observer = null
  }
})
</script>

<template>
  <div ref="el" :style="visible ? null : { minHeight: minHeight + 'px' }">
    <slot v-if="visible" />
  </div>
</template>
