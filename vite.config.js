import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'

// `command` is 'build' for production output and 'serve' for the dev server.
export default defineConfig(({ command }) => ({
  plugins: [
    vue(),
    vueJsx(),
  ],
  build: {
    rolldownOptions: {
      treeshake: command === 'build'
        ? { manualPureFunctions: ['console.debug'] }
        : undefined,
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return
          }
          if (id.includes('node_modules/vuestic-ui')) {
            return 'ui-vendor'
          }
          if (
            id.includes('node_modules/html2canvas') ||
            id.includes('node_modules/@svgdotjs')
          ) {
            return 'viz-vendor'
          }
          // Anchor on node_modules: matching a bare '/vue/' caught only the
          // `vue` package shim, leaving @vue/runtime-core, @vue/reactivity and
          // @vue/shared - the actual bulk of the framework - in 'vendor'.
          if (
            id.includes('node_modules/vue/') ||
            id.includes('node_modules/@vue/') ||
            id.includes('node_modules/pinia/') ||
            id.includes('node_modules/vue-toastification/')
          ) {
            return 'vue-vendor'
          }
          return 'vendor'
        }
      }
    }
  },
  // Production debug calls are marked pure in Rolldown's treeshaker above.
  // `warn` and `error` stay: they are what a user pastes into a bug report.
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
}))
