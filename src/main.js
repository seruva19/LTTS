import './assets/main.css'
import './assets/notifications.css'

import { createVuestic } from "vuestic-ui"
import "vuestic-ui/css"

import { createApp } from 'vue'
import { createPinia } from 'pinia'


import app from './App.vue'
import ArtifactReplay from './components/artifact-replay.vue'
import { getCurrentTheme } from './theme'
import { useLauncher } from './launch'

const isArtifactReplay = new URLSearchParams(window.location.search).has('artifact-replay')
const application = createApp(isArtifactReplay ? ArtifactReplay : app)

application.use(createPinia())
application.use(createVuestic({ config: { colors: { currentPresetName: getCurrentTheme() } } }))

const mountApplication = () => {
  application.mount('#app')
}

if (isArtifactReplay) {
  mountApplication()
} else {
  const appLauncher = useLauncher()
  appLauncher.initialize().then(mountApplication).catch(error => {
  // Show error in DOM since toast system isn't available yet
  // textContent, not innerHTML: `error.message` can carry server-supplied text.
  const panel = document.createElement('div')
  panel.style.cssText = 'padding: 2rem; color: #dc2626; font-family: system-ui;'
  const heading = document.createElement('h2')
  heading.textContent = 'Failed to initialize application'
  const detail = document.createElement('p')
  detail.textContent = error?.message ?? String(error)
  const hint = document.createElement('p')
  hint.textContent = 'Please check the console for more details.'
  panel.append(heading, detail, hint)
  document.body.replaceChildren(panel)
  console.error('failed to load configuration!', error)
  })
}
