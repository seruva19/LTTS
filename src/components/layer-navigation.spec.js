// @vitest-environment node
//
// Document-level keyboard shortcuts must never fire while the user is typing.
// The navigation panel binds Space/Enter/j/k on `document` and calls
// preventDefault(); without a focus guard those keys vanish from every text
// field in the app, so a prompt typed into Model Input arrived as
// "ThecapitalofFranceis".

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const SOURCES = [
  '../composables/useLayerNavigation.js',
  'transformer-controls.vue',
  'model-log-panel.vue',
].map((name) => ({
  name,
  source: readFileSync(fileURLToPath(new URL(`./${name}`, import.meta.url)), 'utf-8'),
}))

describe('global keyboard shortcuts', () => {
  for (const { name, source } of SOURCES) {
    it(`${name} ignores key events aimed at a text field`, () => {
      const bindsGlobally = /(document|window)\.addEventListener\(\s*['"]keydown/.test(source)
      if (!bindsGlobally) return

      // Only handlers that swallow the event need the guard.
      if (!/preventDefault\(\)/.test(source)) return

      const guards = /activeElement/.test(source)
      expect(
        guards,
        `${name} binds a document-level keydown handler that calls ` +
          'preventDefault() without checking document.activeElement'
      ).toBe(true)
    })
  }
})
