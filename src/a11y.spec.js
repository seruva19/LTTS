// @vitest-environment node
//
// Keyboard access is easy to lose one component at a time: a `@click` on a
// <div> looks and behaves like a button for mouse users while being invisible
// to keyboard and screen-reader users. This is a source scan rather than a
// component test, because the regression it guards against is a new file
// nobody thought to write a test for.

import { describe, expect, it } from 'vitest'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = fileURLToPath(new URL('.', import.meta.url))

function vueFiles(dir) {
  const found = []
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      if (entry === 'node_modules') continue
      found.push(...vueFiles(full))
    } else if (entry.endsWith('.vue')) {
      found.push(full)
    }
  }
  return found
}

// Opening tag of a <div>/<span> up to its closing '>', across newlines.
const TAG = /<(div|span)\b[^>]*>/gs

describe('keyboard accessibility', () => {
  it('no non-interactive element handles @click without a keyboard path', () => {
    const offenders = []

    for (const file of vueFiles(SRC)) {
      const source = readFileSync(file, 'utf-8')
      const template = source.slice(
        source.indexOf('<template>'),
        source.lastIndexOf('</template>')
      )
      if (!template) continue

      for (const match of template.matchAll(TAG)) {
        const tag = match[0]
        // `@click.stop` only stops propagation; it is not a control.
        if (!/@click(?!\.stop)/.test(tag)) continue
        const hasKeyboard = /tabindex/.test(tag) && /@keydown/.test(tag)
        if (!hasKeyboard) {
          const line = template.slice(0, match.index).split('\n').length
          offenders.push(`${file.replace(SRC, '')}:${line} ${tag.slice(0, 60)}`)
        }
      }
    }

    expect(offenders).toEqual([])
  })
})
