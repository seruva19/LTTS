// @vitest-environment jsdom

import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const visualization = {
  id: 'causal-1',
  timestamp: new Date(),
  module: 'causal_scan',
  type: 'heatmap',
  modulePath: null,
  data: {
    type: 'heatmap',
    values: [[0.5, -0.25], [0.1, 0]],
    dimensions: { rows: 2, cols: 2 },
    row_labels: ['layers.0', 'layers.1'],
    column_labels: ['seed 0', 'seed 1'],
    color_scale: 'diverging_zero'
  }
}

vi.mock('@/state', () => ({
  useLttsState: () => ({
    ws: {
      visualizations: [visualization],
      visualizationEvents: [],
      isInferenceRunning: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    },
    settings: { settings: { controls: {} } },
    ui: { setSelectedLayer: vi.fn() },
    blueprints: { loadedBlueprint: { layers: [] } }
  })
}))

import OutputTimeline from './output-timeline.vue'

describe('output heatmap rendering', () => {
  it('renders causal layer and seed labels with numeric values', async () => {
    const wrapper = mount(OutputTimeline, {
      global: {
        stubs: {
          VaIcon: { template: '<i />' },
          VaButton: { template: '<button><slot /></button>' }
        }
      }
    })

    await wrapper.get('button[title="Expand artifacts panel"]').trigger('click')
    const artifact = wrapper.findAll('button').find(button => button.text().includes('causal_scan'))
    expect(artifact).toBeTruthy()
    await artifact.trigger('click')

    expect(wrapper.text()).toContain('seed 0 · seed 1')
    expect(wrapper.text()).toContain('layers.0')
    expect(wrapper.text()).toContain('0.5000 · -0.2500')
    expect(wrapper.findAll('.artifact-heatmap span')).toHaveLength(4)
  })
})
