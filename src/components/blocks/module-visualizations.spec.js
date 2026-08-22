// @vitest-environment jsdom
//
// Every payload shape ModuleDataSender can produce must reach a real renderer,
// not the generic JSON fallback. The fallback is legible but useless for an
// image (it prints the base64) and for a chart (it prints the series), so a
// module can look like it "works" end to end while showing nothing usable.

import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

let VISUALIZATIONS = []

vi.mock('@/state', () => ({
  useLttsState: () => ({
    modules: { modules: {}, moduleList: [], isMuted: () => false, toggleMute: vi.fn() },
    ws: {
      getRecentVisualizationsByModule: () => VISUALIZATIONS,
      isConnected: true
    },
    blueprint: { mutedModules: [], toggleModuleMute: vi.fn() }
  })
}))

vi.mock('@/components/ui-renderer/dynamic-form.vue', () => ({
  default: { name: 'DynamicForm', template: '<div />' }
}))

import Module from './module.vue'

const VA_STUBS = {
  VaButton: { template: '<button><slot /></button>' },
  VaIcon: { template: '<i />' },
  VaChip: { template: '<span><slot /></span>' },
  VaCollapse: { template: '<div class="va-collapse"><slot /></div>' },
  VaSelect: true,
  VaInput: true,
  VaSwitch: true,
  VaSlider: true,
  VaCheckbox: true,
  VaProgressCircle: true
}

function render(payloadData, subtype) {
  VISUALIZATIONS = [
    {
      id: 'viz-1',
      timestamp: new Date(),
      module: 'demo_module',
      type: subtype,
      data: payloadData,
      modulePath: 'model.layers.0',
      forwardPass: 1,
      emitMode: 'all'
    }
  ]
  return mount(Module, {
    props: {
      moduleAttachment: { name: 'demo_module', parameters: {} },
      layer: { name: 'model.layers.0', path: 'model.layers.0' }
    },
    global: { stubs: VA_STUBS }
  })
}

// Exactly what each ModuleDataSender.send_* helper puts on the wire.
const PAYLOADS = {
  scalar: { data: { type: 'scalar', value: 1.25, label: 'L2 norm' }, expect: '.scalar-viz' },
  table: {
    data: { type: 'table', headers: ['a', 'b'], rows: [[1, 2]] },
    expect: '.table-viz'
  },
  text: { data: { type: 'text', content: 'hello', preformatted: false }, expect: '.text-viz' },
  heatmap: {
    data: { type: 'heatmap', values: [[0, 1], [1, 0]], dimensions: { rows: 2, cols: 2 } },
    expect: '.heatmap-viz'
  },
  chart: {
    // The shape docs/MODULE_AUTHORING.md documents for send_chart.
    data: {
      type: 'chart',
      chart_type: 'line',
      data: {
        series: [{ name: 'L2', points: [{ x: 0, y: 1 }, { x: 1, y: 2 }] }],
        x_label: 'Layer',
        y_label: 'Norm'
      }
    },
    expect: '.chart-viz'
  },
  image: {
    data: { type: 'image', base64: 'data:image/png;base64,iVBORw0KGgo=' },
    expect: '.image-viz'
  }
}

describe('module visualization rendering', () => {
  for (const [kind, spec] of Object.entries(PAYLOADS)) {
    it(`renders a ${kind} payload with its own renderer`, () => {
      const wrapper = render(spec.data, kind)
      expect(wrapper.find(spec.expect).exists()).toBe(true)
    })
  }

  it('renders chart series as a chart, not as dumped JSON', () => {
    const wrapper = render(PAYLOADS.chart.data, 'chart')
    expect(wrapper.find('.chart-fallback').exists()).toBe(false)
  })

  it('renders an image payload as an <img>, not as dumped base64', () => {
    const wrapper = render(PAYLOADS.image.data, 'image')
    const img = wrapper.find('img.viz-image')
    expect(img.exists()).toBe(true)
    expect(img.attributes('src')).toContain('data:image/png;base64,')
  })
})
