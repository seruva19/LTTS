// @vitest-environment jsdom

import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/state', () => ({
  useLttsState: () => globalThis.__LTTS_RESEARCH_TEST_STATE__
}))

vi.mock('@/composables/useLayerNavigation', () => ({
  scrollToLayerByName: vi.fn()
}))

vi.mock('axios', () => ({
  default: { post: vi.fn() }
}))

import axios from 'axios'
import ResearchStudyPanel from './research-study-panel.vue'
import { useResearchStore } from '@/state/research-store'

const artifactCollection = (name, value) => ({
  schema_version: 1,
  name,
  manifest_fingerprint: 'sha256:dataset',
  metadata: {},
  records: [
    {
      schema_version: 1,
      artifact_id: `${name}-artifact`,
      metric: 'effect',
      value,
      module: 'probe',
      subtype: 'scalar',
      layer_path: 'layers.0',
      component: null,
      token_index: null,
      token: null,
      seed: 0,
      example_id: 'prompt',
      stage: 'scan',
      run_id: name,
      metadata: {}
    }
  ]
})

const studyResult = (id, value) => ({
  study_id: id,
  created_at: id === 'baseline'
    ? '2026-08-24T00:00:00Z'
    : '2026-08-24T00:00:01Z',
  manifest: { name: 'comparison', split: 'validation' },
  seeds: [0],
  trials: [{}],
  candidates: [{ layer: 'layers.0', mean_effect: value, observations: 1 }],
  artifacts: artifactCollection(id, value)
})

describe('Research Study run comparison', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    axios.post.mockReset()
    globalThis.__LTTS_RESEARCH_TEST_STATE__ = {
      blueprints: {
        loadedBlueprint: {
          source: { id: 'debug/dummy' },
          layers: [
            {
              id: 'layer-0',
              name: 'layers.0',
              fullPath: 'model.layers.0',
              preModules: [],
              postModules: []
            }
          ]
        }
      },
      modules: { attachedModules: [], isMuted: () => false },
      ws: {
        visualizations: [],
        isInferenceRunning: false,
        clientToken: 'test-client',
        replayMessage: vi.fn()
      },
      ui: {
        userPrompt: 'hello',
        setSelectedLayer: vi.fn()
      }
    }
  })

  it('keeps comparison collapsed, calls the shared API, and emits one table artifact', async () => {
    const research = useResearchStore()
    research.ingestStudyResult(studyResult('baseline', 0.2))
    research.ingestStudyResult(studyResult('candidate', 0.5))
    axios.post.mockResolvedValue({
      data: {
        schema_version: 1,
        baseline_name: 'baseline',
        candidate_name: 'candidate',
        manifest_match: true,
        points: [
          {
            metric: 'effect',
            module: 'probe',
            subtype: 'scalar',
            layer_path: 'model.layers.0',
            status: 'matched',
            baseline_mean: 0.2,
            candidate_mean: 0.5,
            delta: 0.3,
            matched_trials: 1,
            paired_delta_std: 0
          }
        ]
      }
    })

    const wrapper = mount(ResearchStudyPanel, {
      global: {
        stubs: { VaIcon: { template: '<i />' } }
      }
    })

    const details = wrapper.find('details.research-compare')
    expect(details.exists()).toBe(true)
    expect(details.attributes('open')).toBeUndefined()
    details.element.open = true
    await details.trigger('toggle')
    const compareButton = wrapper.get('button.research-compare-run')
    expect(compareButton.attributes('disabled')).toBeUndefined()

    await compareButton.trigger('click')
    await flushPromises()

    expect(axios.post).toHaveBeenCalledWith(
      expect.stringContaining('/api/research/studies/compare'),
      expect.objectContaining({
        baseline: expect.objectContaining({ name: 'baseline' }),
        candidate: expect.objectContaining({ name: 'candidate' })
      })
    )
    expect(research.activeStage).toBe('compare')
    expect(research.comparisonForLayer('layers.0').delta).toBe(0.3)
    expect(globalThis.__LTTS_RESEARCH_TEST_STATE__.ws.replayMessage).toHaveBeenCalledTimes(1)
    expect(globalThis.__LTTS_RESEARCH_TEST_STATE__.ws.replayMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'visualization',
        module: 'research_compare',
        subtype: 'table'
      })
    )
    expect(wrapper.text()).toContain('+0.300')
  })

  it('keeps causal scan collapsed and submits a paired intervention sweep', async () => {
    const layer = globalThis.__LTTS_RESEARCH_TEST_STATE__.blueprints.loadedBlueprint.layers[0]
    layer.postModules = [
      {
        name: 'activation_anisotropy',
        parameters: { emit_scalar: true }
      },
      {
        name: 'neuron_ablation',
        parameters: { enabled: true, neuron_indices: '0' }
      }
    ]
    globalThis.__LTTS_RESEARCH_TEST_STATE__.modules.modules = {
      activation_anisotropy: {
        metadata: { description: 'Activation metric', event_types: ['layer_after'] }
      },
      neuron_ablation: {
        metadata: { description: 'Causal ablation intervention', event_types: ['layer_after'] }
      }
    }
    axios.post.mockResolvedValue({
      data: {
        schema_version: 1,
        scan_id: 'scan-1',
        metric: 'value',
        points: [
          {
            layer: 'model.layers.0',
            score: 1.2,
            delta_from_baseline: 0.2,
            delta_std: 0,
            observations: 1,
            seed_deltas: [{ seed: 0, delta: 0.2 }]
          }
        ]
      }
    })

    const wrapper = mount(ResearchStudyPanel, {
      global: { stubs: { VaIcon: { template: '<i />' } } }
    })
    const details = wrapper.get('details.research-causal-scan')
    expect(details.attributes('open')).toBeUndefined()
    details.element.open = true
    await details.trigger('toggle')

    const button = details.get('button.research-compare-run')
    expect(button.attributes('disabled')).toBeUndefined()
    await button.trigger('click')
    await flushPromises()

    expect(axios.post).toHaveBeenCalledWith(
      expect.stringContaining('/api/research/causal-scan'),
      expect.objectContaining({
        layers: ['model.layers.0'],
        intervention: expect.objectContaining({
          name: 'neuron_ablation',
          parameters: expect.objectContaining({ enabled: true, neuron_indices: '0' })
        }),
        measurement_module: 'activation_anisotropy',
        metric: 'value'
      })
    )
    expect(useResearchStore().causalForLayer('layers.0').delta_from_baseline).toBe(0.2)
    expect(wrapper.text()).toContain('+0.200')
    expect(globalThis.__LTTS_RESEARCH_TEST_STATE__.ws.replayMessage).not.toHaveBeenCalled()
  })
})
