// @vitest-environment jsdom

import { mount, shallowMount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import LayerCard from './LayerCard.vue'
import ModelInfoHeader from './ModelInfoHeader.vue'
import ModelLevelModules from '../model-level-modules.vue'

const removeModuleFromLayer = vi.fn()

vi.mock('@/state', () => ({
  useLttsState: () => ({
    modules: {
      moduleList: [{ name: 'residual_stream_norm' }],
      isLoading: false,
      modules: {
        model_control_vector: {
          metadata: { module_type: 'model_level' },
          attachment_contract: {
            model_level: true,
            supported_topologies: ['decoder_only'],
            required_capabilities: [],
            optional_dependencies: []
          }
        },
        memory_monitor: {
          metadata: { module_type: 'model_level' },
          attachment_contract: {
            model_level: true,
            supported_topologies: ['encoder_only', 'decoder_only'],
            required_capabilities: [],
            optional_dependencies: []
          }
        }
      }
    },
    blueprints: {
      loadedBlueprint: {
        architectureTopology: 'encoder_only',
        capabilities: { hidden_states: true },
        modelModules: []
      },
      addPreModuleToLayer: vi.fn(),
      addPostModuleToLayer: vi.fn(),
      removeModuleFromLayer,
      addModelModule: vi.fn(),
      removeModelModule: vi.fn(),
      updateModelModuleSettings: vi.fn()
    }
  })
}))

describe('transformer architecture presentation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows topology and supported capabilities in the model header', () => {
    const wrapper = mount(ModelInfoHeader, {
      props: {
        blueprint: {
          name: 'T5 research model',
          architecture: 'T5 · Encoder Decoder',
          architectureFamily: 't5',
          architectureTopology: 'encoder_decoder',
          capabilities: {
            encoder: true,
            decoder: true,
            self_attention: true,
            cross_attention: true,
            seq2seq_lm: true,
            moe: false
          },
          layers: [],
          totalParameters: 1024
        },
        totalModules: 4
      }
    })

    expect(wrapper.text()).toContain('T5 · Encoder Decoder')
    expect(wrapper.text()).toContain('Encoder')
    expect(wrapper.text()).toContain('Decoder')
    expect(wrapper.text()).toContain('Cross-attention')
    expect(wrapper.text()).toContain('Seq2seq LM')
    expect(wrapper.text()).not.toContain('MoE')
  })

  it('marks structural nodes clearly and hides unsafe attachment controls', () => {
    const wrapper = shallowMount(LayerCard, {
      props: {
        layer: {
          id: 'decoder-projection',
          name: 'decoder.layers.0.self_attn.q_proj',
          type: 'Linear',
          moduleRole: 'attention_projection',
          architectureStage: 'decoder',
          layerIndex: 0,
          attentionKind: 'self',
          attachable: false,
          preModules: [],
          postModules: []
        }
      },
      global: {
        provide: {
          layerState: {
            isExpanded: () => true,
            isSelected: () => false,
            toggleExpanded: vi.fn(),
            preModuleSelection: {},
            postModuleSelection: {}
          },
          architectureContext: {
            topology: 'encoder_decoder',
            capabilities: {}
          },
          preModuleOptions: [],
          postModuleOptions: [],
          handleLayerHeaderClick: vi.fn(),
          showContextMenu: vi.fn(),
          getLayerIndex: () => 0,
          filterByBreadcrumb: vi.fn()
        },
        stubs: {
          VaAlert: { name: 'VaAlert', template: '<div><slot /></div>' },
          VaBadge: { name: 'VaBadge', template: '<span />' },
          VaButton: { name: 'VaButton', template: '<button />' },
          VaIcon: { name: 'VaIcon', template: '<i />' },
          VaSelect: { name: 'VaSelect', template: '<select />' }
        }
      }
    })

    expect(wrapper.classes()).toContain('not-attachable')
    expect(wrapper.text()).toContain('attention projection')
    expect(wrapper.text()).toContain('decoder')
    expect(wrapper.text()).toContain('structure only')
    expect(wrapper.text()).toContain('not a safe hook target')
    expect(wrapper.findAllComponents({ name: 'VaSelect' })).toHaveLength(0)
  })

  it('filters attachment options by topology, capabilities, and dependencies', () => {
    const wrapper = shallowMount(LayerCard, {
      props: {
        layer: {
          id: 'encoder-block',
          name: 'bert.encoder.layer.0',
          type: 'BertLayer',
          moduleRole: 'block',
          architectureStage: 'encoder',
          layerIndex: 0,
          attachable: true,
          preModules: [],
          postModules: []
        }
      },
      global: {
        provide: {
          layerState: {
            isExpanded: () => true,
            isSelected: () => false,
            toggleExpanded: vi.fn(),
            preModuleSelection: {},
            postModuleSelection: {}
          },
          architectureContext: {
            topology: 'encoder_only',
            capabilities: {
              hidden_states: true,
              gradients: false
            }
          },
          preModuleOptions: [],
          postModuleOptions: [
            {
              text: 'residual_stream_norm',
              value: 'residual_stream_norm',
              targetRoles: ['block'],
              supportedTopologies: ['encoder_only', 'decoder_only'],
              requiredCapabilities: ['hidden_states'],
              optionalDependencies: []
            },
            {
              text: 'sae_features',
              value: 'sae_features',
              targetRoles: ['block'],
              supportedTopologies: ['decoder_only'],
              requiredCapabilities: ['hidden_states'],
              optionalDependencies: []
            },
            {
              text: 'gradient_importance',
              value: 'gradient_importance',
              targetRoles: ['block'],
              supportedTopologies: ['encoder_only'],
              requiredCapabilities: ['gradients'],
              optionalDependencies: []
            },
            {
              text: 'optional_probe',
              value: 'optional_probe',
              targetRoles: ['block'],
              supportedTopologies: ['encoder_only'],
              requiredCapabilities: [],
              optionalDependencies: [{ package: 'probe_pkg', available: false }]
            }
          ],
          handleLayerHeaderClick: vi.fn(),
          showContextMenu: vi.fn(),
          getLayerIndex: () => 0,
          filterByBreadcrumb: vi.fn()
        },
        stubs: {
          VaAlert: { name: 'VaAlert', template: '<div><slot /></div>' },
          VaBadge: { name: 'VaBadge', template: '<span />' },
          VaButton: { name: 'VaButton', template: '<button />' },
          VaIcon: { name: 'VaIcon', template: '<i />' },
          VaSelect: {
            name: 'VaSelect',
            props: ['options'],
            template: '<select><option v-for="option in options" :key="option.value">{{ option.text }}</option></select>'
          }
        }
      }
    })

    const selects = wrapper.findAllComponents({ name: 'VaSelect' })
    expect(selects).toHaveLength(2)
    expect(selects[1].props('options').map(option => option.value)).toEqual([
      'residual_stream_norm'
    ])
    expect(wrapper.text()).toContain('sae_features (supports decoder_only)')
    expect(wrapper.text()).toContain('gradient_importance (missing gradients)')
    expect(wrapper.text()).toContain('optional_probe (install probe_pkg)')
  })

  it('does not fabricate a default model module and filters model-level choices', () => {
    const wrapper = shallowMount(ModelLevelModules, {
      global: {
        stubs: {
          VaAlert: { template: '<div><slot /></div>' },
          VaButton: { template: '<button><slot /></button>' },
          VaInput: true,
          VaModal: { template: '<div><slot /><slot name="footer" /></div>' },
          VaSelect: true,
          VaSlider: true,
          VaSwitch: true,
          VaTextarea: true
        }
      }
    })

    expect(wrapper.text()).toContain('No model-level modules attached')
    expect(wrapper.vm.availableModelModules.map(option => option.value)).toEqual([
      'memory_monitor'
    ])
    expect(wrapper.vm.unavailableModelModules).toEqual([
      {
        name: 'model_control_vector',
        compatible: false,
        reason: 'supports decoder_only'
      }
    ])
  })
})
