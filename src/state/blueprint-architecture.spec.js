// @vitest-environment jsdom
//
// The blueprint must carry the model's architecture description. When it did
// not, the UI fell back to topology "unknown" and reported every module that
// declares a supported topology as "Unavailable for this architecture:
// <module> (supports decoder_only)" — on a decoder-only model. The same gap
// made the required-capability check vacuous, because an empty capability map
// never compares `false`.

import { describe, expect, it, vi, beforeEach } from 'vitest'

const post = vi.fn()

vi.mock('axios', () => ({
  default: {
    post: (...args) => post(...args),
    get: vi.fn(),
    delete: vi.fn(),
  },
}))

vi.mock('@/composables/useToast', () => ({
  useToast: () => ({ success: vi.fn(), error: vi.fn(), info: vi.fn() }),
}))

import { createBlueprint } from './blueprint-store'

const MODEL_STRUCT = {
  name: 'model',
  layers: [{ name: 'model.layers.0', moduleRole: 'block' }],
  parameters: 134515008,
  architectureTopology: 'decoder_only',
  architectureFamily: 'llama',
  capabilities: {
    hidden_states: true,
    self_attention: true,
    cross_attention: false,
    unembedding: true,
  },
  categorization: { categories: {} },
}

describe('createBlueprint', () => {
  beforeEach(() => {
    post.mockReset()
    post.mockResolvedValue({ data: { model: JSON.stringify(MODEL_STRUCT) } })
  })

  it('carries the architecture description from the model structure', async () => {
    const blueprint = await createBlueprint({ id: 'HuggingFaceTB/SmolLM-135M' })

    expect(blueprint.architectureTopology).toBe('decoder_only')
    expect(blueprint.architectureFamily).toBe('llama')
    expect(blueprint.capabilities.self_attention).toBe(true)
    // A false capability has to survive: the compatibility check looks for it
    // explicitly, so losing it turns a real incompatibility into a silent pass.
    expect(blueprint.capabilities.cross_attention).toBe(false)
  })

  it('leaves the topology null rather than guessing when the backend omits it', async () => {
    const { architectureTopology: _drop, ...withoutTopology } = MODEL_STRUCT
    post.mockResolvedValue({ data: { model: JSON.stringify(withoutTopology) } })

    const blueprint = await createBlueprint({ id: 'whatever' })

    expect(blueprint.architectureTopology).toBeNull()
    expect(blueprint.capabilities).toEqual(MODEL_STRUCT.capabilities)
  })

  it('accepts architecture metadata nested in legacy categorization payloads', async () => {
    post.mockResolvedValue({
      data: {
        model: JSON.stringify({
          name: 'debug',
          layers: [{ name: 'layers.0' }],
          categorization: {
            categories: {},
            architectureTopology: 'decoder_only',
            architectureFamily: 'debug_transformer',
            capabilities: { hidden_states: true, eager_attention: false },
          },
        }),
      },
    })

    const blueprint = await createBlueprint({ id: 'debug/dummy' })

    expect(blueprint.architectureTopology).toBe('decoder_only')
    expect(blueprint.architectureFamily).toBe('debug_transformer')
    expect(blueprint.capabilities).toEqual({ hidden_states: true, eager_attention: false })
  })
})
