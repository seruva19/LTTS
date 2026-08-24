import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useResearchStore } from './research-store'

describe('research study state', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('derives ordered layer candidates only from attributable numeric artifacts', () => {
    const store = useResearchStore()
    const layers = [
      { id: 'a', name: 'transformer.h.5' },
      { id: 'b', name: 'transformer.h.8' }
    ]

    const candidates = store.ingestArtifacts([
      { id: 'one', modulePath: 'transformer.h.5', type: 'table', data: { score: 0.21 } },
      { id: 'two', modulePath: 'model.transformer.h.8', type: 'heatmap', data: { effect: -0.73 } },
      { id: 'ignored-text', modulePath: 'transformer.h.5', data: { score: 'not-a-number' } },
      { id: 'ignored-null', modulePath: 'transformer.h.5', data: { score: null } },
      { id: 'ignored-model', modulePath: 'model.output', data: { score: 99 } }
    ], layers)

    expect(candidates).toEqual([
      expect.objectContaining({ layerName: 'transformer.h.8', score: -0.73, artifactId: 'two' }),
      expect.objectContaining({ layerName: 'transformer.h.5', score: 0.21, artifactId: 'one' })
    ])
    expect(store.selectedLayerName).toBe('transformer.h.8')
  })

  it('keeps only the strongest effect for each layer and never invents results', () => {
    const store = useResearchStore()
    const layers = [{ id: 'a', name: 'block.1' }]

    expect(store.ingestArtifacts([], layers)).toEqual([])
    expect(store.lastRunAt).toBeNull()

    store.ingestArtifacts([
      { id: 'weak', modulePath: 'block.1', data: { delta: 0.1 } },
      { id: 'strong', modulePath: 'block.1', data: { delta: -0.4 } }
    ], layers)

    expect(store.candidates).toHaveLength(1)
    expect(store.candidates[0]).toEqual(expect.objectContaining({ artifactId: 'strong', score: -0.4 }))
    expect(store.lastRunAt).not.toBeNull()
  })

  it('opens, closes, and validates stage transitions', () => {
    const store = useResearchStore()
    expect(store.isOpen).toBe(false)
    store.open()
    expect(store.isOpen).toBe(true)
    store.setStage('inspect')
    expect(store.activeStage).toBe('inspect')
    store.setStage('unknown')
    expect(store.activeStage).toBe('inspect')
    store.close()
    expect(store.isOpen).toBe(false)
  })

  it('records immutable snapshots only when a real candidate exists', () => {
    const store = useResearchStore()
    store.manifest = 'facts-v1'
    expect(store.recordRun()).toBeNull()
    store.ingestArtifacts([
      { id: 'effect', modulePath: 'block.1', data: { effect: 0.5 } }
    ], [{ id: 'one', name: 'block.1' }])

    const run = store.recordRun()
    expect(run).toEqual(expect.objectContaining({ manifest: 'facts-v1', split: 'validation' }))
    expect(store.runs).toHaveLength(1)
    store.candidates[0].score = 0.1
    expect(store.runs[0].candidates[0].score).toBe(0.5)
  })

  it('ingests a server study result without fabricating client-side scores', () => {
    const store = useResearchStore()
    const run = store.ingestStudyResult({
      study_id: 'study-1',
      created_at: '2026-08-24T00:00:00Z',
      manifest: { name: 'facts-v1', split: 'test' },
      seeds: [0, 1],
      trials: [{}, {}],
      candidates: [
        { layer: 'transformer.h.7', mean_effect: 0.6, std_effect: 0.1, observations: 2 },
        { layer: 'transformer.h.8', mean_effect: null, observations: 2 }
      ]
    })

    expect(store.candidates).toEqual([
      expect.objectContaining({ layerName: 'transformer.h.7', score: 0.6, observations: 2 })
    ])
    expect(run).toEqual(expect.objectContaining({ id: 'study-1', trialCount: 2 }))
    expect(store.runs).toHaveLength(1)
  })
})
