<script setup>
import { computed, onMounted, ref } from 'vue'
import ModuleRenderer from '@/components/blocks/module.vue'
import { useLttsState } from '@/state'

const state = useLttsState()
const audit = ref(null)
const loading = ref(true)
const error = ref('')
const query = ref('')

const moduleFromMessage = (message) => (
  message?.module || message?.data?.module || message?.data?.module_name || null
)

const cards = computed(() => {
  const needle = query.value.trim().toLowerCase()
  return (audit.value?.modules || [])
    .filter((name) => !needle || name.toLowerCase().includes(needle))
    .map((name) => ({
      name,
      attachment: {
        id: `artifact-replay:${name}`,
        name,
        parameters: { enabled: true, emit_mode: 'all' },
      },
    }))
})

const replayArtifacts = (messages, focusedRows) => {
  state.ws.clearVisualizations()
  const focusedMessages = focusedRows
    .flatMap((row) => row.messages || [])
    .map((entry) => entry.message || entry)
  const allMessages = [...focusedMessages, ...messages]

  for (const name of audit.value.modules) {
    const candidates = allMessages.filter((message) => (
      message?.type === 'visualization' && moduleFromMessage(message) === name
    ))
    const preferred = candidates.find((message) => (
      name === 'model_control_vector' && JSON.stringify(message).includes('"trained":true')
    )) || candidates[0]
    if (preferred) {
      state.ws.replayMessage({ token: 'artifact-replay', message: preferred })
    }
  }
}

onMounted(async () => {
  try {
    const [auditResponse, focusedResponse] = await Promise.all([
      fetch('/artifacts/all-modules.json?v=cosine-pca-1', { cache: 'no-store' }),
      fetch('/artifacts/optional-modules.json?v=cosine-pca-1', { cache: 'no-store' }),
    ])
    if (!auditResponse.ok || !focusedResponse.ok) {
      throw new Error('Artifact evidence is unavailable')
    }
    audit.value = await auditResponse.json()
    replayArtifacts(audit.value.messages || [], await focusedResponse.json())
  } catch (reason) {
    error.value = reason?.message || String(reason)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <main class="artifact-replay">
    <header class="replay-header">
      <div>
        <p class="eyebrow">RECORDED LIVE INFERENCE · GPT-2</p>
        <h1>Rendered module artifacts</h1>
        <p class="subtitle">
          Real WebSocket outputs replayed through the same normalization and
          Vue visualization components used during live inference.
        </p>
      </div>
      <div class="summary" v-if="audit">
        <strong>{{ audit.counts.PASS }}/{{ audit.modules.length }}</strong>
        <span>modules passed</span>
      </div>
    </header>

    <div class="replay-toolbar">
      <input v-model="query" type="search" placeholder="Filter modules" />
      <a href="/artifacts/all-modules.json">Raw evidence JSON</a>
      <a href="/">Back to live app</a>
    </div>

    <div v-if="loading" class="state-panel">Loading recorded artifacts…</div>
    <div v-else-if="error" class="state-panel error">{{ error }}</div>
    <section v-else class="artifact-grid">
      <ModuleRenderer
        v-for="card in cards"
        :key="card.name"
        :module-attachment="card.attachment"
        :layer="{}"
        position="post"
      />
    </section>
  </main>
</template>

<style scoped>
.artifact-replay{height:100vh;overflow-y:auto;overscroll-behavior:contain;padding:32px;background:radial-gradient(circle at top,#182238 0,var(--va-background-primary) 42%);color:var(--va-text-primary)}
.replay-header{max-width:1500px;margin:0 auto 22px;display:flex;justify-content:space-between;align-items:end;gap:24px}
.eyebrow{margin:0 0 7px;color:var(--va-primary);font-size:12px;font-weight:800;letter-spacing:.14em}.replay-header h1{margin:0;font-size:clamp(30px,5vw,56px);line-height:1}.subtitle{max-width:780px;color:var(--va-text-secondary)}
.summary{display:flex;flex-direction:column;align-items:end}.summary strong{font-size:32px;color:var(--va-success)}.summary span{color:var(--va-text-secondary)}
.replay-toolbar{max-width:1500px;margin:0 auto 18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}.replay-toolbar input{min-width:280px;padding:10px 13px;border:1px solid var(--va-background-border);border-radius:8px;background:var(--va-background-element);color:var(--va-text-primary)}.replay-toolbar a{color:var(--va-primary)}
.artifact-grid{max-width:1500px;margin:auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px;align-items:start}.state-panel{max-width:1500px;margin:auto;padding:24px;border:1px solid var(--va-background-border);border-radius:10px;background:var(--va-background-element)}.error{color:var(--va-danger)}
@media(max-width:600px){.artifact-replay{padding:18px}.replay-header{align-items:start;flex-direction:column}.summary{align-items:start}.artifact-grid{grid-template-columns:1fr}}
</style>
