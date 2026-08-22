<script setup>
import { useLttsState } from '../../state'
import { computed } from 'vue'

const state = useLttsState()
</script>

<template>
  <VaModal class="modal-window" v-model="state.modules.showModulesList" max-height="50vh" ok-text="Close" fixed-layout>
    <h5 class="va-h5">
      Blueprints
    </h5>
    <VaDataTable :items="items">
    </VaDataTable>
  </VaModal>
</template>

<style>

</style>