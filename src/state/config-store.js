import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'
import { baseUrl } from '../constants'
import { useToast } from '@/composables/useToast'

export const useConfigStore = defineStore('config', () => {
  const config = ref({})
  const isLoading = ref(false)
  const error = ref(null)
  const toast = useToast()

  async function loadConfig() {
    isLoading.value = true
    error.value = null
    try {
      const response = await axios.get(`${baseUrl}/api/config`)
      config.value = response.data
      toast.success('Configuration loaded')
    } catch (err) {
      error.value = `Failed to load configuration: ${err.message}`
      toast.error(err) // Pass full error object for detailed info
    } finally {
      isLoading.value = false
    }
  }

  async function reloadConfig() {
    isLoading.value = true
    error.value = null
    try {
      const response = await axios.post(`${baseUrl}/api/config/reload`)
      config.value = response.data
      toast.success('Configuration reloaded')
    } catch (err) {
      error.value = `Failed to reload configuration: ${err.message}`
      toast.error(err) // Pass full error object for detailed info
    } finally {
      isLoading.value = false
    }
  }

  return { config, isLoading, error, loadConfig, reloadConfig }
})
