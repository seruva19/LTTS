// Single source of truth for the settings shape.
//
// Client fallback and canonical settings shape until server defaults load.

// Shown as the placeholder in settings and used as the last-resort fallback.
// It is deliberately NOT the default value: pinning it in the personal config
// would outrank a local LTTS-modules checkout.
export const MODULE_REGISTRY_URL =
  'https://raw.githubusercontent.com/seruva19/LTTS-modules/main/registry.json'

export function createDefaultSettings() {
  return {
    storage: {
      save_location: 'localStorage',
      auto_save_settings: true,
      settings_file_path: 'user/config.yaml'
    },
    load_last_blueprint_on_launch: false,
    blueprint: {
      autoload_blueprint: false,
      load_last_blueprint_on_launch: false,
      auto_save_on_changes: true,
      default_blueprint_name: 'Untitled Blueprint',
      save_location: 'localStorage',
      blueprints_folder: 'blueprints/'
    },
    ui: {
      theme: 'auto',
      compact_mode: false,
      show_layer_details: true,
      auto_expand_layers: false
    },
    performance: {
      max_concurrent_models: 3,
      enable_gpu: true,
      memory_limit: 8192,
      cache_size: 1024
    },
    visualization: {
      default_attention_view: 'heatmap',
      show_token_ids: false,
      highlight_threshold: 0.1,
      color_scheme: 'viridis'
    },
    api: {
      server_url: 'http://localhost:8000',
      timeout: 30000,
      retry_attempts: 3
    },
    modules: {
      auto_load_modules: true,
      enable_websocket_updates: true,
      data_retention_minutes: 60,
      registry_url: ''
    },
    display: {
      show_parameter_counts: true,
      show_module_indicators: true,
      compact_mode: false
    },
    controls: {
      show_jump_buttons: true,
      auto_scroll_to_target: true
    }
  }
}

export function deepMerge(target, source) {
  const result = { ...target }
  for (const key in source) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = deepMerge(target[key] || {}, source[key])
    } else {
      result[key] = source[key]
    }
  }
  return result
}
