export function useLayerFormatting() {
  const formatNumber = (num) => {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B'
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M'
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K'
    return num?.toLocaleString() || '0'
  }

  const formatShape = (shape) => {
    if (!shape || !Array.isArray(shape)) return 'N/A'
    return `[${shape.join(', ')}]`
  }

  const formatMemory = (bytes) => {
    if (!bytes || bytes === 0) return 'N/A'
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`
    return `${bytes} B`
  }

  const formatPrecision = (dtype) => {
    if (!dtype) return 'N/A'
    const precisionMap = {
      'torch.float32': 'FP32',
      'torch.float16': 'FP16',
      'torch.bfloat16': 'BF16',
      'torch.int8': 'INT8',
      'torch.int32': 'INT32',
      'torch.int64': 'INT64',
      'float32': 'FP32',
      'float16': 'FP16',
      'bfloat16': 'BF16',
      'int8': 'INT8',
      'int32': 'INT32',
      'int64': 'INT64'
    }
    return precisionMap[dtype] || dtype
  }

  const formatActivationStats = (stats) => {
    if (!stats) return null
    return {
      mean: stats.mean?.toFixed(4) || 'N/A',
      std: stats.std?.toFixed(4) || 'N/A',
      min: stats.min?.toFixed(4) || 'N/A',
      max: stats.max?.toFixed(4) || 'N/A'
    }
  }

  const getLayerTypeInfo = (layer) => {
    const type = layer.type?.toLowerCase() || ''
    const name = layer.name?.toLowerCase() || ''

    if (type.includes('attention') || type.includes('attn') || name.includes('attention')) {
      return { badge: 'ATTN', color: 'primary', icon: '👁️' }
    } else if (type.includes('sequential') || type.includes('feed') || type.includes('ffn') || type.includes('mlp') || name.includes('feed_forward')) {
      return { badge: 'FFN', color: 'success', icon: '🔗' }
    } else if (type.includes('layernorm') || type.includes('rmsnorm') || name.includes('norm')) {
      return { badge: 'NORM', color: 'warning', icon: '📏' }
    } else if (type.includes('embedding') || type.includes('embed') || name.includes('embed')) {
      return { badge: 'EMB', color: 'info', icon: '📝' }
    } else if (type.includes('moe') || type.includes('expert') || name.includes('moe')) {
      return { badge: 'MoE', color: 'danger', icon: '🔀' }
    } else if (type.includes('transformer') || name.includes('layers.')) {
      return { badge: 'TFRM', color: 'secondary', icon: '🔄' }
    } else if (type.includes('linear') || name.includes('lm_head')) {
      return { badge: 'LM', color: 'info', icon: '🎯' }
    }
    return { badge: 'LAYER', color: 'secondary', icon: '⚡' }
  }

  return {
    formatNumber,
    formatShape,
    formatMemory,
    formatPrecision,
    formatActivationStats,
    getLayerTypeInfo
  }
}
