// Load html2canvas only when capturing a thumbnail.

export function useBlueprintThumbnail() {
  async function captureThumbnail(element, options = {}) {
    const {
      width = 320,
      height = 180,
      quality = 0.8,
      backgroundColor = '#ffffff'
    } = options

    try {
      if (!element) {
        throw new Error('Element not found for thumbnail capture')
      }

      const { default: html2canvas } = await import('html2canvas')
      const canvas = await html2canvas(element, {
        backgroundColor,
        scale: 1,
        logging: false,
        useCORS: true,
        allowTaint: true,
        width: element.offsetWidth,
        height: element.offsetHeight
      })

      // Create a smaller canvas for the thumbnail
      const thumbnailCanvas = document.createElement('canvas')
      thumbnailCanvas.width = width
      thumbnailCanvas.height = height

      const ctx = thumbnailCanvas.getContext('2d')

      // Calculate scaling to maintain aspect ratio
      const scale = Math.min(
        width / canvas.width,
        height / canvas.height
      )

      const scaledWidth = canvas.width * scale
      const scaledHeight = canvas.height * scale

      // Center the image
      const x = (width - scaledWidth) / 2
      const y = (height - scaledHeight) / 2

      // Fill background
      ctx.fillStyle = backgroundColor
      ctx.fillRect(0, 0, width, height)

      // Draw scaled image
      ctx.drawImage(canvas, x, y, scaledWidth, scaledHeight)

      // Convert to data URL
      return thumbnailCanvas.toDataURL('image/jpeg', quality)
    } catch (error) {
      console.error('Failed to capture thumbnail:', error)
      return null
    }
  }

  async function captureBlueprintWorkspace() {
    const workspace = document.querySelector('.layers-container')
    if (!workspace) {
      console.warn('Blueprint workspace not found')
      return null
    }

    return captureThumbnail(workspace, {
      width: 360,
      height: 200,
      quality: 0.85
    })
  }

  function escapeXml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;')
  }

  function generateSVGThumbnail(blueprint) {
    const width = 360
    const height = 200
    const layerCount = blueprint.layers?.length || 0
    const moduleCount = blueprint.moduleCount || 0

    const svg = `
      <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="${width}" height="${height}" fill="#f8f9fa"/>
        <text x="${width / 2}" y="${height / 2 - 20}"
          text-anchor="middle"
          font-family="Arial"
          font-size="24"
          font-weight="bold"
          fill="#2c3e50">
          ${escapeXml(blueprint.name || 'Blueprint')}
        </text>
        <text x="${width / 2}" y="${height / 2 + 10}"
          text-anchor="middle"
          font-family="Arial"
          font-size="14"
          fill="#7f8c8d">
          ${layerCount} layers · ${moduleCount} modules
        </text>
      </svg>
    `

    // Not btoa(): it throws on any code point above U+00FF, so a blueprint
    // named in Cyrillic, CJK or with an emoji would crash thumbnail generation.
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
  }

  return {
    captureThumbnail,
    captureBlueprintWorkspace,
    generateSVGThumbnail
  }
}
