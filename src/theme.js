function extractThemePreference(settings) {
  const rawTheme = settings?.ui?.theme
  if (typeof rawTheme === 'string') {
    return rawTheme
  }
  if (rawTheme && typeof rawTheme === 'object' && typeof rawTheme.value === 'string') {
    return rawTheme.value
  }
  return null
}

export const getCurrentTheme = () => {
  try {
    const storedSettings = localStorage.getItem('ltts-settings')
    if (storedSettings) {
      const parsed = JSON.parse(storedSettings)
      const storedTheme = extractThemePreference(parsed)
      if (storedTheme === 'light' || storedTheme === 'dark') {
        return storedTheme
      }
    }
  } catch {
    // Fall back to system preference when persisted settings are unavailable.
  }

  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }

  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    return 'light'
  }

  return 'light'
}
