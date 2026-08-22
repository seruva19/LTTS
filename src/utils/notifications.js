// Toast notifications without external dependencies.

let notificationContainer = null
let notificationId = 0

// Create notification container
function createContainer() {
  if (notificationContainer) return notificationContainer

  notificationContainer = document.createElement('div')
  notificationContainer.id = 'ltts-notifications'
  notificationContainer.className = 'ltts-notifications-container'
  document.body.appendChild(notificationContainer)

  return notificationContainer
}

// Notification types with their styling
const NOTIFICATION_TYPES = {
  success: {
    icon: '✓',
    className: 'ltts-notification--success',
    defaultDuration: 4000
  },
  error: {
    icon: '⚠',
    className: 'ltts-notification--error',
    defaultDuration: 0 // Manual close only
  },
  warning: {
    icon: '⚡',
    className: 'ltts-notification--warning',
    defaultDuration: 5000
  },
  info: {
    icon: 'ℹ',
    className: 'ltts-notification--info',
    defaultDuration: 2000
  },
  loading: {
    icon: '⟳',
    className: 'ltts-notification--loading',
    defaultDuration: 0 // Manual close only
  }
}

/**
 * Show a notification
 * @param {string} message - The message to display
 * @param {string} type - Type of notification (success, error, warning, info, loading)
 * @param {Object} options - Additional options
 * @returns {Object} Notification object with close method
 */
function showNotification(message, type = 'info', options = {}) {
  const container = createContainer()
  const config = NOTIFICATION_TYPES[type] || NOTIFICATION_TYPES.info
  const id = ++notificationId

  // Create notification element
  const notification = document.createElement('div')
  notification.className = `ltts-notification ${config.className}`
  notification.dataset.id = id

  // Handle error objects
  let displayMessage = message
  if (type === 'error' && message instanceof Error) {
    displayMessage = `${message.name}: ${message.message}`
    if (message.stack) {
      console.error('Detailed error stack:', message.stack)
    }
  } else if (type === 'error' && typeof message === 'object' && message.message) {
    displayMessage = message.message
    if (message.stack) {
      console.error('Detailed error stack:', message.stack)
    }
  }

  // Filter info messages for error objects
  if (type === 'info' && (message instanceof Error || (typeof message === 'object' && message.message))) {
    displayMessage = "Operation in progress..."
  }

  // Built with textContent, not innerHTML: `displayMessage` carries text that
  // originates from WebSocket frames and backend error strings.
  const content = document.createElement('div')
  content.className = 'ltts-notification__content'

  const icon = document.createElement('span')
  icon.className = 'ltts-notification__icon'
  icon.textContent = config.icon

  const text = document.createElement('span')
  text.className = 'ltts-notification__message'
  text.textContent = String(displayMessage)

  content.appendChild(icon)
  content.appendChild(text)
  notification.appendChild(content)

  // Add click to close
  notification.addEventListener('click', () => {
    closeNotification(id)
  })

  // Add to container with animation
  container.appendChild(notification)

  // Trigger animation
  requestAnimationFrame(() => {
    notification.classList.add('ltts-notification--show')
  })

  // Auto-close if duration is set
  const duration = options.duration !== undefined ? options.duration : config.defaultDuration
  let timeoutId = null

  if (duration > 0) {
    timeoutId = setTimeout(() => {
      closeNotification(id)
    }, duration)
  }

  // Return notification object
  return {
    id,
    close: () => closeNotification(id),
    element: notification
  }
}

/**
 * Close a notification by ID
 * @param {number} id - Notification ID
 */
function closeNotification(id) {
  const notification = document.querySelector(`[data-id="${id}"]`)
  if (!notification) return

  notification.classList.add('ltts-notification--hide')

  setTimeout(() => {
    if (notification.parentNode) {
      notification.parentNode.removeChild(notification)
    }
  }, 300)
}

/**
 * Close all notifications
 */
function closeAllNotifications() {
  const notifications = document.querySelectorAll('.ltts-notification')
  notifications.forEach(notification => {
    const id = notification.dataset.id
    if (id) closeNotification(parseInt(id))
  })
}

// Export notification functions
export const notifications = {
  success: (message, options) => showNotification(message, 'success', options),
  error: (message, options) => showNotification(message, 'error', options),
  warning: (message, options) => showNotification(message, 'warning', options),
  info: (message, options) => showNotification(message, 'info', options),
  loading: (message, options) => showNotification(message, 'loading', options),
  close: closeNotification,
  closeAll: closeAllNotifications
}
