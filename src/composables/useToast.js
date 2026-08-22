import { notifications } from '@/utils/notifications'

/**
 * Toast composable for LTTS application
 * Provides a unified interface for showing notifications
 */
export function useToast() {
  /**
   * Show a success toast
   * @param {string} message - The message to display
   * @param {Object} options - Additional options
   */
  const success = (message, options = {}) => {
    return notifications.success(message, options)
  }

  /**
   * Show an error toast
   * @param {string|Error} message - The message to display or Error object
   * @param {Object} options - Additional options
   */
  const error = (message, options = {}) => {
    return notifications.error(message, options)
  }

  /**
   * Show a warning toast
   * @param {string} message - The message to display
   * @param {Object} options - Additional options
   */
  const warning = (message, options = {}) => {
    return notifications.warning(message, options)
  }

  /**
   * Show an info toast
   * @param {string} message - The message to display
   * @param {Object} options - Additional options
   */
  const info = (message, options = {}) => {
    return notifications.info(message, options)
  }

  /**
   * Show a loading toast
   * @param {string} message - The message to display
   * @param {Object} options - Additional options
   */
  const loading = (message, options = {}) => {
    return notifications.loading(message, options)
  }

  /**
   * Show a custom toast with full control
   * @param {string} message - The message to display
   * @param {string} type - The type of toast (success, error, warning, info)
   * @param {Object} options - Additional options
   */
  const custom = (message, type = 'info', options = {}) => {
    return notifications[type] ? notifications[type](message, options) : notifications.info(message, options)
  }

  return {
    success,
    error,
    warning,
    info,
    loading,
    custom,
    close: notifications.close,
    closeAll: notifications.closeAll
  }
}
