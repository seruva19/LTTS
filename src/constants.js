// API + WebSocket base-URL resolution.
//
// Priority: explicit Vite env vars > dev-server default > same-origin.
//
// A *production* build (served by FastAPI from dist/, possibly behind a reverse
// proxy or tunnel — Colab port-proxy, ngrok, cloudflared, an SSH tunnel, …)
// uses RELATIVE / same-origin URLs, so the app works on whatever host it is
// actually served from without a rebuild. The Vite dev server (on :5173) still
// targets the backend on :8000 by default.
//
// Override either value at build time with VITE_LTTS_BASE_URL / VITE_LTTS_HOST.
const _envHost = import.meta.env.VITE_LTTS_HOST
const _envBase = import.meta.env.VITE_LTTS_BASE_URL
const _devHost = 'localhost:8000'

// HTTP base for axios. Empty string => relative requests (`/api/...`), i.e.
// same-origin as the page that loaded the app.
export const baseUrl =
  _envBase ??
  (_envHost
    ? `http://${_envHost}`
    : import.meta.env.DEV
      ? `http://${_devHost}`
      : '')

// Host[:port] used to build the WebSocket URL. In a production build with no
// explicit override, fall back to the page's own host at runtime.
export const wsBaseUrl =
  _envHost ||
  (import.meta.env.DEV
    ? _devHost
    : typeof window !== 'undefined' && window.location.host
      ? window.location.host
      : _devHost)
