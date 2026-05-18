import { defineConfig } from 'vite'
import path from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

/**
 * Cross-origin isolation headers.
 *
 * Required for SharedArrayBuffer, which threaded ONNX Runtime Web needs to
 * spin up worker pools.
 *
 * COEP `credentialless` rather than `require-corp` because most model CDNs
 * (HuggingFace, S3, raw GitHub) do not ship `Cross-Origin-Resource-Policy:
 * cross-origin`. `credentialless` keeps the page isolated, satisfies the
 * SharedArrayBuffer requirement, and permits cross-origin no-credentials
 * fetches of public model weights — exactly the access pattern Corundum
 * wants. Browser support: Chrome 96+, Firefox 124+, Safari 17.5+.
 */
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
} as const

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // Self-signed HTTPS for LAN testing. Chromium only exposes WebGPU and
    // SharedArrayBuffer in secure contexts, and http://<lan-ip> doesn't
    // qualify even on private IPs — other devices need https://<lan-ip>.
    basicSsl(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: true,
    headers: crossOriginIsolation,
  },
  preview: {
    host: true,
    headers: crossOriginIsolation,
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // ORT-Web ships a hefty WASM blob — let Vite pre-bundle the JS but exclude
    // the worker copy so it can resolve its asset URLs at runtime.
    exclude: ['onnxruntime-web'],
  },
})
