import { defineConfig, type Plugin } from 'vite'
import path from 'node:path'
import fs from 'node:fs/promises'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

/**
 * Dev-only endpoint: POST /__save_jewelry_preset with a JSON body writes
 * it to src/acts/boltz/jewelry-presets.json. The settings panel uses this
 * to persist the user's tweaked look as the new bundled default. Disabled
 * outside `vite` dev (no middleware is registered on build/preview).
 */
function jewelryPresetSavePlugin(): Plugin {
  const target = path.resolve(__dirname, 'src/acts/boltz/jewelry-presets.json')
  return {
    name: 'corundum-jewelry-preset-save',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__save_jewelry_preset', async (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405
          res.end('POST only')
          return
        }
        try {
          let body = ''
          for await (const chunk of req) body += chunk
          const parsed = JSON.parse(body)
          await fs.writeFile(target, JSON.stringify(parsed, null, 2) + '\n', 'utf8')
          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ ok: true, path: target }))
          server.config.logger.info(
            `\x1b[32m[jewelry-preset]\x1b[0m wrote ${path.relative(__dirname, target)}`,
          )
        } catch (e) {
          res.statusCode = 400
          res.end((e as Error).message)
        }
      })
    },
  }
}

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
    jewelryPresetSavePlugin(),
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
