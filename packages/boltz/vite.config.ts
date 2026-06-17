import { defineConfig, type Plugin } from 'vite'
import path from 'node:path'
import fs from 'node:fs/promises'
import { execFile } from 'node:child_process'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

/**
 * Dev-only ligand preprocessing endpoint: POST /__smiles_to_blob with
 * { smiles, code? } shells out to scripts/python/smiles_to_blob.py (in the
 * boltz-dev RDKit venv) and returns the Boltz ligand blob as JSON. This is
 * the server-side half of "bring your own SMILES" — RDKit's distance-geometry
 * bounds + ETKDG embedding can't run in-browser, so the heavy-atom blob is
 * computed here once (~tens of ms) and the client featurizes + infers locally.
 *
 * Interpreter is configurable via env BOLTZ_PY; defaults to the local
 * boltz-dev venv. In production this same script runs behind a real endpoint
 * at the same route, so the client code is deployment-agnostic.
 */
const DEFAULT_BOLTZ_PY =
  '/Users/gtaghon/LocalCompute/exclusive_projects/biocircus/boltz-dev/.venv/bin/python'

function smilesToBlobPlugin(): Plugin {
  const py = process.env.BOLTZ_PY || DEFAULT_BOLTZ_PY
  const script = path.resolve(__dirname, 'scripts/python/smiles_to_blob.py')
  return {
    name: 'corundum-smiles-to-blob',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__smiles_to_blob', async (req, res) => {
        const reply = (status: number, body: unknown) => {
          res.statusCode = status
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(body))
        }
        if (req.method !== 'POST') return reply(405, { ok: false, error: 'POST only' })
        try {
          let raw = ''
          for await (const chunk of req) raw += chunk
          const { smiles, code } = JSON.parse(raw) as { smiles?: string; code?: string }
          if (typeof smiles !== 'string' || smiles.trim().length === 0) {
            return reply(400, { ok: false, error: 'missing "smiles"' })
          }
          if (smiles.length > 4000) {
            return reply(400, { ok: false, error: 'SMILES too long (>4000 chars)' })
          }
          // execFile (not a shell) — SMILES is passed as an argv element, so
          // there is no shell-injection surface even with exotic characters.
          const argv = ['--stdout', '--smiles', smiles]
          if (code && /^[A-Za-z0-9]{1,8}$/.test(code)) argv.push('--code', code)
          const out = await new Promise<string>((resolve, reject) => {
            execFile(
              py, [script, ...argv],
              { timeout: 30_000, maxBuffer: 32 * 1024 * 1024 },
              (err, stdout, stderr) => {
                if (err && !stdout) reject(new Error(stderr || err.message))
                else resolve(stdout)
              },
            )
          })
          const parsed = JSON.parse(out)
          reply(parsed.ok ? 200 : 422, parsed)
          server.config.logger.info(
            parsed.ok
              ? `\x1b[32m[smiles-to-blob]\x1b[0m ${parsed.code} (${parsed.data?.num_atoms} atoms)`
              : `\x1b[31m[smiles-to-blob]\x1b[0m ${parsed.error}`,
          )
        } catch (e) {
          reply(500, { ok: false, error: (e as Error).message })
        }
      })
    },
  }
}

/**
 * Dev-only endpoint: POST /__save_jewelry_preset with a JSON body writes
 * it to src/acts/boltz/jewelry-presets.json. The settings panel uses this
 * to persist the user's tweaked look as the new bundled default. Disabled
 * outside `vite` dev (no middleware is registered on build/preview).
 */
function jewelryPresetSavePlugin(): Plugin {
  return makePresetSavePlugin(
    'corundum-jewelry-preset-save',
    '/__save_jewelry_preset',
    'src/acts/boltz/jewelry-presets.json',
    'jewelry-preset',
  )
}

/** Shared body of the preset-save plugins — they only differ by URL +
 *  target file + log tag. */
function makePresetSavePlugin(
  name: string,
  route: string,
  relTarget: string,
  logTag: string,
): Plugin {
  const target = path.resolve(__dirname, relTarget)
  return {
    name,
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(route, async (req, res) => {
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
            `\x1b[32m[${logTag}]\x1b[0m wrote ${path.relative(__dirname, target)}`,
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
    smilesToBlobPlugin(),
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
