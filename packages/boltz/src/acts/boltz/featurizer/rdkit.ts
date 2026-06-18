/**
 * Lazy in-browser RDKit (WASM) loader.
 *
 * Loads the corundum fork of RDKit MinimalLib (`@corundum/rdkit`), which adds
 * the `get_boltz_blob(smiles)` op (ETKDGv3 embedding + distance-geometry bounds
 * + the six constraint groups Boltz-2 needs). This replaces the Python
 * `/__smiles_to_blob` serverlet — "bring your own SMILES" now runs fully
 * client-side.
 *
 * The package is the versioned source of truth (a normal dependency). Its
 * built artifacts are synced into `public/rdkit/` by `scripts/sync-rdkit.mjs`
 * (run on dev/build) and served from a stable path. We deliberately do NOT
 * `?url`-import the .wasm: that would emit it into `dist/assets/`, which the
 * prod build strips (`rm -f dist/assets/*.wasm`, there to drop ORT's wasm).
 * Files under `public/` are copied to the dist root and survive.
 *
 * The glue (.js) is loaded via a one-time <script> injection + `locateFile`,
 * which is the load path RDKit's emscripten MODULARIZE output is built for.
 */
import type { RDKitModule } from '@corundum/rdkit'

// `window.initRDKitModule` is declared globally by @corundum/rdkit's types.

const RDKIT_JS_URL = '/rdkit/RDKit_minimal.js'
const RDKIT_WASM_URL = '/rdkit/RDKit_minimal.wasm'

let modPromise: Promise<RDKitModule> | null = null

function injectScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`)
    if (existing) {
      if (existing.dataset.loaded === '1') return resolve()
      existing.addEventListener('load', () => resolve())
      existing.addEventListener('error', () => reject(new Error(`failed to load ${src}`)))
      return
    }
    const s = document.createElement('script')
    s.src = src
    s.async = true
    s.addEventListener('load', () => {
      s.dataset.loaded = '1'
      resolve()
    })
    s.addEventListener('error', () => reject(new Error(`failed to load ${src}`)))
    document.head.appendChild(s)
  })
}

async function load(): Promise<RDKitModule> {
  if (!window.initRDKitModule) {
    await injectScript(RDKIT_JS_URL)
  }
  if (!window.initRDKitModule) {
    throw new Error('RDKit failed to register initRDKitModule')
  }
  return window.initRDKitModule({ locateFile: () => RDKIT_WASM_URL })
}

/**
 * Get the (singleton) RDKit module, loading + instantiating it on first use.
 * Concurrent callers share one in-flight load. A failed load is not cached, so
 * the next call retries.
 */
export function getRDKit(): Promise<RDKitModule> {
  if (!modPromise) {
    modPromise = load().catch((e) => {
      modPromise = null
      throw e
    })
  }
  return modPromise
}
