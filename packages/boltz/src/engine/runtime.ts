/**
 * Thin wrapper over ONNX Runtime Web.
 *
 * The act layer should never import 'onnxruntime-web' directly — it goes
 * through this module so we have a single place to:
 *   - configure execution providers and threading,
 *   - wire up logging and error envelopes,
 *   - swap to a worker-hosted session when one is available.
 */
import type { BackendId } from './backend'

let configured = false

export interface RuntimeOptions {
  /** Override EP preference; defaults to ['webgpu', 'wasm']. */
  executionProviders?: BackendId[]
  /** WASM thread count; 0 = auto. Ignored on non-WASM paths. */
  wasmNumThreads?: number
}

/**
 * Lazy-load and configure ORT-Web. Safe to call repeatedly; the first call
 * wins. Returns the ort module so callers can construct sessions.
 */
export async function getRuntime(opts: RuntimeOptions = {}) {
  const ort = await import('onnxruntime-web')
  if (configured) return ort

  const env = ort.env as unknown as {
    wasm: {
      numThreads: number
      simd: boolean
      proxy?: boolean
      wasmPaths?: string | Record<string, string>
    }
    logLevel?: string
  }

  if (typeof opts.wasmNumThreads === 'number') {
    env.wasm.numThreads = opts.wasmNumThreads
  }
  env.wasm.simd = true
  env.logLevel = 'warning'

  // Cloudflare Pages caps individual deployed assets at 25 MiB; the
  // ort-wasm-simd-threaded.jsep binary is 26 MiB and would block upload.
  // Loading it from jsDelivr keeps the deployed bundle slim and works
  // under our COEP:credentialless policy (no credentials on cross-origin
  // fetch). Pin the major.minor to onnxruntime-web's installed version so
  // a registry surprise can't break the runtime.
  env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'

  configured = true
  return ort
}

/**
 * Default EP preference, ordered most-to-least preferred. ORT-Web tries each
 * in turn and falls back when one is unavailable for a given model.
 */
export const DEFAULT_EP_ORDER: BackendId[] = ['webgpu', 'wasm']
