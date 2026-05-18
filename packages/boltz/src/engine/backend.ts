/**
 * Backend capability probe.
 *
 * Reports which ONNX Runtime Web execution providers can actually light up on
 * this device, surfacing the kind of detail a Rain Computing user wants to
 * see: which GPU, what adapter, why a backend fell back, threading state.
 */

export type BackendId = 'webgpu' | 'webnn' | 'wasm'

export interface BackendCapability {
  id: BackendId
  available: boolean
  /** Short label suitable for a chip / badge. */
  label: string
  /** Long-form human description. */
  detail: string
  /** True if this is the EP we'd pick first. Only one capability is preferred. */
  preferred: boolean
}

export interface WasmEnvironment {
  threads: boolean
  simd: boolean
  crossOriginIsolated: boolean
  hardwareConcurrency: number
}

export interface BackendReport {
  capabilities: BackendCapability[]
  selected: BackendId | null
  wasm: WasmEnvironment
  ort: { version: string }
  probedAt: number
}

// ────────────────────────────────────────────────────────────────────────────
// Individual probes

interface GpuAdapterLike {
  info?: { vendor?: string; architecture?: string; device?: string; description?: string }
  features?: ReadonlySet<string>
  limits?: Record<string, number>
}

async function probeWebGPU(): Promise<Omit<BackendCapability, 'preferred'>> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return {
      id: 'webgpu',
      available: false,
      label: 'WebGPU · unavailable',
      detail: 'navigator.gpu is not defined in this context.',
    }
  }
  try {
    const adapter = (await (navigator as unknown as {
      gpu: { requestAdapter: () => Promise<GpuAdapterLike | null> }
    }).gpu.requestAdapter()) ?? null
    if (!adapter) {
      return {
        id: 'webgpu',
        available: false,
        label: 'WebGPU · no adapter',
        detail: 'navigator.gpu present but no adapter was returned (likely no compatible GPU).',
      }
    }
    const info = adapter.info ?? {}
    const adapterLabel =
      [info.vendor, info.architecture, info.device].filter(Boolean).join(' · ') ||
      info.description ||
      'unnamed adapter'
    return {
      id: 'webgpu',
      available: true,
      label: `WebGPU · ${adapterLabel}`,
      detail: `Adapter ${adapterLabel}. ${adapter.features?.size ?? 0} features advertised.`,
    }
  } catch (err) {
    return {
      id: 'webgpu',
      available: false,
      label: 'WebGPU · error',
      detail: `requestAdapter threw: ${(err as Error).message}`,
    }
  }
}

function probeWebNN(): Omit<BackendCapability, 'preferred'> {
  const hasML = typeof navigator !== 'undefined' && 'ml' in navigator
  if (!hasML) {
    return {
      id: 'webnn',
      available: false,
      label: 'WebNN · unavailable',
      detail: 'navigator.ml is not defined. WebNN is still shipping behind flags on most browsers.',
    }
  }
  return {
    id: 'webnn',
    available: true,
    label: 'WebNN · present',
    detail: 'navigator.ml detected. Device-specific NPU acceleration may be available.',
  }
}

function probeWasm(): {
  capability: Omit<BackendCapability, 'preferred'>
  env: WasmEnvironment
} {
  const threads =
    typeof SharedArrayBuffer !== 'undefined' &&
    typeof globalThis.crossOriginIsolated !== 'undefined' &&
    globalThis.crossOriginIsolated === true
  const env: WasmEnvironment = {
    threads,
    simd: typeof WebAssembly !== 'undefined',
    crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    hardwareConcurrency:
      typeof navigator !== 'undefined' && 'hardwareConcurrency' in navigator
        ? navigator.hardwareConcurrency
        : 1,
  }
  return {
    capability: {
      id: 'wasm',
      available: true,
      label: threads ? `WASM · threaded (${env.hardwareConcurrency} cores)` : 'WASM · single-thread',
      detail: threads
        ? `Cross-origin isolated; threads + SharedArrayBuffer available across ${env.hardwareConcurrency} cores.`
        : 'Cross-origin isolation not enabled, so threaded WASM is off. Single-thread WASM still works.',
    },
    env,
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Combined report

export async function probeBackends(): Promise<BackendReport> {
  const [webgpu, webnn, wasm] = await Promise.all([
    probeWebGPU(),
    Promise.resolve(probeWebNN()),
    Promise.resolve(probeWasm()),
  ])

  // Pick preferred EP: WebGPU > WebNN > WASM (matches ENGINE.md guidance —
  // WebGPU is broadly fastest today, WebNN is the future Apple-NE / NPU path,
  // WASM is the universal fallback).
  const ordered = [webgpu, webnn, wasm.capability]
  const firstAvailable = ordered.find((c) => c.available)?.id ?? null

  const capabilities: BackendCapability[] = ordered.map((c) => ({
    ...c,
    preferred: c.id === firstAvailable,
  }))

  // Lazy-load ORT only to read its version string. The full runtime stays
  // out of the critical path until an act actually wants to run a model.
  let ortVersion = 'unknown'
  try {
    const ort = await import('onnxruntime-web')
    ortVersion = (ort as unknown as { env?: { versions?: { common?: string } } }).env?.versions?.common ?? 'unknown'
  } catch {
    /* leave 'unknown' */
  }

  return {
    capabilities,
    selected: firstAvailable,
    wasm: wasm.env,
    ort: { version: ortVersion },
    probedAt: Date.now(),
  }
}
