/**
 * Model manifest contract.
 *
 * Each act declares the model(s) it depends on as a ModelManifest. The
 * engine layer uses this to fetch + cache weights (OPFS), to negotiate
 * EP selection, and to surface useful diagnostics in the UI
 * ("≈ 47 MB · ESM-2 35M · expects 1×L int64 tokens").
 *
 * Some ONNX files reference an external `.onnx.data` sidecar (Boltz-2's
 * graphs do — the bulk of the weights live in the sidecar, not the .onnx
 * itself). Declare those via `externalDataUrl` + friends; the fetcher
 * downloads both and the worker passes the sidecar bytes to
 * `InferenceSession.create({ externalData: [...] })`.
 */
import type { BackendId } from '../backend'

export interface ModelManifest {
  /** Stable identifier, kebab-case. */
  id: string
  /** Display name shown in act UI. */
  displayName: string
  /** Original-source model citation (paper / repo / authors). */
  provenance: string
  /** URL to the ONNX file. Absolute or relative-to-public. */
  url: string
  /** Approximate downloaded size in bytes (used for UI progress estimation). */
  approxBytes: number
  /** Execution providers in preferred order, e.g., ['webgpu', 'wasm']. */
  executionProviders: BackendId[]
  /** Optional opset version, for compatibility messaging. */
  opset?: number
  /** Optional notes the act wants surfaced ("requires fp16-capable GPU", etc). */
  notes?: string

  // ─────────────────────────────────────────────────────────────────────────
  // External data (sidecar) — set when the .onnx references a separate
  // .onnx.data file that ORT-Web needs alongside it.

  /** URL to the external-data sidecar. Same origin as `url`, typically. */
  externalDataUrl?: string
  /** Approximate sidecar size in bytes. Usually dominates `approxBytes`. */
  externalDataApproxBytes?: number
  /**
   * Path string the .onnx file uses to reference the sidecar. ORT-Web's
   * `externalData` option keys on this path. Typically just the filename
   * (e.g. "trunk_int8.onnx.data").
   */
  externalDataFilename?: string
}

/** True if the manifest declares an external-data sidecar. */
export function hasExternalData(m: ModelManifest): boolean {
  return (
    typeof m.externalDataUrl === 'string' &&
    typeof m.externalDataFilename === 'string'
  )
}

/** Sum of model + sidecar bytes, for total-progress display. */
export function totalApproxBytes(m: ModelManifest): number {
  return m.approxBytes + (m.externalDataApproxBytes ?? 0)
}

/** Type guard: future acts will export their manifest, and we'll iterate these. */
export function isModelManifest(value: unknown): value is ModelManifest {
  if (!value || typeof value !== 'object') return false
  const m = value as Partial<ModelManifest>
  return typeof m.id === 'string' && typeof m.url === 'string' && Array.isArray(m.executionProviders)
}
