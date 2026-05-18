/**
 * Engine worker.
 *
 * The single worker that owns every InferenceSession in the app. The main
 * thread never imports onnxruntime-web directly; everything flows through
 * Comlink. Acts get a slim hook that hides the proxy.
 *
 * Surface:
 *   probe()                       → BackendReport
 *   warmRuntime()                 → { version }
 *   loadSession(manifest, onProgress)
 *                                 → SessionHandle
 *   run(handle, feeds)            → outputs
 *   dispose(handle)               → void
 *   listSessions()                → SessionHandle[]
 */
import * as Comlink from 'comlink'
import type * as ort from 'onnxruntime-web'
import { probeBackends, type BackendReport } from './backend'
import { getRuntime } from './runtime'
import { fetchModel, type FetchProgress } from './fetcher'
import type { ModelManifest } from './models/registry'

/**
 * Serializable mirror of ORT's `InferenceSession.ValueMetadata`. We keep
 * just the fields we care about so the value crosses Comlink cleanly
 * (the underlying ORT types are not necessarily structured-clonable).
 */
export interface IOMetadataEntry {
  name: string
  isTensor: boolean
  /** Tensor.Type string (e.g. 'float32', 'float16', 'int64'). Undefined when isTensor=false. */
  dtype?: string
  /** Mixed shape: number for fixed dims, string for symbolic ones. Undefined when isTensor=false. */
  shape?: ReadonlyArray<number | string>
}

export interface SessionHandle {
  id: string
  modelId: string
  inputNames: readonly string[]
  outputNames: readonly string[]
  inputMetadata: readonly IOMetadataEntry[]
  outputMetadata: readonly IOMetadataEntry[]
  executionProvider: string
  /** Bytes the model occupied as a Uint8Array prior to session compilation. */
  approxBytes: number
}

export interface TensorPayload {
  data:
    | Float32Array
    | Float64Array
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array
  dims: readonly number[]
  /** ORT tensor type; inferred from `data` if omitted on write. */
  type?: ort.Tensor.Type
}

export type FeedDict = Record<string, TensorPayload>
export type FetchEvent = FetchProgress | { phase: 'compiling' }

interface SessionEntry {
  handle: SessionHandle
  session: ort.InferenceSession
}

const sessions = new Map<string, SessionEntry>()
let handleCounter = 0

/**
 * Convert ORT's ValueMetadata array (which may carry non-plain getter
 * properties) into a plain serializable shape so Comlink can ship it
 * across the worker boundary. Shape elements are normalised to
 * `number | string` directly.
 */
function toSerializableMetadata(
  src: readonly ort.InferenceSession.ValueMetadata[],
): IOMetadataEntry[] {
  return src.map((m) => {
    if (m.isTensor) {
      return {
        name: m.name,
        isTensor: true,
        dtype: m.type,
        shape: m.shape.map((d) => (typeof d === 'number' ? d : String(d))),
      }
    }
    return { name: m.name, isTensor: false }
  })
}

function inferType(data: TensorPayload['data']): ort.Tensor.Type {
  if (data instanceof Float32Array) return 'float32'
  if (data instanceof Float64Array) return 'float64'
  if (data instanceof Int8Array) return 'int8'
  if (data instanceof Uint8Array) return 'uint8'
  if (data instanceof Int16Array) return 'int16'
  if (data instanceof Uint16Array) return 'uint16'
  if (data instanceof Int32Array) return 'int32'
  if (data instanceof BigInt64Array) return 'int64'
  if (data instanceof BigUint64Array) return 'uint64'
  throw new Error('Unsupported tensor data type')
}

const api = {
  async probe(): Promise<BackendReport> {
    return probeBackends()
  },

  async warmRuntime(): Promise<{ version: string }> {
    const ortMod = await getRuntime()
    const env = (ortMod as unknown as { env?: { versions?: { common?: string } } }).env
    return { version: env?.versions?.common ?? 'unknown' }
  },

  async loadSession(
    manifest: ModelManifest,
    onProgress?: (event: FetchEvent) => void,
  ): Promise<SessionHandle> {
    const fetched = await fetchModel(manifest, {
      onProgress: onProgress
        ? (p) => {
            onProgress(p)
          }
        : undefined,
    })

    onProgress?.({ phase: 'compiling' })

    const ortMod = await getRuntime()

    // ORT-Web takes external-data sidecars via the `externalData` session
    // option — an array of {data, path} pairs. The `path` must match the
    // filename string the .onnx references internally.
    const sessionOpts: ort.InferenceSession.SessionOptions = {
      executionProviders: manifest.executionProviders.map((id) => id),
      graphOptimizationLevel: 'all',
    }
    if (fetched.externalDataBytes && fetched.externalDataFilename) {
      ;(sessionOpts as ort.InferenceSession.SessionOptions & {
        externalData?: Array<{ data: Uint8Array; path: string }>
      }).externalData = [
        {
          data: fetched.externalDataBytes,
          path: fetched.externalDataFilename,
        },
      ]
    }

    const session = await ortMod.InferenceSession.create(fetched.modelBytes, sessionOpts)

    const id = `s${++handleCounter}__${manifest.id}`
    const handle: SessionHandle = {
      id,
      modelId: manifest.id,
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      inputMetadata: toSerializableMetadata(session.inputMetadata),
      outputMetadata: toSerializableMetadata(session.outputMetadata),
      executionProvider: manifest.executionProviders[0] ?? 'unknown',
      approxBytes:
        fetched.modelBytes.byteLength +
        (fetched.externalDataBytes?.byteLength ?? 0),
    }
    sessions.set(id, { handle, session })
    return handle
  },

  async run(
    handleId: string,
    feeds: FeedDict,
  ): Promise<Record<string, TensorPayload>> {
    const entry = sessions.get(handleId)
    if (!entry) throw new Error(`No session for handle ${handleId}`)
    const ortMod = await getRuntime()
    const inputs: Record<string, ort.Tensor> = {}
    for (const [name, payload] of Object.entries(feeds)) {
      const type = payload.type ?? inferType(payload.data)
      inputs[name] = new ortMod.Tensor(type, payload.data, [...payload.dims])
    }
    const results = await entry.session.run(inputs)
    const out: Record<string, TensorPayload> = {}
    for (const [name, tensor] of Object.entries(results)) {
      out[name] = {
        data: tensor.data as TensorPayload['data'],
        dims: tensor.dims,
        type: tensor.type,
      }
    }
    return out
  },

  async dispose(handleId: string): Promise<void> {
    const entry = sessions.get(handleId)
    if (!entry) return
    await entry.session.release()
    sessions.delete(handleId)
  },

  async listSessions(): Promise<SessionHandle[]> {
    return Array.from(sessions.values()).map((s) => s.handle)
  },
}

export type EngineWorkerApi = typeof api

// Guard against the file being accidentally imported from the main thread.
// Comlink.expose with a window endpoint would still work but would also
// expose this API to the whole page — not what we want.
const isWorker =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== 'undefined' &&
  typeof (self as unknown as { window?: unknown }).window === 'undefined'

if (isWorker) {
  Comlink.expose(api)
}
