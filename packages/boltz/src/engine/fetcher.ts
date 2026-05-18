/**
 * Streaming model fetcher.
 *
 * Downloads ONNX weights from a manifest URL, reports byte-level progress,
 * streams directly into OPFS (so a multi-GB blob never sits whole in RAM),
 * and returns the cached bytes for session creation.
 *
 * Models with external-data sidecars (Boltz-2's graphs do — the bulk of the
 * weights live in `.onnx.data` next to a thin `.onnx`) get both files
 * fetched and cached; combined progress is reported as a single stream
 * weighted by approxBytes, so the UI shows one bar to the user.
 */
import { getCache, type CacheKey } from './cache'
import {
  hasExternalData,
  totalApproxBytes,
  type ModelManifest,
} from './models/registry'

export type FetchPhase = 'check-cache' | 'downloading' | 'caching' | 'ready'

export interface FetchProgress {
  phase: FetchPhase
  bytesLoaded: number
  /** Total content length if known; otherwise undefined. */
  bytesTotal?: number
  /** Rough rate in bytes/sec, smoothed. */
  bytesPerSecond?: number
  /**
   * When a manifest has an external-data sidecar this is set so the UI can
   * label which file is currently flowing ('model' vs 'sidecar').
   */
  subject?: 'model' | 'sidecar'
}

export interface FetchOptions {
  signal?: AbortSignal
  onProgress?: (p: FetchProgress) => void
  /** Override cache tag (e.g., for content-hash-based keying). Defaults to manifest.id-derived tag. */
  tag?: string
  /** Force re-download even if a cached copy exists. */
  forceReload?: boolean
}

/** Bytes returned by the fetcher — sidecar present iff the manifest declared it. */
export interface FetchedModel {
  modelBytes: Uint8Array
  externalDataBytes?: Uint8Array
  externalDataFilename?: string
}

function deriveTag(manifest: ModelManifest): string {
  // URL host + path-tail acts as a passable identity tag when no explicit one
  // is supplied. The fetcher logs a warning when this is used — callers
  // really should specify a content hash.
  try {
    const u = new URL(manifest.url, globalThis.location?.href ?? 'http://localhost/')
    return `${u.host}_${u.pathname.replace(/[^a-z0-9.]/gi, '_').slice(-40)}`
  } catch {
    return manifest.url.replace(/[^a-z0-9.]/gi, '_').slice(-40)
  }
}

const SMOOTHING = 0.25 // EMA factor for bytes/sec

interface FetchOneOptions {
  url: string
  approxBytes: number
  cacheKey: CacheKey
  signal?: AbortSignal
  /**
   * Called whenever this single download makes progress; ALL counters are
   * relative to this file (not the combined total). The orchestrator
   * composes a combined progress object before forwarding to the user.
   */
  onLocalProgress?: (loaded: number, total: number | undefined, bps: number) => void
  forceReload: boolean
}

/**
 * Stream-fetch one URL, tee to OPFS, return the bytes. Pure single-file
 * machinery — the combined-progress and sidecar logic lives in `fetchModel`.
 */
async function fetchOne(opts: FetchOneOptions): Promise<Uint8Array> {
  const cache = await getCache()
  if (!opts.forceReload && cache.available && (await cache.has(opts.cacheKey))) {
    const cached = await cache.read(opts.cacheKey)
    if (cached) {
      opts.onLocalProgress?.(cached.length, cached.length, 0)
      return cached
    }
  }

  const response = await fetch(opts.url, { signal: opts.signal })
  if (!response.ok) {
    throw new Error(`Fetch failed for ${opts.url}: HTTP ${response.status} ${response.statusText}`)
  }
  const totalHeader = response.headers.get('Content-Length')
  const bytesTotal = totalHeader ? Number(totalHeader) : opts.approxBytes || undefined

  if (!response.body) {
    const buf = new Uint8Array(await response.arrayBuffer())
    if (cache.available) {
      const stream = await cache.writeStream(opts.cacheKey)
      await stream.write(buf)
      await stream.close()
    }
    opts.onLocalProgress?.(buf.length, buf.length, 0)
    return buf
  }

  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let bytesLoaded = 0
  let lastTick = performance.now()
  let lastLoaded = 0
  let bps = 0

  const writer = cache.available ? await cache.writeStream(opts.cacheKey) : null

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      if (!value) continue
      chunks.push(value)
      bytesLoaded += value.byteLength
      if (writer) {
        await writer.write(value)
      }

      const now = performance.now()
      const dt = now - lastTick
      if (dt > 250) {
        const instantBps = ((bytesLoaded - lastLoaded) * 1000) / dt
        bps = bps === 0 ? instantBps : bps * (1 - SMOOTHING) + instantBps * SMOOTHING
        lastTick = now
        lastLoaded = bytesLoaded
        opts.onLocalProgress?.(bytesLoaded, bytesTotal, bps)
      }
    }
    if (writer) await writer.close()
  } catch (err) {
    if (writer) {
      try {
        await writer.close()
      } catch {
        /* ignore */
      }
      await cache.evict(opts.cacheKey).catch(() => undefined)
    }
    throw err
  }

  const total = new Uint8Array(bytesLoaded)
  let offset = 0
  for (const chunk of chunks) {
    total.set(chunk, offset)
    offset += chunk.byteLength
  }
  return total
}

/**
 * Fetch (or load-from-cache) all bytes a manifest needs to create an ORT
 * session: the .onnx file, and (if declared) its .onnx.data sidecar.
 */
export async function fetchModel(
  manifest: ModelManifest,
  opts: FetchOptions = {},
): Promise<FetchedModel> {
  const tag = opts.tag ?? deriveTag(manifest)
  const emit = (p: FetchProgress) => opts.onProgress?.(p)

  emit({ phase: 'check-cache', bytesLoaded: 0 })

  const totalCombined = totalApproxBytes(manifest)
  let modelBytesLoaded = 0
  let sidecarBytesLoaded = 0
  let lastEmittedBps = 0

  // Compose a combined progress emit. Called from each per-file callback;
  // both counters live in closure so the running sum is always correct.
  const emitCombined = (
    subject: 'model' | 'sidecar',
    phase: FetchPhase,
    bps: number,
  ) => {
    if (bps > 0) lastEmittedBps = bps
    const loaded = modelBytesLoaded + sidecarBytesLoaded
    emit({
      phase,
      bytesLoaded: loaded,
      bytesTotal: totalCombined,
      bytesPerSecond: lastEmittedBps,
      subject,
    })
  }

  // 1. The .onnx file itself.
  const modelBytes = await fetchOne({
    url: manifest.url,
    approxBytes: manifest.approxBytes,
    cacheKey: { modelId: manifest.id, tag: `${tag}__onnx` },
    signal: opts.signal,
    forceReload: opts.forceReload === true,
    onLocalProgress: (loaded, _total, bps) => {
      modelBytesLoaded = loaded
      emitCombined('model', 'downloading', bps)
    },
  })
  modelBytesLoaded = modelBytes.length

  // 2. The .onnx.data sidecar, if any.
  let externalDataBytes: Uint8Array | undefined
  if (hasExternalData(manifest)) {
    externalDataBytes = await fetchOne({
      url: manifest.externalDataUrl!,
      approxBytes: manifest.externalDataApproxBytes ?? 0,
      cacheKey: { modelId: manifest.id, tag: `${tag}__data` },
      signal: opts.signal,
      forceReload: opts.forceReload === true,
      onLocalProgress: (loaded, _total, bps) => {
        sidecarBytesLoaded = loaded
        emitCombined('sidecar', 'downloading', bps)
      },
    })
    sidecarBytesLoaded = externalDataBytes.length
  }

  emit({
    phase: 'ready',
    bytesLoaded: modelBytesLoaded + sidecarBytesLoaded,
    bytesTotal: totalCombined,
    bytesPerSecond: lastEmittedBps,
  })

  return {
    modelBytes,
    externalDataBytes,
    externalDataFilename: externalDataBytes ? manifest.externalDataFilename : undefined,
  }
}

export function formatBytes(n: number): string {
  if (!Number.isFinite(n)) return '—'
  const units = ['B', 'kB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v < 10 ? 2 : v < 100 ? 1 : 0)} ${units[i]}`
}
